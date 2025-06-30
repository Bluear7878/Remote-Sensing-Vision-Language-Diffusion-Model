import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from PIL import Image

# SR3 specific imports
import configs.sr3 as SR3
import data.dataset as SR_Dataset
import models.sr3_model as sr3_model
import utils.logger as Logger
import utils.tensor2img as T2I
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from models.util import *


@dataclass
class PipelineConfig:
    """A dataclass to centrally manage all pipeline configurations."""
    # --- Path Settings ---
    input_img: str
    output_dir: str = "./results"
    model_yaml: str = "./model_configs/juggernautXL.yaml"
    prompt_yaml: str = "./prompts/prompt_config.yaml"

    # --- Device Settings ---
    sr_model_device: str = "cuda:0"
    base_model_device: str = "cuda:1"

    # --- Pre-processing & SR3 Settings ---
    upscale_factor: int = 8

    # --- SUPIR Refinement Settings ---
    a_prompt: str = ("Cinematic, High Contrast, highly detailed aerial photo taken using a high-resolution drone or satellite, "
                     "hyper detailed photo-realistic maximum detail, 32k, Color Grading, ultra HD, "
                     "extreme meticulous detailing of terrain textures and structures, hyper sharpness, no deformations.")
    n_prompt: str = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, "
                     "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
                     "signature, jpeg artifacts, deformed, lowres, over-smooth, cloud cover, heavy fog, motion blur, lens flare")

    # --- Sampling Hyperparameters ---
    min_size: int = 1024
    edm_steps: int = 50
    s_churn: int = 5
    s_noise: float = 1.003
    s_cfg: float = 7.5
    s_stage1: int = -1
    s_stage2: float = 1.0
    img_threshold: float = 0.3
    seed: int = -1
    num_samples: int = 1
    color_fix_type: str = "Wavelet"
    linear_cfg: bool = True
    linear_s_stage2: bool = False
    spt_linear_cfg: float = 4.0
    spt_linear_s_stage2: float = 0.0

    # --- Model Loading Settings ---
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"
    no_llava: bool = False
    use_tile_vae: bool = False
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64
    load_8bit_llava: bool = False

    def __post_init__(self):
        """After initialization, converts paths to Path objects and creates the output directory."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = Path(self.input_img)
        self.filename = self.input_path.stem


class SuperResolutionPipeline:
    """
    Encapsulates the Super-Resolution pipeline.
    Improves efficiency by loading models only once.
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.sr3_model = None
        self.llava_model = None
        self.llava_tokenizer = None
        self.llava_image_processor = None
        self.refinement_model = None

        self._load_all_models()

    def _load_all_models(self):
        """Loads all models required for the pipeline."""
        print("Loading all models...")
        self._load_sr3_model()
        if not self.cfg.no_llava:
            self._load_llava_model()
        self._load_refinement_model()
        print("All models loaded successfully.")

    def _load_sr3_model(self):
        """Loads the SR3 model."""
        sr3_args = SR3.SR3_Config()
        sr3_opt = Logger.parse(sr3_args)
        self.sr3_model = sr3_model.create_model(sr3_opt)
        self.sr3_model.set_new_noise_schedule(
            sr3_opt['model']['beta_schedule']['val'], schedule_phase='val'
        )

    def _load_llava_model(self):
        """Loads the LLaVA model and its related components."""
        self.llava_tokenizer, self.llava_model, self.llava_image_processor = load_llava(
            device=self.cfg.base_model_device
        )

    def _load_refinement_model(self):
        """Loads the refinement model."""
        self.refinement_model = create_SR_model(self.cfg.model_yaml)
        self.refinement_model.to(self.cfg.sr_model_device)

    def run_stage1_sr3_upscale(self, image_path: Path) -> Image.Image:
        """Stage 1: Performs initial upscaling using the SR3 model."""
        print(f"Running SR3 upscale for {image_path.name}...")
        loader = SR_Dataset.dataloader(str(image_path), self.cfg.upscale_factor)
        # The dataloader contains only one image, so we use iter to get it directly.
        val_data = next(iter(loader))

        self.sr3_model.feed_data(val_data)
        self.sr3_model.test(continous=True)

        sr_tensor = self.sr3_model.SR
        if sr_tensor.dim() == 4:
            sr_tensor = sr_tensor[-1]

        sr_img_np = T2I.tensor2img(sr_tensor, min_max=(-1, 1))
        sr_pil = Image.fromarray(sr_img_np)

        output_path = self.cfg.output_dir / f"sr3_{self.cfg.filename}.png"
        sr_pil.save(output_path)
        print(f"SR3 upscaled image saved to {output_path}")
        return sr_pil

    def run_stage2_captioning(self, sr_image: Image.Image) -> str:
        """Stage 2: Generates an image caption using the LLaVA model."""
        if self.cfg.no_llava or self.llava_model is None:
            print("Skipping LLaVA captioning.")
            return ""

        print("Generating image caption with LLaVA...")
        with open(self.cfg.prompt_yaml, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        img_prompt = prompts["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)

        image_tensor = process_images([sr_image], self.llava_image_processor, self.llava_model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.cfg.base_model_device) for _image in image_tensor]

        caption = get_img_describe(
            image_tensor=image_tensor, image=sr_image, model=self.llava_model,
            tokenizer=self.llava_tokenizer, prompt=img_prompt,
            max_new_tokens=256, conv_templates=conv_templates,
            image_token_index=IMAGE_TOKEN_INDEX, device=self.cfg.base_model_device
        )
        print(f"Generated Caption: {caption}")
        return caption

    def run_stage3_refinement(self, sr_image: Image.Image, caption: str):
        """Stage 3: Performs final SR refinement using the model."""
        print("Running refinement...")
        lq_img, h0, w0 = PIL2Tensor(sr_image, upscale=1, min_size=self.cfg.min_size)
        lq_img = lq_img.unsqueeze(0).to(self.cfg.sr_model_device)[:, :3, :, :]

        # Organize parameters for the sampling function into a dictionary.
        sampling_params = {
            "num_steps": self.cfg.edm_steps,
            "restoration_scale": self.cfg.s_stage1,
            "s_churn": self.cfg.s_churn,
            "s_noise": self.cfg.s_noise,
            "cfg_scale": self.cfg.s_cfg,
            "control_scale": self.cfg.s_stage2,
            "seed": self.cfg.seed,
            "num_samples": self.cfg.num_samples,
            "p_p": self.cfg.a_prompt,
            "n_p": self.cfg.n_prompt,
            "color_fix_type": self.cfg.color_fix_type,
            "use_linear_CFG": self.cfg.linear_cfg,
            "use_linear_control_scale": self.cfg.linear_s_stage2,
            "cfg_scale_start": self.cfg.spt_linear_cfg,
            "control_scale_start": self.cfg.spt_linear_s_stage2,
            "img_threshold": self.cfg.img_threshold,
            "dec_img": 1,
        }

        sample_function = getattr(self.refinement_model, "just_sampling", None)
        if not callable(sample_function):
            raise RuntimeError("`just_sampling` function not found in the refinement model.")

        samples = sample_function(lq_img, caption, **sampling_params)

        for i, sample in enumerate(samples):
            output_path = self.cfg.output_dir / f"{self.cfg.filename}_final_{i}.png"
            Tensor2PIL(sample, h0, w0).save(output_path)
            print(f"Final result saved to: {output_path}")

    def process(self):
        """Runs the entire Super-Resolution pipeline."""
        # Stage 1: SR3
        sr3_image = self.run_stage1_sr3_upscale(self.cfg.input_path)

        # Stage 2: Captioning
        caption = self.run_stage2_captioning(sr3_image)

        # Stage 3: Refinement
        self.run_stage3_refinement(sr_image=sr3_image, caption=caption)


def main():
    parser = argparse.ArgumentParser(description="Advanced Super-Resolution Pipeline")
    parser.add_argument("--input_img", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the output images.")
    parser.add_argument("--upscale_factor", type=int, default=8, help="Initial upscaling factor for SR3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--img_threshold", type=float, default=0.3, help="Image threshold for refinement.")
    parser.add_argument("--edm_steps", type=int, default=50, help="Number of EDM sampling steps.")

    args = parser.parse_args()

    # Create config object from dataclass and override with CLI arguments
    config = PipelineConfig(
        input_img=args.input_img,
        output_dir=args.output_dir,
        upscale_factor=args.upscale_factor,
        seed=args.seed,
        img_threshold=args.img_threshold,
        edm_steps=args.edm_steps,
    )

    pipeline = SuperResolutionPipeline(config)
    pipeline.process()


if __name__ == "__main__":
    main()
