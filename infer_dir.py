#!/usr/bin/env python
# env : llama3_metrics
# python3 infer_dir.py --image_dir "LR/dataset/dir" --save_dir "./results" --img_threshold 0.4
# coding: utf-8

import argparse
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from PIL import Image
from Texture_eval_mk import *
from tqdm import tqdm

import configs.sr3 as SR3
import data.dataset as SR_Dataset
import models.sr3_model as sr3_model
import utils.logger as Logger
import utils.tensor2img as T2I
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from models.dataloader import *
from models.util import *


@dataclass
class BatchInferenceConfig:
    """A dataclass to manage all configurations for batch inference."""
    # --- Path and Device Settings ---
    image_dir: str
    save_dir: str = "./results"
    model_yaml: str = "./model_configs/juggernautXL.yaml"
    prompt_yaml: str = "./prompts/prompt_config.yaml"
    sr_device: str = "cuda:0"
    base_device: str = "cuda:1"

    # --- Processing Parameters ---
    upscale: int = 8
    min_size: int = 1024
    num_steps: int = 50
    num_samples: int = 1
    seed: int = 1234

    # --- Prompts ---
    a_prompt: str = (
        "Cinematic, High Contrast, highly detailed aerial photo taken using "
        "a high-resolution drone or satellite, hyper detailed photo-realistic maximum detail, "
        "32k, Color Grading, ultra HD, extreme meticulous detailing of terrain textures and structures, "
        "hyper sharpness, no deformations."
    )
    n_prompt: str = (
        "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, "
        "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
        "signature, jpeg artifacts, deformed, lowres, over-smooth, cloud cover, heavy fog, motion blur, lens flare"
    )

    # --- Sampling Hyperparameters ---
    s_stage1: int = -1
    s_churn: int = 5
    s_noise: float = 1.003
    s_cfg: float = 7.5
    s_stage2: float = 1.0
    img_threshold: float = 0.3
    dec_img: float = 1.0

    # --- Advanced Settings ---
    color_fix_type: str = "Wavelet"
    linear_CFG: bool = False
    linear_s_stage2: bool = False
    spt_linear_CFG: float = 4.0
    spt_linear_s_stage2: float = 0.0

class ImageBatchProcessor:
    """
    Encapsulates the batch processing pipeline for super-resolution and captioning.
    Loads models once and processes a directory of images.
    """
    def __init__(self, cfg: BatchInferenceConfig):
        self.cfg = cfg
        self.output_dir = Path(self.cfg.save_dir) / "output"
        self.sr3_output_dir = Path(self.cfg.save_dir) / "sr3_output"
        self._setup_directories()

        # Initialize models and related components
        self.sr3_diffusion = None
        self.llava_model = None
        self.llava_tokenizer = None
        self.llava_image_processor = None
        self.refinement_model = None
        self.img_prompt_template = ""

        self._load_all_models()

    def _setup_directories(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sr3_output_dir.mkdir(parents=True, exist_ok=True)

    def _load_all_models(self):
        """Loads all necessary models and configurations."""
        print("Loading all models...")
        # 1. SR3 Diffusion Model
        sr3_args = SR3.SR3_Config()
        sr3_opt = Logger.parse(sr3_args)
        self.sr3_diffusion = sr3_model.create_model(sr3_opt)
        self.sr3_diffusion.set_new_noise_schedule(
            sr3_opt['model']['beta_schedule']['val'], schedule_phase='val'
        )

        # 2. LLaVA Model
        self.llava_tokenizer, self.llava_model, self.llava_image_processor = load_llava()
        self.llava_model.to(self.cfg.base_device)

        # 3. Refinement Model
        self.refinement_model = create_SR_model(self.cfg.model_yaml, 'Q')
        self.refinement_model.to(self.cfg.sr_device)

        # 4. Prompt Template
        with open(self.cfg.prompt_yaml, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        self.img_prompt_template = prompts["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
        print("All models and configurations loaded.")

    def _process_single_image(self, image_path: Path):
        """Processes a single image through the full SR and captioning pipeline."""
        filename_stem = image_path.stem

        # --- Stage 1: SR3 Upscaling ---
        loader = SR_Dataset.dataloader(str(image_path), self.cfg.upscale)
        self.sr3_diffusion.feed_data(next(iter(loader)))
        self.sr3_diffusion.test(continous=True)

        sr_tensor = self.sr3_diffusion.SR.squeeze()
        sr_img_np = T2I.tensor2img(sr_tensor, min_max=(-1, 1))
        sr_pil = Image.fromarray(sr_img_np)

        # --- Stage 2: Image Captioning with LLaVA ---
        image_tensor = process_images([sr_pil], self.llava_image_processor, self.llava_model.config)
        image_tensor = [_img.to(dtype=torch.float16, device=self.cfg.base_device) for _img in image_tensor]

        with torch.no_grad():
            image_caption = get_img_describe(
                image_tensor=image_tensor, image=sr_pil, model=self.llava_model,
                tokenizer=self.llava_tokenizer, prompt=self.img_prompt_template,
                max_new_tokens=256, conv_templates=conv_templates,
                image_token_index=IMAGE_TOKEN_INDEX, device=self.cfg.base_device
            )

        # --- Stage 3: Refinement ---
        lq_img, h0, w0 = PIL2Tensor(sr_pil, upscale=1, min_size=self.cfg.min_size)
        lq_img = lq_img.unsqueeze(0).to(self.cfg.sr_device)[:, :3, :, :]

        sampling_params = {
            "num_steps": self.cfg.num_steps, "restoration_scale": self.cfg.s_stage1,
            "s_churn": self.cfg.s_churn, "s_noise": self.cfg.s_noise,
            "cfg_scale": self.cfg.s_cfg, "control_scale": self.cfg.s_stage2,
            "seed": self.cfg.seed, "num_samples": self.cfg.num_samples,
            "p_p": self.cfg.a_prompt, "n_p": self.cfg.n_prompt,
            "color_fix_type": self.cfg.color_fix_type,
            "use_linear_CFG": self.cfg.linear_CFG,
            "use_linear_control_scale": self.cfg.linear_s_stage2,
            "cfg_scale_start": self.cfg.spt_linear_CFG,
            "control_scale_start": self.cfg.spt_linear_s_stage2,
            "img_threshold": self.cfg.img_threshold, "dec_img": self.cfg.dec_img,
        }

        sample_function = getattr(self.refinement_model, "just_sampling")
        with torch.no_grad():
            samples = sample_function(lq_img, image_caption, **sampling_params)

        # --- Stage 4: Save Results ---
        for idx, sample in enumerate(samples):
            final_img_path = self.output_dir / f"{filename_stem}_{idx}.png"
            sr3_img_path = self.sr3_output_dir / f"sr3_{filename_stem}_{idx}.png"
            Tensor2PIL(sample, h0, w0).save(final_img_path)
            sr_pil.save(sr3_img_path)

        # --- Memory Cleanup ---
        torch.cuda.empty_cache()
        gc.collect()

    def run(self):
        """Finds all images in the directory and processes them in a batch."""
        image_dir = Path(self.cfg.image_dir)
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in valid_exts]

        if not image_paths:
            print(f"No valid images found in the specified directory: {image_dir}")
            return

        print(f"Found {len(image_paths)} images to process.")
        for image_path in tqdm(sorted(image_paths), desc="Processing images"):
            try:
                self._process_single_image(image_path)
            except Exception as e:
                print(f"Failed to process {image_path.name}: {e}")

        print("\nAll images have been processed. Results saved to:")
        print(f"  ▶ Final SR: {self.output_dir}")
        print(f"  ▶ SR3 Intermediate: {self.sr3_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Batch super-resolution and caption generation")
    # Add arguments that are most likely to be changed by the user
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save the results")
    parser.add_argument("--upscale", type=int, default=8, help="Upscaling factor")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--img_threshold", type=float, default=0.3, help="Image threshold for refinement")

    args = parser.parse_args()

    # Create config object and override with any CLI arguments
    config = BatchInferenceConfig(
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        upscale=args.upscale,
        num_steps=args.num_steps,
        seed=args.seed,
        img_threshold=args.img_threshold  # Pass the value from parser
    )

    processor = ImageBatchProcessor(config)
    processor.run()

if __name__ == "__main__":
    main()
