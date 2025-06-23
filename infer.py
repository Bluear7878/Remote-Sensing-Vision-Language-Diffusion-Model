# env : llama3_metrics
#python infer.py --input_img /home/delta1/GMK/raw_data/WHU-RS19_28_224/lr_28/viaduct_23.png
#               --output ./results
#               --scale 8

SR_MODEL_CUDA   = "cuda:0"
BASE_MODEL_CUDA = "cuda:1"

#LAVA_BASE_MODEL = "lmms-lab/llama3-llava-next-8b"
#LAVA_FT_PATH    = "/home/ict04/ocr_sr/HSJ/aSUPTextIR_proj/SUPIR/CKPT_PTH/Llava-next"
#DEFAULT_SUPIR_YAML = "./options/SUPIR_v0_juggernautXL.yaml"
PROMPT_YAML = "/home/delta1/GMK/Texture/prompts/prompt_config.yaml"
SUPIR_YAML = "/home/delta1/GMK/Texture/model_configs/juggernautXL.yaml"

import argparse
import csv
import gc
# ───────────────────────────────────────────────────────────────
# 1) import & Config
# ───────────────────────────────────────────────────────────────
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lpips
import torch
import torchmetrics.functional as TMF
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import BitsAndBytesConfig

import data as Data
import data.single_img as Single_Img
import SR3.config.sr3 as SR3
import SR3.model as sr3_model
import SR3.utill.logger as Logger
import SR3.utill.tensor2img as T2I
from GLYPHSR.ControlNet import *
from GLYPHSR.dataloader import *
from GLYPHSR.util import *
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model
from Texture_eval_mk import *


@dataclass
class Config:
    img_dir: str
    save_dir: str = "./results"
    hq_meta_filename: str = "metadata_HQ.jsonl"
    neg_meta_filename: str = "metadata_with_neg.jsonl"
    yaml_file: str = "./options/SUPIR_v0_juggernautXL.yaml"
    prompt_yaml_file: str = PROMPT_YAML

    upscale: int = 8
    SUPIR_sign: str = "Q"
    seed: int = -1
    min_size: int = 1024
    edm_steps: int = 50
    text_steps: int = 50
    s_stage1: int = -1
    s_churn: int = 5
    s_noise: float = 1.003
    s_cfg: float = 7.5
    s_stage2: float = 1.0
    num_samples: int = 1

    a_prompt: str = ("Cinematic, High Contrast, highly detailed aerial photo taken using a high-resolution drone or satellite, hyper detailed photo-realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing of terrain textures and structures, hyper sharpness, no deformations.")

    n_prompt: str = ("painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth, cloud cover, heavy fog, motion blur, lens flare")

    color_fix_type: str = "Wavelet"
    linear_CFG: bool = True
    linear_s_stage2: bool = False
    spt_linear_CFG: float = 4.0
    spt_linear_s_stage2: float = 0.0
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"

    no_llava: bool = False
    loading_half_params: bool = False
    use_tile_vae: bool = False
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64
    load_8bit_llava: bool = False
    log_history: bool = False

    top_k: int = 3
    beams_num: int = 10
    world_size: int = 1
    local_rank: int = 0
    mask_prob: float = 0.3
    metric_size: tuple[int,int] | None = None

def pipeline(input_img_path, output_dir,scale):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(input_img_path)
    filename = os.path.splitext(basename)[0]
    # 1. Load Prompt
    with open(PROMPT_YAML, "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    img_prompt = PROMPTS["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)


    # 2. Load SR3 model
    sr3_args = SR3.SR3_Config()
    sr3_opt = Logger.parse(sr3_args)
    diffusion = sr3_model.create_model(sr3_opt)
    diffusion.set_new_noise_schedule(
        sr3_opt['model']['beta_schedule']['val'], schedule_phase='val')

    # 3. Run SR3 Inference
    loader = Single_Img.single_image_dataloader(
        input_img_path, scale)
    for val_data in loader:
        diffusion.feed_data(val_data)
    diffusion.test(continous=True)
    sr_tensor = diffusion.SR
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor[-1]
    sr_img_np = T2I.tensor2img(sr_tensor, min_max=(-1, 1))
    sr_pil = Image.fromarray(sr_img_np)
    output_path = os.path.join(output_dir, f"sr3_{filename}.png")
    sr_pil.save(output_path)


    # 4. Load LLaVA model
    tokenizer, llava_model, image_processor = load_llava(device=BASE_MODEL_CUDA)
    image_tensor = process_images([sr_pil], image_processor, llava_model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=BASE_MODEL_CUDA) for _image in image_tensor]

    # 5. Get caption
    image_caption = get_img_describe(
            image_tensor=image_tensor, image=sr_pil, model=llava_model,
            tokenizer=tokenizer, prompt = img_prompt,
            max_new_tokens = 256,
            conv_templates=conv_templates, image_token_index=IMAGE_TOKEN_INDEX,
            device=BASE_MODEL_CUDA
        )


    # 6. SR Refinement (SUPIR)

    args = Config(img_dir=".")

    LQ_img, h0, w0 = PIL2Tensor(sr_pil, upscale=1, min_size=args.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(SR_MODEL_CUDA)[:, :3, :, :]

    SR_model = create_SR_model(SUPIR_YAML,SUPIR_sign='Q')
    SR_model.to(SR_MODEL_CUDA)

    sample_func = "just_sampling"
    sample_function = getattr(SR_model, sample_func, None)


    if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = time.time()

    samples = sample_function(LQ_img, image_caption,img_threshold=0.3, dec_img=1,
                            num_steps=50, restoration_scale=args.s_stage1, s_churn=args.s_churn,
                                        s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2, seed=args.seed,
                                        num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
                                        use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                        cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)

    if torch.cuda.is_available():
            torch.cuda.synchronize()
    end_time = time.time()
    infer_time = end_time - start_time

    for _i, sample in enumerate(samples):
            output_path = os.path.join(output_dir, f"{filename}.png")
            Tensor2PIL(sample, h0, w0).save(output_path)
            print(f"Saved result: {output_path}")

    #image_files = ['/home/ict04/ocr_sr/KMK/GYLPH-SR/dataset/SR3_RSSCN7_28_224/results/0_83_sr.png']

    #image_path = image_files[0]

    #filename = os.path.basename(image_path)
    #name = os.path.splitext(filename)[0]
    #name = "test"
    #image = Image.open(image_path)
    #width, height = sr_pil.size
    #image_sizes = [sr_pil.size]
    input_tensor = image_tensor[0]  # shape: [3, H, W] or [1, 3, H, W]
    if input_tensor.dim() == 4:
        input_tensor = input_tensor[0]  # Remove batch dim

    output_tensor = samples[0]  # shape: [3, H, W] (이미 batch 제거됨)

    # clamp to same range if needed (optional)
    input_tensor = input_tensor.float().cpu()
    output_tensor = output_tensor.float().cpu()

    print("📥 Input tensor stats (image_tensor):")
    print("  min:", input_tensor.min().item(), "max:", input_tensor.max().item())
    print("📤 Output tensor stats (samples):")
    print("  min:", output_tensor.min().item(), "max:", output_tensor.max().item())
    print(infer_time)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="./results", help="Directory to save outputs")
    parser.add_argument("--scale", type=int, default=8, help="Upscaling factor applied to the input image before super-resolution")

    args = parser.parse_args()

    pipeline(args.input_img, args.output, args.scale)
