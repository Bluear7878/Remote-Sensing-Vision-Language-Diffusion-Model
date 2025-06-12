#!/usr/bin/env python
# env : llama3_ft
# python3 infer_dir.py \
#  --image_dir "/home/ict04/ocr_sr/KMK/GYLPH-SR/dataset/SR3_RSSCN7_28_224/results" \
#  --save_dir "./results"
# coding: utf-8

import argparse
import gc
import os
import time

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from GLYPHSR.dataloader import *
from GLYPHSR.util import *
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from Texture_eval_mk import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch super-resolution and caption generation using LLAVA and SUPIR"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images to process"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory to save the output images (default: ./results)"
    )
    parser.add_argument(
        "--supir_yaml",
        type=str,
        default="/home/ict04/ocr_sr/KMK/GYLPH-SR/model_configs/juggernautXL.yaml",
        help="Path to the SUPIR model configuration YAML file"
    )
    parser.add_argument(
        "--prompt_yaml",
        type=str,
        default="/home/ict04/ocr_sr/KMK/GYLPH-SR/prompts/prompt_config.yaml",
        help="Path to the YAML file containing caption prompt settings"
    )
    parser.add_argument(
        "--sr_device",
        type=str,
        default="cuda:0",
        help="CUDA device for running the SR model (default: cuda:0)"
    )
    parser.add_argument(
        "--base_device",
        type=str,
        default="cuda:1",
        help="CUDA device for running the LLAVA (caption) model (default: cuda:1)"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=8,
        help="Upscaling factor for the low-resolution image (default: 8)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=1024,
        help="Minimum size for resizing the image before SR (default: 1024)"
    )
    parser.add_argument(
        "--s_stage1",
        type=int,
        default=-1,
        help="Stage-1 restoration scale for SUPIR (default: -1)"
    )
    parser.add_argument(
        "--s_churn",
        type=int,
        default=5,
        help="Churn parameter for SUPIR (default: 5)"
    )
    parser.add_argument(
        "--s_noise",
        type=float,
        default=1.003,
        help="Noise parameter for SUPIR (default: 1.003)"
    )
    parser.add_argument(
        "--s_cfg",
        type=float,
        default=7.5,
        help="CFG scale parameter for SUPIR (default: 7.5)"
    )
    parser.add_argument(
        "--s_stage2",
        type=float,
        default=1.0,
        help="Stage-2 control scale for SUPIR (default: 1.0)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of sampling steps for SUPIR (default: 50)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of SR samples to generate per image (default: 1)"
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default=(
            "Cinematic, High Contrast, highly detailed aerial photo taken using "
            "a high-resolution drone or satellite, hyper detailed photo-realistic maximum detail, "
            "32k, Color Grading, ultra HD, extreme meticulous detailing of terrain textures and structures, "
            "hyper sharpness, no deformations."
        ),
        help="Positive prompt string"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default=(
            "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, "
            "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
            "signature, jpeg artifacts, deformed, lowres, over-smooth, cloud cover, heavy fog, motion blur, lens flare"
        ),
        help="Negative prompt string"
    )
    parser.add_argument(
        "--color_fix_type",
        type=str,
        default="Wavelet",
        help="Color correction method to apply (default: Wavelet)"
    )
    parser.add_argument(
        "--linear_CFG",
        action="store_true",
        help="Enable linear CFG scale (default: False; specify flag to enable)"
    )
    parser.add_argument(
        "--linear_s_stage2",
        action="store_true",
        help="Enable linear control scale for stage 2 (default: False; specify flag to enable)"
    )
    parser.add_argument(
        "--spt_linear_CFG",
        type=float,
        default=4.0,
        help="Starting value for spectral linear CFG scale (default: 4.0)"
    )
    parser.add_argument(
        "--spt_linear_s_stage2",
        type=float,
        default=0.0,
        help="Starting value for spectral linear control scale in stage 2 (default: 0.0)"
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()

def main():
    args = parse_args()

    # Create the save directory if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Load caption prompt settings from YAML
    with open(args.prompt_yaml, "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    img_prompt = PROMPTS["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)

    # Load LLAVA model for image captioning
    tokenizer, llava_model, image_processor = load_llava()
    llava_model.to(args.base_device)

    # Load SUPIR model for super-resolution
    SR_model = create_SR_model(args.supir_yaml, SUPIR_sign='Q')
    SR_model.to(args.sr_device)

    # Gather list of image files in the input directory
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_paths = [
        os.path.join(args.image_dir, fname)
        for fname in sorted(os.listdir(args.image_dir))
        if fname.lower().endswith(valid_exts)
    ]
    if not image_paths:
        print(f"No valid images found in the specified directory: {args.image_dir}")
        return

    # Retrieve the sampling function from the SR model
    sample_function = getattr(SR_model, "just_sampling", None)
    if sample_function is None:
        raise RuntimeError("The SR model does not have a 'just_sampling' method.")

    # Process each image: generate caption and perform SR
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)

        # Load original image as PIL Image
        image = Image.open(image_path).convert("RGB")

        # Prepare input tensor for LLAVA captioning
        image_tensor = process_images([image], image_processor, llava_model.config)
        image_tensor = [
            _img.to(dtype=torch.float16, device=args.base_device)
            for _img in image_tensor
        ]

        # Convert PIL image to low-resolution tensor for SUPIR
        LQ_img, h0, w0 = PIL2Tensor(
            image,
            upscale=args.upscale,
            min_size=args.min_size
        )
        LQ_img = LQ_img.unsqueeze(0).to(args.sr_device)[:, :3, :, :]

        # Generate image caption with LLAVA
        with torch.no_grad():
            image_caption = get_img_describe(
                image_tensor=image_tensor,
                image=image,
                model=llava_model,
                tokenizer=tokenizer,
                prompt=img_prompt,
                max_new_tokens=256,
                conv_templates=conv_templates,
                image_token_index=IMAGE_TOKEN_INDEX,
                device=args.base_device
            )

        # Perform super-resolution with SUPIR
        with torch.no_grad():
            samples = sample_function(
                LQ_img,
                image_caption,
                img_threshold=0.1,
                dec_img=0.99,
                num_steps=args.num_steps,
                restoration_scale=args.s_stage1,
                s_churn=args.s_churn,
                s_noise=args.s_noise,
                cfg_scale=args.s_cfg,
                control_scale=args.s_stage2,
                seed=args.seed,
                num_samples=args.num_samples,
                p_p=args.a_prompt,
                n_p=args.n_prompt,
                color_fix_type=args.color_fix_type,
                use_linear_CFG=args.linear_CFG,
                use_linear_control_scale=args.linear_s_stage2,
                cfg_scale_start=args.spt_linear_CFG,
                control_scale_start=args.spt_linear_s_stage2
            )

        # Save each generated SR sample to disk
        for idx, sample in enumerate(samples):
            output_filename = f"{name}_{idx}.png"
            output_path = os.path.join(args.save_dir, output_filename)
            Tensor2PIL(sample, h0, w0).save(output_path)

        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    print("All images have been processed. Results saved to:")
    print(f"  ▶ {args.save_dir}")


if __name__ == "__main__":
    main()
