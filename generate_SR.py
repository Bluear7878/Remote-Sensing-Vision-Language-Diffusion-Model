#!/usr/bin/env python
# env : llama3_ft
# python3 generate_SR.py \
#  --meta_jsonl "/home/ict04/ocr_sr/KMK/GYLPH-SR/results/SR3_RSSCN7_28_224_sample.jsonl" \
#  --save_dir "./results/SR3_RSSCN7_28_224_sample"
# coding: utf-8

import argparse
import gc
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

from GLYPHSR.util import *
from Texture_eval_mk import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform SR on images using captions from a JSONL metadata file"
    )
    parser.add_argument(
        "--meta_jsonl",
        type=str,
        required=True,
        help="Path to the JSONL file containing image paths and captions"
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
        default="cuda",
        help="CUDA device to run the SR model on (default: cuda:0)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results",
        help="Directory to save SR output images (default: ./sr_results)"
    )
    parser.add_argument(
        "--upscale",
        type=int,
        default=1,
        help="Upscaling factor for the low-resolution image (default: 1)"
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
        help="Positive prompt string for SUPIR (default: cinematic style)"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default=(
            "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, "
            "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
            "signature, jpeg artifacts, deformed, lowres, over-smooth, cloud cover, heavy fog, motion blur, lens flare"
        ),
        help="Negative prompt string for SUPIR (default: standard negations)"
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

    # Ensure SR output directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Load JSONL metadata into a list of dicts
    metadata_list = []
    with open(args.meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Each entry must contain "image_path" and "caption"
                if "image_path" in entry and "caption" in entry:
                    metadata_list.append(entry)
            except json.JSONDecodeError:
                continue

    if not metadata_list:
        print(f"No valid metadata entries found in: {args.meta_jsonl}")
        return

    # Load SUPIR model once
    SR_model = create_SR_model(args.supir_yaml, SUPIR_sign='Q')
    SR_model.to(args.sr_device)

    # Retrieve the sampling function from the SR model
    sample_function = getattr(SR_model, "just_sampling", None)
    if sample_function is None:
        raise RuntimeError("SR model does not have a 'just_sampling' method.")

    # Iterate over each metadata entry
    for entry in tqdm(metadata_list, desc="Running SR on images"):
        image_path = entry["image_path"]
        caption    = entry["caption"]
        filename   = os.path.basename(image_path)
        name, _    = os.path.splitext(filename)

        # Load the low-resolution image
        image = Image.open(image_path).convert("RGB")

        # Convert PIL image to low-resolution tensor for SUPIR
        LQ_img, h0, w0 = PIL2Tensor(
            image,
            upscale=args.upscale,
            min_size=args.min_size
        )
        LQ_img = LQ_img.unsqueeze(0).to(args.sr_device)[:, :3, :, :]


        # Run super-resolution with stored caption (caption is passed even if not strictly used)
        with torch.no_grad():
            samples = sample_function(
                LQ_img,
                caption,
                img_threshold=-0.1,
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

        # Clear GPU memory before next iteration
        torch.cuda.empty_cache()
        gc.collect()

    print("SR processing complete. Outputs saved to:")
    print(f"  ▶ {args.save_dir}")


if __name__ == "__main__":
    main()
