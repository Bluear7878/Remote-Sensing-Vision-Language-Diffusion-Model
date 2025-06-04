"""
CUDA_VISIBLE_DEVICES=0 python kkw_Gsr.py \
    --json_path /home/delta1/Texture/Prompt/prompt_v0/CUTE80_image_x4_prompts.jsonl \
    --save_dir /home/delta1/KKW/SR_sample/output/image_ratio_results/CUTE80/imagex4 \
    --img_path /home/delta1/KKW/datasets/CUTE80 
"""
import os
import json
import copy
import argparse
import yaml
import torch
from dataclasses import dataclass, fields
from omegaconf import OmegaConf
from PIL import Image
from transformers import BitsAndBytesConfig
from sgm.util import instantiate_from_config
from GLYPHSR.ControlNet import load_TS_ControlNet
from GLYPHSR.OCR import get_img_describe, generate_ocr_text, get_text_position
from GLYPHSR.util import degrade_image, PIL2Tensor, Tensor2PIL
from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from peft import PeftModel
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="GLYPH-SR: OCR-guided super-resolution pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ─── Paths & I/O ──────────────────────────────────────
    parser.add_argument("--img_path", type=Path,
                        default="/home/delta1/KKW/datasets/CUTE80")
    parser.add_argument("--save_dir", type=Path, default="/home/delta1/KKW/SR_sample/output/image_ratio_results")
    parser.add_argument("--meta_filename", default="metadata.jsonl")
    parser.add_argument("--hq_meta_filename", default="metadata_HQ.jsonl")
    parser.add_argument("--neg_meta_filename", default="metadata_with_neg.jsonl")
    parser.add_argument("--yaml_file", type=Path, default="./model_configs/juggernautXL.yaml")
    parser.add_argument("--prompt_yaml_file", type=Path, default="./prompts/prompt_config.yaml")

    # ─── Hyper-parameters ────────────────────────────────
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--sign", default="Q")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--min_size", type=int, default=1024)
    parser.add_argument("--edm_steps", type=int, default=50)
    parser.add_argument("--text_steps", type=int, default=50)
    parser.add_argument("--s_stage1", type=int, default=-1)
    parser.add_argument("--s_churn", type=int, default=5)
    parser.add_argument("--s_noise", type=float, default=1.003)
    parser.add_argument("--s_cfg", type=float, default=7.5)
    parser.add_argument("--s_stage2", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--down_factor", type=int, default=1)

    parser.add_argument("--GYLPH_pretrained_ckpt", type=Path,
                        default="/home/delta1/Texture/TEXIR/checkpoints/best/smg/cp2.ckpt")
    parser.add_argument("--VLM_pretrained_ckpt", type=Path,
                        default="/home/delta1/TEXTURE-Diffusion-Based-Super-Resolution-Model-for-Enhanced-TEXT-Clarity/CKPT_PTH/Llava-next")

    # ─── Prompts ─────────────────────────────────────────
    parser.add_argument("--a_prompt", default=(
        "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
        "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
        "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    ))
    parser.add_argument("--a_text_prompt", default=(
        "literal text restoration, "
        "No other characters except text."
        "Focus on restoring text details. Do not degrade overall image quality. "
        "ultra-sharp glyphs, pixel-level text clarity, perfect readability, "
        "preserve exact font shapes and spacing, high contrast, legible"
    ))
    parser.add_argument("--n_prompt", default=(
        "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
        "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    ))
    parser.add_argument("--n_text_prompt", default=(
        "No non-textual shapes, No object-like artifacts, No abstract forms, "
        "No objects obscuring the text"
        "No random shapes, No blur, No halos, No ringing, No distortion, "
        "No character misplacement, No illegible segments, low resolution."
    ))

    # ─── Color & dtype ───────────────────────────────────
    parser.add_argument("--color_fix_type", default="Wavelet", choices=["Wavelet", "None"])
    parser.add_argument("--linear_CFG",      action="store_true",  default=True)
    parser.add_argument("--linear_s_stage2", action="store_true",  default=False)
    parser.add_argument("--spt_linear_CFG",      type=float, default=4.0)
    parser.add_argument("--spt_linear_s_stage2", type=float, default=0.0)
    parser.add_argument("--ae_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--diff_dtype", default="fp16", choices=["bf16", "fp16", "fp32"])

    # ─── Boolean flags ──────────────────────────────────
    parser.add_argument("--no_llava",            action="store_true", default=False)
    parser.add_argument("--loading_half_params", action="store_true", default=False)
    parser.add_argument("--use_tile_vae",        action="store_true", default=False)
    parser.add_argument("--load_8bit_llava",     action="store_true", default=False)
    parser.add_argument("--log_history",         action="store_true", default=False)

    # ─── Tiling & distributed ───────────────────────────
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)
    parser.add_argument("--top_k",      type=int, default=3)
    parser.add_argument("--beams_num",  type=int, default=10)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mask_prob",  type=float, default=0.3)

    # args ratio
    parser.add_argument("--json_path",  type=Path, required=True)

    return parser.parse_args() # py 파일 변환시 중요 ################################# 

def main():
    args = parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f.readlines()]

    print("Number of prompts:", len(prompts))

    # 모델 load ################################################
    cfg_path = args.yaml_file
    GYLPH_pretrained_ckpt = args.GYLPH_pretrained_ckpt

    GLYPH_SR_model, _ = load_TS_ControlNet(
        cfg_path=cfg_path,
        args=args,
        device="cuda",
        sign=args.sign,
        pretrained=GYLPH_pretrained_ckpt
    )
    GLYPH_SR_model.eval()

    for image_ratio in range(9, 10):

        # 저장할 디렉터리 설정 ####################################
        image_ratio = image_ratio / 10

        save_dir = os.path.join(args.save_dir, "ImageRatio_" + str(image_ratio)) # 경로 설정
        os.makedirs(save_dir, exist_ok=True)

        for prompt in prompts:
            texture_prompt_with_loc = [prompt["text_location"] + ' ' + prompt["image_caption"]]

            image_path = os.path.join(args.img_path, prompt["file_path"][1:])
            image_name = os.path.basename(image_path)
            image = Image.open(image_path).convert("RGB")

            # Apply degradation as needed
            if args.down_factor>1:
                degraded = degrade_image(image, down_factor=args.down_factor)
            else:
                degraded = image

            # Prepare low-quality tensor for SR
            LQ_img, h0, w0 = PIL2Tensor(degraded, upscale=args.upscale, min_size=args.min_size)
            LQ_img = LQ_img.unsqueeze(0).to("cuda")[:, :3, :, :]

            # Combine prompts
            a_text = args.a_text_prompt + args.a_prompt
            n_text = args.n_text_prompt + args.n_prompt

            # Sampling
            sample_fn = getattr(GLYPH_SR_model, "knagwoo_sampling")
            samples = sample_fn(
                LQ_img,
                [prompt["image_caption"]],
                texture_prompt_with_loc,
                image_steps=args.edm_steps//2,
                text_steps=args.text_steps//2,
                restoration_scale=args.s_stage1,
                s_churn=args.s_churn,
                s_noise=args.s_noise,
                cfg_scale=args.s_cfg,
                control_scale=args.s_stage2,
                seed=args.seed,
                num_samples=args.num_samples,
                p_p=args.a_prompt,
                a_text_prompt=a_text,
                n_p=args.n_prompt,
                n_text_prompt=n_text,
                color_fix_type=args.color_fix_type,
                use_linear_args=args.linear_CFG,
                use_linear_control_scale=args.linear_s_stage2,
                cfg_scale_start=args.spt_linear_CFG,
                control_scale_start=args.spt_linear_s_stage2,
                image_focus_ratio = image_ratio,   
            )

            # Convert tensor back to PIL and save
            output_path = os.path.join(save_dir, image_name)
            output_img = Tensor2PIL(samples[0], h0, w0)
            output_img.save(output_path)
            print(f"Saved super-resolved image to {output_path}")


if __name__ == "__main__":
    main()