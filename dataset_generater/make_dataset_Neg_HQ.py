#!/usr/bin/env python3
#env : llama3_ft
#CUDA_VISIBLE_DEVICES=0,2 python3 -m runner.make_dataset_Neg_HQ --img_dir ./train_samples --world_size 2 --local_rank 0
#CUDA_VISIBLE_DEVICES=1,3 python3 -m runner.make_dataset_Neg_HQ --img_dir ./train_samples --world_size 2 --local_rank 1
#CUDA_VISIBLE_DEVICES=1,3 python3 -m runner.make_dataset_Neg_HQ --img_dir ./train_samples


"""
Pipeline script for generating negative low-quality (Neg-LQ) samples
and building a new metadata JSONL file.
"""
import argparse
import os
import yaml
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates

from GLYPHSR.util import convert_dtype
from dataset_generater.dataset_maker import build_neg_lq,create_SUPIR_model
from GLYPHSR.ControlNet import load_TS_ControlNet



def parse_args():
    parser = argparse.ArgumentParser(description="Generate Neg-LQ dataset and metadata.")
    # Paths and I/O
    parser.add_argument("--img_dir",       type=str, required=True)
    parser.add_argument("--save_dir",      type=str, required=False)
    parser.add_argument("--hq_meta_filename",  
                        type=str, default="metadata_HQ.jsonl",
                        help="Filename for the HQ metadata JSONL (id, prompt, HQ_path, OCR).")
    parser.add_argument("--neg_meta_filename", 
                        type=str, default="metadata_with_neg.jsonl",
                        help="Filename for the NEG metadata JSONL (id, prompt, neg_path, OCR, SR_OCR).")
    parser.add_argument("--down_factors",  nargs='+', type=int, default=[10, 15],
                        help="Downsampling factors to generate low-quality inputs")
    parser.add_argument(
        "--yaml_file", "-y",
        type=str,
        default="/home/delta1/Texture/GYLPH-SR/model_configs/SUPIR.yaml",
        help="Path to SUPIR YAML (contains ocr_prompt template)."
    )
    parser.add_argument(
        "--prompt_yaml_file",
        type=str,
        default="/home/delta1/Texture/Prompt/prompt_config.yaml",
        help="Path to prompt_config.yaml (contains ocr_prompt template)."
    )
    # LoRA & SUPIR settings
    parser.add_argument("--upscale",       type=int, default=1)
    parser.add_argument("--SUPIR_sign",    type=str, default='Q', choices=['F', 'Q'])
    parser.add_argument("--seed",          type=int, default=-1)
    parser.add_argument("--min_size",      type=int, default=1024)
    parser.add_argument("--edm_steps",     type=int, default=50)
    parser.add_argument("--text_steps",    type=int, default=50)
    parser.add_argument("--s_stage1",      type=int, default=-1)
    parser.add_argument("--s_churn",       type=int, default=5)
    parser.add_argument("--s_noise",       type=float, default=1.003)
    parser.add_argument("--s_cfg",         type=float, default=7.5)
    parser.add_argument("--s_stage2",      type=float, default=1.0)
    parser.add_argument("--num_samples",   type=int, default=1)
    # Prompts
    parser.add_argument("--a_prompt",      type=str,
                        default=('Cinematic, High Contrast, highly detailed, taken using a ' \
                                 'Canon EOS R camera, hyper detailed photo - realistic maximum detail, ' \
                                 '32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
                                 'detailing, hyper sharpness, perfect without deformations.'))
    parser.add_argument("--a_text_prompt", type=str,
                        default=('Focus on restoring text details. Do not degrade overall ' \
                                 'image quality. Keep the rest of the image as is, with minimal modifications.'))
    parser.add_argument("--n_prompt",      type=str,
                        default=('painting, oil painting, illustration, drawing, art, sketch, oil painting, ' \
                                 'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, ' \
                                 'worst quality, low quality, frames, watermark, signature, jpeg artifacts, ' \
                                 'deformed, lowres, over-smooth'))
    parser.add_argument("--n_text_prompt", type=str,
                        default=('artifacts, blur, overly smooth, cartoonish, painting style, reducing resolution, ' \
                                 'messy, dirty, unknown text, random letters.'))
    # Color & config
    parser.add_argument("--color_fix_type",      type=str, default='Wavelet',
                        choices=["None", "AdaIn", "Wavelet"])
    parser.add_argument("--linear_CFG",          action='store_true', default=True)
    parser.add_argument("--linear_s_stage2",     action='store_true', default=False)
    parser.add_argument("--spt_linear_CFG",      type=float, default=4.0)
    parser.add_argument("--spt_linear_s_stage2", type=float, default=0.0)
    parser.add_argument("--ae_dtype",            type=str, default="bf16", choices=['fp32', 'bf16'])
    parser.add_argument("--diff_dtype",          type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
    # Flags
    parser.add_argument("--no_llava",            action='store_true', default=False)
    parser.add_argument("--loading_half_params", action='store_true', default=False)
    parser.add_argument("--use_tile_vae",        action='store_true', default=False)
    parser.add_argument("--encoder_tile_size",   type=int, default=512)
    parser.add_argument("--decoder_tile_size",   type=int, default=64)
    parser.add_argument("--load_8bit_llava",     action='store_true', default=False)
    parser.add_argument("--log_history",         action='store_true', default=False,
                        help="If enabled, save the terminal arguments and generated captions to a log file")
    parser.add_argument("--top_k",                type=int, default=3)
    parser.add_argument("--beams_num",           type=int, default=10)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--mask_prob", type=float, default=0.3)
    return parser.parse_args()


def main():
    conv_template = "llava_llama_3"
    MODEL_PATH = "lmms-lab/llama3-llava-next-8b"
    SR_MODEL_CUDA = "cuda:0"
    OCR_MODEL_CUDA = "cuda:1"
    args = parse_args()
    # Set seeds
    torch.manual_seed(args.seed)

    # Load prompts
    if not os.path.isfile(args.prompt_yaml_file):
        raise FileNotFoundError(f"YAML file not found: {args.prompt_yaml_file}")
    with open(args.prompt_yaml_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    img_prompt = prompts['img_prompt'].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
    ocr_prompt = prompts['ocr_prompt'].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)

    tokenizer, ocr_model, image_processor, _ = load_pretrained_model(
    MODEL_PATH, None, conv_template, device_map="cpu", attn_implementation=None)

    ocr_model.eval().to(OCR_MODEL_CUDA)

    # Load SUPIR model
    sr_model = create_SUPIR_model(args.yaml_file, SUPIR_sign=args.SUPIR_sign)
    if args.loading_half_params:
        sr_model = sr_model.half()
    if args.use_tile_vae:
        sr_model.init_tile_vae(encoder_tile_size=args.encoder_tile_size,
                                decoder_tile_size=args.decoder_tile_size)
    sr_model.ae_dtype = convert_dtype(args.ae_dtype)
    sr_model.model.dtype = convert_dtype(args.diff_dtype)
    sr_model = sr_model.to(SR_MODEL_CUDA)

    # ----- Run Neg-LQ Builder -----
    build_neg_lq(
        root_dir=args.img_dir,
        image_processor=image_processor,
        ocr_model=ocr_model,
        sr_model=sr_model,
        ocr_prompt=ocr_prompt,
        img_prompt=img_prompt,
        tokenizer=tokenizer,
        conv_templates=conv_templates,
        conv_template=conv_template,
        down_factors=args.down_factors,
        args=args,
        hq_meta_filename=args.hq_meta_filename,
        neg_meta_filename=args.neg_meta_filename,
        world_size   = args.world_size,
        local_rank   = args.local_rank,
        mask_prob = args.mask_prob,
    )

if __name__ == '__main__':
    main()
