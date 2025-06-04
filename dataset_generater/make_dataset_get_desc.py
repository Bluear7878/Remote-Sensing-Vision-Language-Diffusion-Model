#!/usr/bin/env python3
#env : llama3_ft
#CUDA_VISIBLE_DEVICES=0 python3 -m runner.run_dataset_maker --image_dir /home/delta1/KKW/datasets/SVT/origin

import argparse
import os
import torch
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from dataset_generater.dataset_maker import make_image_loader, generate_prompt_meta_jsonl

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text-to-image prompts metadata from an image folder."
    )
    parser.add_argument(
        "--prompt_yaml_file",
        type=str,
        default="/home/delta1/Texture/GYLPH-SR/prompts/prompt_config.yaml",
        help="Path to prompt_config.yaml (contains ocr_prompt template)."
    )
    parser.add_argument(
        "--image_dir", "-i",
        type=str,
        required=True,
        help="Directory containing images to process."
    )
    parser.add_argument(
        "--prompt_bank", "-p",
        type=str,
        default="./prompt_bank",
        help="Directory where prompts_meta.jsonl will be saved."
    )
    parser.add_argument(
        "--model_path", "-m",
        type=str,
        default="lmms-lab/llama3-llava-next-8b",
        help="Pretrained LLaVA model identifier or local path."
    )
    parser.add_argument(
        "--conv_template", "-c",
        type=str,
        default="llava_llama_3",
        help="Key of the conversation template to use."
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=1,
        help="Batch size for image loading."
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=4,
        help="Number of workers for DataLoader."
    )
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=10000,
        help="Maximum number of prompts to generate."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Torch device (e.g. 'cuda', 'cuda:0', or 'cpu')."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    loader = make_image_loader(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    tokenizer, base_model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.conv_template,
        device_map="auto",
        attn_implementation=None
    )
    base_model.eval().to(args.device)

    meta_path, count = generate_prompt_meta_jsonl(
        yaml_file=args.prompt_yaml_file,
        loader=loader,
        image_processor=image_processor,
        base_model=base_model,
        tokenizer=tokenizer,
        conv_templates=conv_templates,
        conv_template=args.conv_template,
        IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX,
        prompt_bank=args.prompt_bank,
        max_samples=args.max_samples,
        device=args.device,
    )

    print(f"✅ Saved {count} prompts to {meta_path}")

if __name__ == "__main__":
    main()
