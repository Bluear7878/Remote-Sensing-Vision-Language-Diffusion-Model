#!/usr/bin/env python
# env : llama3_ft
#CUDA_VISIBLE_DEVICES=0 python3 generate_meta.py   --image_dir "/home/ict04/ocr_sr/KMK/GYLPH-SR/dataset/SR3_RSSCN7_28_224/results"   --output_jsonl "./results/SR3_RSSCN7_28_224.jsonl"
# coding: utf-8

import argparse
import json
import os

import torch
import yaml
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from models.util import *
from Texture_eval_mk import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSONL metadata with image captions using LLAVA"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images to caption"
    )
    parser.add_argument(
        "--prompt_yaml",
        type=str,
        default="./prompts/prompt_config.yaml",
        help="Path to the YAML file containing caption prompt settings"
    )
    parser.add_argument(
        "--base_device",
        type=str,
        default="cuda",
        help="CUDA device to run the LLAVA model on (default: cuda:0)"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Path to the output JSONL file where metadata will be saved"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Create parent directory for output JSONL if it doesn't exist
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    # Load caption prompts from YAML
    with open(args.prompt_yaml, "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    img_prompt = PROMPTS["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)

    # Load LLAVA model and associated tokenizer + processor
    tokenizer, llava_model, image_processor = load_llava(device = args.base_device)
    llava_model.to(args.base_device)

    # Gather list of image file paths in the input directory
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_paths = [
        os.path.join(args.image_dir, fname)
        for fname in sorted(os.listdir(args.image_dir))
        if fname.lower().endswith(valid_exts)
    ]
    if not image_paths:
        print(f"No images found in directory: {args.image_dir}")
        return

    # Open output JSONL file for writing
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        # Process each image: generate a caption and write JSON object
        for image_path in tqdm(image_paths, desc="Generating captions"):
            filename = os.path.basename(image_path)
            name, _ = os.path.splitext(filename)

            # Load image with PIL and ensure RGB mode
            image = Image.open(image_path).convert("RGB")

            # Prepare LLAVA input tensor (batch of 1)
            image_tensor = process_images([image], image_processor, llava_model.config)
            image_tensor = [
                _img.to(dtype=torch.float16, device=args.base_device)
                for _img in image_tensor
            ]

            # Generate caption with LLAVA (no gradient)
            with torch.no_grad():
                caption = get_img_describe(
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

            # Build metadata dictionary
            metadata = {
                "image_path": image_path,
                "caption": caption
            }

            # Write one JSON object per line
            out_f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print(f"Metadata JSONL file written to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
