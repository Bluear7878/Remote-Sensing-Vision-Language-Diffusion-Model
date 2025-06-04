# Environment: nunchaku
# Run command: python3 build_dataset.py

import os, json, cv2, random, numpy as np, torch
import gc
from glob import glob
from PIL import Image
from tqdm import tqdm
from itertools import islice

from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision

# ------------------ File path setup --------------------
prompt_meta   = "./prompt_bank/prompts_meta.jsonl"  # Input JSONL file with prompt metadata
save_root     = "./train_dataset"                     # Root directory for saving output dataset
hq_dir        = os.path.join(save_root, "hq")       # Directory for saving generated high-quality images
os.makedirs(hq_dir, exist_ok=True)                   # Create output directory if it doesn't exist
meta_out_path = os.path.join(save_root, "metadata.jsonl")  # Output JSONL metadata file path
meta_dir      = os.path.dirname(meta_out_path)               # Directory of metadata file, for relative path calculation

# ------------------ Model loading ----------------------
# Determine the computation precision (bfloat16 or fp16) dynamically for compatibility and performance
precision = get_precision()

# Load the customized Flux Transformer 2D model checkpoint for the diffusion pipeline
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/svdq-{precision}-flux.1-dev"
)

# Path to LoRA weights for realistic style enhancement in the transformer model
lora_path = "XLabs-AI/flux-RealismLora/lora.safetensors"

# Initialize the Flux diffusion pipeline with the loaded transformer model
# Use bfloat16 precision and move model to CUDA device for faster inference
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

# Apply LoRA parameters to the transformer to improve output realism
pipeline.transformer.update_lora_params(lora_path)

# Set the strength of LoRA influence (0.8 means 80% of the LoRA effect)
pipeline.transformer.set_lora_strength(0.8)

# ----------------- Generation parameters ----------------
BATCH_SIZE = 1          # Number of prompts to process in one batch (adjustable for memory constraints)
STEPS = 50              # Number of inference steps for the diffusion process (controls image quality/time trade-off)

def chunk(it, size):
    """
    Utility function to split an iterable into chunks of a specified size.
    This helps to batch-process prompts without loading all into memory at once.
    """
    it = iter(it)
    for first in it:
        yield [first, *list(islice(it, size-1))]

# ------------------ Main dataset generation loop --------------------
with open(prompt_meta, "r") as f_in, open(meta_out_path, "w") as f_out:
    lines = f_in.readlines()  # Load all lines (JSON records) from prompt metadata file

    # Iterate through prompt lines in batches (chunked by BATCH_SIZE)
    for group in tqdm(chunk(lines, BATCH_SIZE)):
        ids, prompts = [], []

        # Parse each JSON line in the batch, extract prompt id and prompt text
        for ln in group:
            rec = json.loads(ln)
            ids.append(rec["id"])
            prompts.append(rec["prompt"])

        # Perform inference with no gradient tracking and mixed precision for efficiency
        # This generates images conditioned on the input prompts
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = pipeline(prompts, num_inference_steps=STEPS).images

        # Save generated images and write corresponding metadata per sample
        for stem, prompt, hq in zip(ids, prompts, images):
            hq_path = os.path.join(hq_dir, f"{stem}.png")
            hq.save(hq_path)  # Save the high-quality image to disk

            # Metadata includes unique id, prompt text, and relative path to image file
            meta_obj = {
                "id": stem,
                "prompt": prompt,
                "hq": os.path.relpath(hq_path, start=meta_dir),
            }
            f_out.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")

        # Explicitly free GPU and CPU memory after each batch to prevent memory leaks or fragmentation
        del images, hq
        gc.collect()
        torch.cuda.empty_cache()

print("Dataset build complete →", meta_out_path)
