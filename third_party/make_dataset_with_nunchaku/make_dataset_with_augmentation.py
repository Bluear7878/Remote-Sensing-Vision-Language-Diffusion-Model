import os, json, cv2, random, gc, torch
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
from itertools import islice
import torchvision.transforms as T
from diffusers import FluxControlNetModel
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision

# List of augmentation keywords to diversify prompt conditions.
# These keywords add various photographic styles and effects to enrich the training data.
AUG_KEYWORDS = [
    "overcast lighting",
    "Fujifilm Pro 400H color palette",
    "cinematic bokeh",
    "vintage grain",
    "soft natural lighting",
    "dramatic shadows",
    "warm color tones",
    "shallow depth of field",
    "sunset glow",
    "golden hour warmth",
    "high-key lighting",
    "low-key moody lighting",
    "lens flare accents",
    "silhouette composition",
    "wide-angle perspective",
    "telephoto compression",
    "tilt-shift miniaturization",
    "macro close-up detail",
    "reflections in water",
    "symmetry and leading lines",
    "rule-of-thirds framing",
    "cinematic aspect ratio",
    "soft vignette",
    "rainy window droplets",
    "misty atmosphere",
    "backlit rim lighting",
    "analog film scratches",
    "HDR-like dynamic range",
    "pastel color grading",
    "neon color accents",
    "cold blue tones",
    "warm amber highlights",
    "soft focus glow",
    "textured overlay grain",
    "dramatic sky replacement",
    "urban night lights",
    "forest dappled light",
    "architectural leading lines",
]

# File paths for prompt metadata and dataset saving
PROMPT_META = "./prompt_bank/prompts_meta.jsonl"
SAVE_ROOT   = "./train_dataset"
HQ_DIR      = f"{SAVE_ROOT}/hq"
META_OUT    = f"{SAVE_ROOT}/metadata.jsonl"
os.makedirs(HQ_DIR, exist_ok=True)  # Ensure directory exists to save generated images
meta_dir = os.path.dirname(META_OUT)  # Base directory for relative paths in metadata

# Load pretrained ControlNet model for upscaling task with half precision (bfloat16) for memory efficiency
controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=torch.bfloat16
)

# Dynamically get precision string for loading transformer weights accordingly (bf16 or fp16)
precision = get_precision()

# Load the custom Flux Transformer model with LoRA adapters for realism enhancement
transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/svdq-{precision}-flux.1-dev"
)
lora_path = "XLabs-AI/flux-RealismLora/lora.safetensors"

# Construct the diffusion pipeline combining transformer, ControlNet, with bfloat16 precision on GPU
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

# Load and apply LoRA parameters to the transformer to improve realism with adjustable strength
pipe.transformer.update_lora_params(lora_path)
pipe.transformer.set_lora_strength(0.8)

# Optional: Apply caching optimization to speed up inference (currently commented out)
# apply_cache_on_pipe(pipe, residual_diff_threshold_multi=0.01)

# Define image preprocessing pipeline converting PIL image to normalized tensor for potential use
to_tensor = T.Compose([
    T.Resize(512, interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize(0.5, 0.5)
])

# Batch size and inference step count for generation
BATCH_SIZE = 1
STEPS = 50

def chunk(it, n):
    """
    Utility generator to yield successive n-sized chunks from an iterable.
    Useful to process large lists in batches without loading all at once.
    """
    it = iter(it)
    for first in it:
        yield [first, *list(islice(it, n-1))]

# Main processing: open prompt metadata input and output metadata file
with open(PROMPT_META, "r", encoding="utf-8") as f_in, \
     open(META_OUT, "w", encoding="utf-8") as f_out:

    # Load all prompt records from jsonl file into memory
    raw_lines = [json.loads(ln) for ln in f_in]

    # Total iterations equals all prompt records multiplied by augmentation keywords count
    total = len(raw_lines) * len(AUG_KEYWORDS)
    pbar = tqdm(total=total, desc="building")

    # Expand each prompt record by augmenting it with all augmentation keywords, creating
    # a new prompt variant per keyword to increase training diversity.
    expanded = []
    for rec in raw_lines:
        for kw in AUG_KEYWORDS:
            new_rec = rec.copy()
            new_rec["aug_kw"] = kw
            expanded.append(new_rec)

    # Process the expanded prompt list in chunks according to batch size
    for group in chunk(expanded, BATCH_SIZE):
        prompts, ids = [], []

        # Prepare prompt strings by combining augmentation keyword with original prompt text
        for rec in group:
            ids.append(rec["id"])
            prompts.append(f"{rec['aug_kw']} , {rec['prompt']}")

        # Run diffusion model inference with automatic mixed precision (bfloat16) for efficiency
        # torch.no_grad disables gradient calculation to save memory and computation
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            images = pipe(prompts, num_inference_steps=STEPS).images

        # Save each generated image to disk and write corresponding metadata
        for img, rec in zip(images, group):
            # Create a filesystem-friendly filename by concatenating id and sanitized augmentation keyword
            stem = f"{rec['id']}_{rec['aug_kw'].replace(' ','_').replace('/','')}"
            hq_path = f"{HQ_DIR}/{stem}.png"
            img.save(hq_path)

            # Output metadata includes unique id, prompt text, and relative path to generated image
            out = {
                "id": stem,
                "prompt": prompts[ids.index(rec["id"])],
                "hq": os.path.relpath(hq_path, meta_dir),
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

        # Explicitly release GPU and CPU memory to maintain stable resource usage during loop
        del images
        gc.collect()
        torch.cuda.empty_cache()

        # Update progress bar according to processed batch size
        pbar.update(len(group))

pbar.close()
print("✅ dataset build complete →", META_OUT)
