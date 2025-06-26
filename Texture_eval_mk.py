LAVA_BASE_MODEL = "lmms-lab/llama3-llava-next-8b"
LAVA_FT_PATH    = "/home/delta1/TEXTURE-Diffusion-Based-Super-Resolution-Model-for-Enhanced-TEXT-Clarity/CKPT_PTH/Llava-next"
PROMPT_YAML     = "/home/delta1/Texture/Prompt/prompt_config.yaml"

import argparse
import csv
import gc
# ───────────────────────────────────────────────────────────────
# 1) import & Config
# ───────────────────────────────────────────────────────────────
import os
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

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from llava.model.builder import load_pretrained_model
from models.ControlNet import *
from models.dataloader import *
from models.util import PIL2Tensor, Tensor2PIL


@dataclass
class Config:
    img_dir: str
    save_dir: str = ""
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


def load_llava(device = "cuda"):
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4")
    tok, base, img_proc, _ = load_pretrained_model(
        LAVA_BASE_MODEL, None, "llava_llama_3", device_map="cpu")
    model = PeftModel.from_pretrained(base, LAVA_FT_PATH,
                                      device_map="cpu")
    model.eval().to(device)
    return tok, model, img_proc
