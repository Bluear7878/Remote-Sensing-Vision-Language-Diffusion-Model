SR_MODEL_CUDA   = "cuda:0"
BASE_MODEL_CUDA = "cuda:1"

LAVA_BASE_MODEL = "lmms-lab/llama3-llava-next-8b"
LAVA_FT_PATH    = "/home/ict04/ocr_sr/HSJ/aSUPTextIR_proj/SUPIR/CKPT_PTH/Llava-next"
DEFAULT_SUPIR_YAML = "./options/SUPIR_v0_juggernautXL.yaml"
PROMPT_YAML     = "/home/ict04/ocr_sr/Texture/Prompt/prompt_config.yaml"

# ───────────────────────────────────────────────────────────────
# 1) import & Config
# ───────────────────────────────────────────────────────────────
import os, gc, csv, yaml, argparse, lpips
import torch, torchmetrics.functional as TMF
import torchvision.transforms.functional as TF
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image

from omegaconf import OmegaConf
from llava.model.builder import load_pretrained_model
from llava.mm_utils      import process_images
from llava.constants     import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation  import conv_templates
from peft                import PeftModel
from transformers        import BitsAndBytesConfig

from GLYPHSR.util          import *
from dataclasses import dataclass
from GLYPHSR.ControlNet import *
from GLYPHSR.dataloader import *
from llava.conversation import conv_templates
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

args = Config(img_dir=".")

PROMPT_YAML = "/home/ict04/ocr_sr/KMK/GYLPH-SR/prompts/prompt_config.yaml"
with open(PROMPT_YAML, "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)
    
img_prompt = PROMPTS["img_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)

SUPIR_YAML = "/home/ict04/ocr_sr/KMK/GYLPH-SR/model_configs/juggernautXL.yaml"
tokenizer, llava_model, image_processor = load_llava()

SR_model = create_SR_model(SUPIR_YAML,SUPIR_sign='Q')
SR_model.to(SR_MODEL_CUDA)

image_files = ['/home/ict04/ocr_sr/KMK/GYLPH-SR/dataset/SR3_RSSCN7_28_224/results/0_83_sr.png']

image_path = image_files[0]

filename = os.path.basename(image_path)  
name = os.path.splitext(filename)[0]    

image = Image.open(image_path)
width, height = image.size
image_sizes = [image.size]

image_tensor = process_images([image], image_processor, llava_model.config)
image_tensor = [_image.to(dtype=torch.float16, device=BASE_MODEL_CUDA) for _image in image_tensor]

image_caption = get_img_describe(
        image_tensor=image_tensor, image=image, model=llava_model, 
        tokenizer=tokenizer, prompt = img_prompt,
        max_new_tokens = 256,
        conv_templates=conv_templates, image_token_index=IMAGE_TOKEN_INDEX, 
        device=BASE_MODEL_CUDA
    )

LQ_img, h0, w0 = PIL2Tensor(image, upscale=4, min_size=args.min_size)
LQ_img = LQ_img.unsqueeze(0).to(SR_MODEL_CUDA)[:, :3, :, :]

sample_func = "just_sampling"
sample_function = getattr(SR_model, sample_func, None)

samples = sample_function(LQ_img, image_caption,img_threshold=0.1, dec_img=1, # img_threshold = 0.1 -> 0.15 -> 0.2 dec_img = 1 
                          num_steps=50, restoration_scale=args.s_stage1, s_churn=args.s_churn,
                                    s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2, seed=args.seed,
                                    num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
                                    use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                    cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)

for _i, sample in enumerate(samples):
        output_filename = f'{name}_{_i}.png'
        output_path = os.path.join(args.save_dir, output_filename)
        Tensor2PIL(sample, h0, w0).save(output_path)
        print(f"Saved result: {output_path}")