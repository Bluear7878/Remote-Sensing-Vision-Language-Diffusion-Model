#!/usr/bin/env python3
"""
python3 train_GLYPH_SR.py \
    --hq_meta_path /home/delta1/Texture/GYLPH-SR/train_sample/metadata_HQ.jsonl \
    --neg_meta_path /home/delta1/Texture/GYLPH-SR/train_sample/metadata_with_neg.jsonl \
    --val_hq_meta_path /home/delta1/Texture/GYLPH-SR/train_sample/metadata_HQ.jsonl \
    --val_neg_meta_path /home/delta1/Texture/GYLPH-SR/train_sample/metadata_with_neg.jsonl \
"""
import os
import argparse
from dataclasses import dataclass, fields
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from sgm.util import instantiate_from_config
from GLYPHSR.ControlNet import load_TS_ControlNet
from GLYPHSR.util import *
from GLYPHSR.dataloader import make_hqlq_dataloader

# Default device
SR_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainConfig:
    # Data paths
    hq_meta_path: str = "/home/delta1/Texture/GYLPH-SR/train_sample/metadata_HQ.jsonl"
    neg_meta_path: str ="/home/delta1/Texture/GYLPH-SR/train_sample/metadata_with_neg.jsonl"
    val_hq_meta_path: str  = "/home/delta1/Texture/GYLPH-SR/train_sample/metadata_HQ.jsonl"
    val_neg_meta_path: str  = "/home/delta1/Texture/GYLPH-SR/train_sample/metadata_with_neg.jsonl"

    # Model & ControlNet
    yaml_file: str = "./model_configs/juggernautXL.yaml"
    pretrained_ckpt: str = ""    # Path to pretrained ControlNet checkpoint
    sign: str = "Q"              # ControlNet sign

    # Training hyperparameters
    batch_size: int = 12
    num_epochs: int = 200
    img_size_train: int = 512
    img_size_val: int = 256
    num_samples: int = 1
    
    upscale: int = 2
    sign: str = "Q"
    seed: int = -1
    min_size: int = 1024
    edm_steps: int = 50
    text_steps: int = 50
    s_stage1: int = -1
    s_churn: int = 5
    s_noise: float = 1.003
    s_cfg: float = 7.5
    s_stage2: float = 1.0
    color_fix_type: str = "Wavelet"
    linear_CFG: bool = True
    linear_s_stage2: bool = False
    spt_linear_CFG: float = 4.0
    spt_linear_s_stage2: float = 0.0
    ae_dtype: str = "bf16"
    diff_dtype: str = "fp16"

    # Flags
    no_llava: bool = False
    loading_half_params: bool = False
    use_tile_vae: bool = False
    encoder_tile_size: int = 512
    decoder_tile_size: int = 64
    load_8bit_llava: bool = False
    log_history: bool = False

    # Sampling & distributed
    top_k: int = 3
    beams_num: int = 10
    world_size: int = 1
    local_rank: int = 0
    mask_prob: float = 0.3

    # Lightning & logging
    gpus: int = 1
    precision: int = 32
    gradient_clip_val: float = 1.0
    log_dir: str = "lightning_logs"
    ckpt_dir: str = "checkpoints"
    ckpt_filename: str = "FT1_epoch{epoch:03d}-train_loss{train/loss:.4f}"
    monitor_metric: str = "train/loss"
    monitor_mode: str = "min"
    
    # Prompts
    a_prompt: str = (
        "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
        "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
        "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations."
    )
    a_text_prompt: str = (
        "literal text restoration, "
        "No other characters except text."
        "Focus on restoring text details. Do not degrade overall image quality. "
        "ultra-sharp glyphs, pixel-level text clarity, perfect readability, "
        "preserve exact font shapes and spacing, high contrast, legible"
    )
    n_prompt: str = (
        "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
        "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
        "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
    )
    n_text_prompt: str = (
        "No non-textual shapes, No object-like artifacts, No abstract forms, "
        "No objects obscuring the text"
        "No random shapes, No blur, No halos, No ringing, No distortion, "
        "No character misplacement, No illegible segments, low resolution."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train GLYPH-SR Text-SR ControlNet")
    for field in fields(TrainConfig):
        arg_name = f"--{field.name}"
        default = field.default
        if default is None:
            parser.add_argument(arg_name, required=True, help=f"{field.name} (required)")
        elif isinstance(default, bool):
            action = 'store_true' if not default else 'store_false'
            parser.add_argument(arg_name, action=action, help=f"flag, default={default}")
        else:
            arg_type = type(default)
            parser.add_argument(arg_name, type=arg_type, default=default,
                                help=f"{field.name}, default={default}")
    return parser.parse_args()


def main():
    # Load config
    args = parse_args()
    cfg = TrainConfig(**vars(args))

    # Ensure directories exist
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Prepare data loaders
    train_loader = make_hqlq_dataloader(
        hq_meta_path=cfg.hq_meta_path,
        neg_meta_path=cfg.neg_meta_path,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size_train,
        include_types=("P_HQ","P_LQ","N_HQ","N_LQ"),
        shuffle=True
    )
    val_loader = make_hqlq_dataloader(
        hq_meta_path=cfg.val_hq_meta_path,
        neg_meta_path=cfg.val_neg_meta_path,
        batch_size=1,
        img_size=cfg.img_size_val,
        include_types=("P_HQ",),
        shuffle=False
    )

    # Load ControlNet-based SR model
    model, _ = load_TS_ControlNet(
        cfg_path=cfg.yaml_file,
        args=cfg,
        device=SR_DEVICE,
        sign=cfg.sign
    )

    # Setup logging and callbacks
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, name="text_sr_controlnet")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.ckpt_dir,
        filename=cfg.ckpt_filename,
        monitor=cfg.monitor_metric,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.gpus,
        precision=cfg.precision,
        max_epochs=cfg.num_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
