import os
import torch, torch.nn as nn
from copy import deepcopy
from contextlib import contextmanager
from diffusers.models.controlnet import ControlNetOutput
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from sgm.models.diffusion import DiffusionEngine
from diffusers.models.controlnet import ControlNetOutput
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import torchmetrics
from sgm.modules.diffusionmodules.denoiser import *
from copy import deepcopy
from contextlib import contextmanager
from diffusers.models.controlnet import ControlNetOutput
from GLYPHSR.SR_model import *
import torchvision.transforms.functional as TF
import torchmetrics.functional as TMF
import lpips

# Recursively move tensors inside nested structures to the target device
def _tree_to(obj, *, device=None):
    if torch.is_tensor(obj):
        return obj.to(device=device)
    if isinstance(obj, ControlNetOutput):
        return ControlNetOutput(
            [_tree_to(t, device=device) for t in obj.down_block_residuals],
            _tree_to(obj.mid_block_residual, device=device)
        )
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to(x, device=device) for x in obj)
    if isinstance(obj, dict):
        return {k: _tree_to(v, device=device) for k, v in obj.items()}
    return obj


# Temporarily swap an attribute value on an object within a context
@contextmanager
def _swap(obj, attr, temp):
    orig = getattr(obj, attr)
    setattr(obj, attr, temp)
    try:
        yield
    finally:
        setattr(obj, attr, orig)

# Compute a weighted mean of residual features from base and delta outputs
def _residual_mean(a, b, beta=0.5):
    if torch.is_tensor(a):
        return (a + b) * beta
    if isinstance(a, (list, tuple)):
        return type(a)(_residual_mean(x, y) for x, y in zip(a, b))
    if isinstance(a, ControlNetOutput):
        return ControlNetOutput(
            [_residual_mean(x, y) for x, y in zip(a.down_block_residuals, b.down_block_residuals)],
            (1-beta) * a.mid_block_residual + beta * b.mid_block_residual
        )
    raise TypeError(type(a))

# ControlNet variant freezing base UNet and training lightweight project modules
class ProjectTSControlNet(nn.Module):
    def __init__(self, base_unet: nn.Module, beta=0.5):
        super().__init__()
        self.base = base_unet.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.beta = beta

        for k in (
            "input_blocks", "middle_block", "output_blocks",
            "time_embed", "label_emb", "out",
            "model_channels", "num_classes",
        ):
            if hasattr(self.base, k):
                setattr(self, k, getattr(self.base, k))

        self.project_modules = nn.ModuleList(
            deepcopy(self.base.project_modules)
        )

        for m in self.project_modules:
            m.requires_grad_(True)
            m.eval()                    

    # Forward pass combining base output with delta from project modules
    def forward(self, *args, **kwargs):
        with torch.no_grad():
            y_base = self.base(*args, **kwargs)

        device = next(self.base.parameters()).device
        args_c, kwargs_c = _tree_to(args, device=device), _tree_to(kwargs, device=device)

        with _swap(self.base, "project_modules", self.project_modules):
            y_delta = self.base(*args_c, **kwargs_c)

        return _residual_mean(y_base, y_delta,self.beta)

    # Ensure base stays frozen while project modules can train
    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        for m in self.project_modules:
            m.train(mode)

# Extract the state_dict field if present in checkpoint dict  
def get_state_dict(d): return d.get('state_dict', d)

# Load checkpoint from .pt or .safetensors and return state dict
def load_state_dict(ckpt_path, location='cpu'):
    _, ext = os.path.splitext(ckpt_path)
    if ext.lower() == ".safetensors":
        import safetensors.torch as st
        sd = st.load_file(ckpt_path, device=location)
    else:
        sd = torch.load(ckpt_path, map_location=location)
    sd = get_state_dict(sd)
    print(f"[✓] loaded weights: {ckpt_path}")
    return sd

# Load pretrained weights into model, move to device, and free memory
def load_TS_pretrained_model(model, ckpt_path,device, strict=False):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.model.load_state_dict(state_dict, strict=strict)
    del state_dict
    torch.cuda.empty_cache()

    model.to(device)
    return model
    
# Build TS ControlNet with configs, scheduler, and optional pretrained weights
def load_TS_ControlNet(
        cfg_path,
        args,
        loss_fn_config = None,
        device="cuda",
        sign=None,
        warm_up_steps=5000,
        total_steps=70000,
        beta =0.5,
        pretrained=None
    ):
    cfg   = OmegaConf.load(cfg_path)

    SR_backbone = instantiate_from_config(cfg.model).cpu()
    
        
    if cfg.SDXL_CKPT is not None:
        SR_backbone.load_state_dict(load_state_dict(cfg.SDXL_CKPT), strict=False)
    if cfg.SUPIR_CKPT is not None:
        SR_backbone.load_state_dict(load_state_dict(cfg.SUPIR_CKPT), strict=False)
    if sign is not None:
        SR_backbone.load_state_dict(load_state_dict(cfg.SUPIR_CKPT_Q), strict=False)
        
    SR_backbone.eval().to(device)

    base_unet = SR_backbone.model.diffusion_model
    SR_backbone.model.diffusion_model = ProjectTSControlNet(base_unet,beta=beta).to(device)

    for p in SR_backbone.parameters():
        p.requires_grad_(False)
    for p in SR_backbone.model.diffusion_model.project_modules.parameters():
        p.requires_grad_(True)

    trainable = SR_backbone.model.diffusion_model.project_modules.parameters()
    
    if loss_fn_config == None:
        loss_fn_config = {
        "target": "sgm.modules.diffusionmodules.loss.StandardDiffusionLoss",
        "params": {
            "sigma_sampler_config": {
                "target": "sgm.modules.diffusionmodules.sigma_sampling.DiscreteSampling",
                "params": {
                    "discretization_config": {
                        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
                    },
                    "num_idx": 1000,
                    "do_append_zero": False,
                    "flip": True,
                },
            },
            "type": "l2",
            "batch2model_keys": ["control_scale"],
        },
        }
    optimizer_config = {
        "target": "torch.optim.AdamW",
        "params": {"weight_decay": 1e-4},
    }
    scheduler_config = {
        "target": "sgm.lr_scheduler.LambdaWarmUpCosineScheduler",
        "params": {
            "warm_up_steps": warm_up_steps,
            "lr_min": 0.1, "lr_max": 1.0,
            "lr_start": 0.01, "max_decay_steps": total_steps,
        },
    }
    SR_backbone.learning_rate = 1e-6
    SR_backbone.input_key = "input_tensor"
    
    if loss_fn_config is not None:
        SR_backbone.loss_fn = instantiate_from_config(loss_fn_config)
    if optimizer_config is not None:
        SR_backbone.optimizer_config = optimizer_config
    if scheduler_config is not None:
        SR_backbone.scheduler_config = scheduler_config
    
    SR_backbone.upscale = 2
    SR_backbone.min_size = 256
    SR_backbone.val_sample_kwargs = {
        "num_steps":                  args.edm_steps,
        "restoration_scale":          args.s_stage1,
        "s_churn":                    args.s_churn,
        "s_noise":                    args.s_noise,
        "cfg_scale":                  args.s_cfg,
        "control_scale":              args.s_stage2,
        "seed":                       args.seed,
        "num_samples":                args.num_samples,
        "p_p":                        args.a_prompt,
        "n_p":                        args.n_prompt,
        "color_fix_type":             args.color_fix_type,
        "use_linear_CFG":             args.linear_CFG,
        "use_linear_control_scale":   args.linear_s_stage2,
        "cfg_scale_start":            args.spt_linear_CFG,
        "control_scale_start":        args.spt_linear_s_stage2,
    }
    
    if pretrained is not None:
        SR_backbone = load_TS_pretrained_model(SR_backbone, pretrained, device=device, strict=False)
        print(f"Loaded {pretrained} to {SR_backbone.__class__.__name__}")

    return SR_backbone, trainable
