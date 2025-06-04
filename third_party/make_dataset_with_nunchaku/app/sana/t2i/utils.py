import torch
from diffusers import SanaPAGPipeline

from nunchaku.models.transformers.transformer_sana import NunchakuSanaTransformer2DModel


def hash_str_to_int(s: str) -> int:
    """Hash a string to an integer."""
    modulus = 10**9 + 7  # Large prime modulus
    hash_int = 0
    for char in s:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def get_pipeline(
    precision: str, use_qencoder: bool = False, device: str | torch.device = "cuda", pipeline_init_kwargs: dict = {}
) -> SanaPAGPipeline:
    if precision == "int4":
        assert torch.device(device).type == "cuda", "int4 only supported on CUDA devices"
        transformer = NunchakuSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m", pag_layers=8)

        pipeline_init_kwargs["transformer"] = transformer
        if use_qencoder:
            raise NotImplementedError("Quantized encoder not supported for Sana for now")
    else:
        assert precision == "bf16"
    pipeline = SanaPAGPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
        variant="bf16",
        torch_dtype=torch.bfloat16,
        pag_applied_layers="transformer_blocks.8",
        **pipeline_init_kwargs,
    )
    if precision == "int4":
        pipeline._set_pag_attn_processor = lambda *args, **kwargs: None

    pipeline = pipeline.to(device)
    return pipeline
