import typing as tp

import torch

from ...utils import ceil_divide, load_state_dict_in_safetensors


def is_nunchaku_format(lora: str | dict[str, torch.Tensor]) -> bool:
    if isinstance(lora, str):
        tensors = load_state_dict_in_safetensors(lora, device="cpu")
    else:
        tensors = lora

    for k in tensors.keys():
        if ".mlp_fc" in k or "mlp_context_fc1" in k:
            return True
    return False


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result
