"""Native operators whose BF16 rounding must survive block compilation."""

import torch
import torch.nn.functional as F


@torch.library.custom_op("lightx2v::causal_layer_norm", mutates_args=(), device_types="cuda")
def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    fp32_input: bool,
) -> torch.Tensor:
    """Keep the same reduction as LNWeight's native PyTorch implementation."""
    normalized = F.layer_norm(x.float() if fp32_input else x, (x.shape[-1],), weight, bias, eps)
    return normalized.to(x.dtype) if fp32_input else normalized


@layer_norm.register_fake
def layer_norm_fake(x, weight, bias, eps, fp32_input):
    return torch.empty_like(x, memory_format=torch.contiguous_format)
