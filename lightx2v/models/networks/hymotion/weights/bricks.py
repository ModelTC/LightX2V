# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from collections.abc import Callable

import torch
from torch import Tensor, nn


def get_activation_layer(act_type: str) -> Callable[[], nn.Module]:
    if act_type == "gelu":
        return lambda: nn.GELU()
    elif act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "relu":
        return nn.ReLU
    elif act_type == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def get_norm_layer(norm_type: str | None):
    if norm_type == "layer":
        return nn.LayerNorm
    elif norm_type == "rms":
        return RMSNorm
    elif norm_type == "none" or norm_type is None:
        return nn.Identity
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))


__all__ = ["Tensor", "torch"]
