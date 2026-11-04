"""Native reductions whose rounding must survive block compilation."""

import torch


@torch.library.custom_op("lightx2v::rms_norm_mean_square", mutates_args=(), device_types="cuda")
def mean_square(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True)


@mean_square.register_fake
def mean_square_fake(x):
    return torch.empty((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)


@torch.library.custom_op("lightx2v::rms_norm_sum_square", mutates_args=(), device_types="cuda")
def sum_square(x: torch.Tensor) -> torch.Tensor:
    return x.square().sum(dim=-1, keepdim=True)


@sum_square.register_fake
def sum_square_fake(x):
    return torch.empty((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)


@torch.library.custom_op("lightx2v::rms_norm_from_sum", mutates_args=(), device_types="cuda")
def rms_norm_from_sum(x: torch.Tensor, squared_sum: torch.Tensor, hidden_dim: int, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(squared_sum / hidden_dim + eps)


@rms_norm_from_sum.register_fake
def rms_norm_from_sum_fake(x, squared_sum, hidden_dim, eps):
    return torch.empty_like(x)
