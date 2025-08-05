from typing import Tuple, Union

import torch


def rms_norm(x, weight, eps):
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * weight
    return x


def rotate_half(x, shape_0, shape_1):
    x_real, x_imag = x.reshape(shape_0, shape_1, -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(2)


def rotary_emb(x, shape_0, shape_1, cos, sin):
    x_out = x * cos + rotate_half(x, shape_0, shape_1) * sin
    return x_out


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape_0, shape_1, shape_2 = xq.shape
    cos = freqs_cis[0].view(shape_0, 1, shape_2)
    sin = freqs_cis[1].view(shape_0, 1, shape_2)
    xq_out = rotary_emb(xq, shape_0, shape_1, cos, sin)
    xk_out = rotary_emb(xk, shape_0, shape_1, cos, sin)
    return xq_out, xk_out
