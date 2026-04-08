from functools import lru_cache

import torch
import torch.nn as nn


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    batch, seq, heads, complex_twice_dim = x.shape
    assert complex_twice_dim % 2 == 0
    complex_dim = complex_twice_dim // 2

    c_f = complex_dim - 2 * (complex_dim // 3)
    c_h = complex_dim // 3
    c_w = complex_dim // 3
    fpart, hpart, wpart = freqs.split([c_f, c_h, c_w], dim=1)

    x_c = torch.view_as_complex(x.to(torch.float64).reshape(batch, seq, heads, -1, 2)).contiguous()
    y_c = x_c.clone()
    gsz = grid_sizes.to(torch.long)
    uniq, inv = torch.unique(gsz, dim=0, return_inverse=True)

    @lru_cache(maxsize=256)
    def _make_freq_grid(f: int, h: int, w: int):
        return torch.cat(
            [
                fpart[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                hpart[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                wpart[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).contiguous()

    for g_idx, (f, h, w) in enumerate(uniq.tolist()):
        idx = (inv == g_idx).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        seq_len = f * h * w
        freq_grid = _make_freq_grid(f, h, w)
        y_c[idx, :seq_len] = x_c[idx, :seq_len] * freq_grid

    return torch.view_as_real(y_c).reshape(batch, seq, heads, -1).float()


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)
