# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Three-dimensional rotary embeddings using complex values or real pairs."""

from __future__ import annotations

import torch


def rope_apply_complex(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1), freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).type_as(x)


def rope_apply_real(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    c = x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        freqs_i = torch.cat(
            [freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2), freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2), freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2)], dim=-2
        ).reshape(seq_len, 1, -1, 2)
        x_i = apply_rotary_pairs(x[i, :seq_len], freqs_i)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).type_as(x)


def apply_rotary_pairs(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Multiply interleaved real/imaginary pairs without complex operations."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine, sine = frequencies[..., 0], frequencies[..., 1]
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imaginary = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(x.dtype)
