"""Rotary arithmetic using real pairs."""

from __future__ import annotations

import torch


def apply_rotary_pairs(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Multiply interleaved real/imaginary pairs without complex operations."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine, sine = frequencies[..., 0], frequencies[..., 1]
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imaginary = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(x.dtype)
