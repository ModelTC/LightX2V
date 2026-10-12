"""Rotary arithmetic and positional precision supported by compute devices."""

from __future__ import annotations

import torch


def positional_compute_dtype(device) -> torch.dtype:
    return torch.float32 if torch.device(device).type in {"npu", "mlu"} else torch.float64


def prepare_rotary_frequencies(frequencies: torch.Tensor, device) -> torch.Tensor:
    """Use explicit real FP32 pairs on NPU and MLU devices."""
    if torch.device(device).type in {"npu", "mlu"}:
        if frequencies.is_complex():
            frequencies = torch.view_as_real(frequencies)
        frequencies = frequencies.float()
    return frequencies.to(device)


def apply_rotary_pairs(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Multiply interleaved real/imaginary pairs without complex operations."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cosine, sine = frequencies[..., 0], frequencies[..., 1]
    real = pairs[..., 0] * cosine - pairs[..., 1] * sine
    imaginary = pairs[..., 0] * sine + pairs[..., 1] * cosine
    return torch.stack((real, imaginary), dim=-1).flatten(-2).to(x.dtype)
