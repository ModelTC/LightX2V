from __future__ import annotations

from typing import Literal

import torch


RopeLayout = Literal["interleaved", "split_half"]


class RopeTemplate:
    def __init__(self, layout: RopeLayout = "interleaved", compute_dtype: torch.dtype = torch.float32):
        if layout not in {"interleaved", "split_half"}:
            raise ValueError(f"Unsupported RoPE layout: {layout}")
        self.layout = layout
        self.compute_dtype = compute_dtype
        self.config = {}

    def set_config(self, config=None):
        if config is not None:
            self.config = config

    def apply_single(self, x: torch.Tensor, freqs, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        return self.apply_single(q, freqs, **kwargs), self.apply_single(k, freqs, **kwargs)


def broadcast_freqs(freqs: torch.Tensor, target: torch.Tensor, unsqueeze_dim: int = -2):
    while freqs.ndim < target.ndim:
        freqs = freqs.unsqueeze(unsqueeze_dim)
    return freqs
