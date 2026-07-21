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

    # RoPE has no checkpoint parameters, but it participates in the same
    # WeightModule tree as linear, attention, and norm implementations.
    def load(self, weight_dict):
        pass

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        return {} if destination is None else destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return {} if destination is None else destination

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        pass

    def named_parameters(self, prefix=""):
        return iter(())

    def prepare_freqs(self, freqs):
        return freqs

    def prepare_positions(self, freqs):
        return None

    def apply_single(self, x: torch.Tensor, freqs, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, q: torch.Tensor, k: torch.Tensor, freqs, **kwargs):
        return self.apply_single(q, freqs, **kwargs), self.apply_single(k, freqs, **kwargs)


def broadcast_freqs(freqs: torch.Tensor, target: torch.Tensor, unsqueeze_dim: int = -2):
    while freqs.ndim < target.ndim:
        freqs = freqs.unsqueeze(unsqueeze_dim)
    return freqs
