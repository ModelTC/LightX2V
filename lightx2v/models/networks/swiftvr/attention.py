"""SwiftVR mask-free shifted-window attention."""

from __future__ import annotations

import torch

from lightx2v.common.ops import attn as _attention_ops  # noqa: F401
from lightx2v.common.ops.attn.template import AttnWeightTemplate
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

ATTENTION_BACKENDS = ("torch_sdpa", "flash_attn2", "flash_attn3", "sage_attn2")


def axis_window_starts(size: int, window: int, shifted: bool, device: torch.device) -> torch.Tensor:
    if size <= window:
        return torch.zeros(1, dtype=torch.long, device=device)

    shift = window // 2 if shifted else 0
    max_start = size - window
    starts = (torch.arange((size + window - 1) // window + 2, device=device) * window - shift).clamp_(0, max_start)
    starts = torch.unique(starts, sorted=True)
    if starts.numel() > 2:
        keep = torch.ones_like(starts, dtype=torch.bool)
        keep[1:-1] = starts[2:] > starts[:-2] + window
        starts = starts[keep]
    return starts


class WindowLayout:
    _cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, int]] = {}

    @classmethod
    def get(cls, frames: int, height: int, width: int, window: tuple[int, int], shifted: bool, device: torch.device):
        key = (frames, height, width, *window, shifted, device.type, device.index)
        if key in cls._cache:
            return cls._cache[key]

        window_height, window_width = window
        height_starts = axis_window_starts(height, window_height, shifted, device)
        width_starts = axis_window_starts(width, window_width, shifted, device)
        height_index = height_starts[:, None] + torch.arange(window_height, device=device)[None]
        width_index = width_starts[:, None] + torch.arange(window_width, device=device)[None]
        spatial = (height_index[:, None, :, None] * width + width_index[None, :, None, :]).reshape(-1, window_height * window_width)
        indices = (torch.arange(frames, device=device)[None, :, None] * (height * width) + spatial[:, None]).reshape(spatial.shape[0], -1)

        window_count, window_length = indices.shape
        owner = torch.empty(frames * height * width, dtype=torch.long)
        local = torch.arange(window_length)
        window_order = range(window_count) if shifted else range(window_count - 1, -1, -1)
        indices_cpu = indices.cpu()
        for window_index in window_order:
            owner[indices_cpu[window_index]] = window_index * window_length + local

        layout = indices.flatten(), owner.to(device), window_count, window_length
        cls._cache[key] = layout
        return layout

    @classmethod
    def clear(cls):
        cls._cache.clear()


@ATTN_WEIGHT_REGISTER("swiftvr_mfswa")
class SwiftVRShiftedWindowAttention(AttnWeightTemplate):
    """Full-temporal, 2D shifted-window attention used by SwiftVR."""

    window_size = (16, 16)
    backend = "torch_sdpa"

    @classmethod
    def configure(cls, window_size=(16, 16), backend="torch_sdpa"):
        cls.window_size = tuple(window_size)
        if backend not in ATTENTION_BACKENDS:
            raise ValueError(f"Unsupported SwiftVR attention backend: {backend}")
        cls.backend = backend
        WindowLayout.clear()

    def __init__(self):
        self.config = {}
        self.dense_attention = ATTN_WEIGHT_REGISTER[self.backend]()

    def apply(self, q, k, v, grid_sizes=None, block_idx=0, **kwargs):
        frames, height, width = grid_sizes
        window = min(self.window_size[0], height), min(self.window_size[1], width)
        indices, owner, window_count, window_length = WindowLayout.get(
            frames,
            height,
            width,
            window,
            shifted=bool(block_idx % 2),
            device=q.device,
        )

        heads, head_dim = q.shape[-2:]
        q = q.index_select(0, indices).view(window_count, window_length, heads, head_dim)
        k = k.index_select(0, indices).view(window_count, window_length, heads, head_dim)
        v = v.index_select(0, indices).view(window_count, window_length, heads, head_dim)

        attention_args = {}
        if self.backend.startswith("flash_attn"):
            cu_seqlens = torch.arange(
                0,
                (window_count + 1) * window_length,
                window_length,
                dtype=torch.int32,
                device=q.device,
            )
            attention_args = {"cu_seqlens_q": cu_seqlens, "cu_seqlens_kv": cu_seqlens}

        output = self.dense_attention.apply(
            q,
            k,
            v,
            max_seqlen_q=window_length,
            max_seqlen_kv=window_length,
            **attention_args,
        )
        return output.reshape(window_count * window_length, heads * head_dim).index_select(0, owner)
