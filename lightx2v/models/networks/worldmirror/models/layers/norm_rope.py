import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lightx2v.common.ops.rope import TorchRealRope

_WORLDMIRROR_NORMALIZED_ROPE = TorchRealRope(layout="split_half")


@dataclass(frozen=True)
class NormalizedRotaryPositionEmbedding2DCache:
    """Per-forward, position-gathered normalized RoPE values."""

    cos: torch.Tensor
    sin: torch.Tensor

    def reshape(self, batch_size: int, token_count: int) -> "NormalizedRotaryPositionEmbedding2DCache":
        def reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[0] * tensor.shape[2] != batch_size * token_count:
                raise ValueError(f"Cannot reshape RoPE cache from batch/tokens={tensor.shape[0]}/{tensor.shape[2]} to {batch_size}/{token_count}.")
            return tensor.reshape(batch_size, 1, token_count, tensor.shape[-1])

        return NormalizedRotaryPositionEmbedding2DCache(
            cos=reshape_tensor(self.cos),
            sin=reshape_tensor(self.sin),
        )

    def select_batch(self, indices: torch.Tensor) -> "NormalizedRotaryPositionEmbedding2DCache":
        return NormalizedRotaryPositionEmbedding2DCache(
            cos=self.cos.index_select(0, indices),
            sin=self.sin.index_select(0, indices),
        )


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self) -> None:
        self.position_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        cache_key = (height, width, torch.device(device))
        if cache_key not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            self.position_cache[cache_key] = torch.cartesian_prod(y_coords, x_coords)

        cached_positions = self.position_cache[cache_key]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1)


class NormalizedRotaryPositionEmbedding2D(nn.Module):
    """DINOv3-aligned 2D Rotary Position Embedding."""

    def __init__(
        self,
        *,
        head_dim: int,
        base: float = 100.0,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Union[float, None] = None,
        jitter_coords: Union[float, None] = None,
        rescale_coords: Union[float, None] = None,
        dtype: Union[torch.dtype, None] = None,
        device: Union[torch.device, None] = None,
        **ignored_kwargs,
    ) -> None:
        super().__init__()
        if len(ignored_kwargs) > 0:
            # maintain parity with DINOv3 implementation that warns on ignored kwargs
            pass

        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by 4 for 2D RoPE")

        self.head_dim = head_dim
        self.base = base
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype

        quarter_dim = head_dim // 4
        self.register_buffer(
            "periods",
            torch.empty(quarter_dim, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_periods()

    def _init_periods(self) -> None:
        quarter_dim = self.periods.shape[0]
        half_dim = self.head_dim // 2
        exponents = 2 * torch.arange(quarter_dim, device=self.periods.device, dtype=self.dtype) / half_dim
        periods = self.base**exponents
        self.periods.data.copy_(periods)

    def _get_sincos_for_grid(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        dd = {"device": device, "dtype": dtype}

        if self.normalize_coords == "max":
            max_hw = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_hw
            coords_w = torch.arange(0.5, W, **dd) / max_hw
        elif self.normalize_coords == "min":
            min_hw = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_hw
            coords_w = torch.arange(0.5, W, **dd) / min_hw
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0

        if self.training:
            if self.shift_coords is not None:
                shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
                coords += shift_hw[None, :]
            if self.jitter_coords is not None:
                jitter_max = np.log(self.jitter_coords)
                jitter_hw = torch.empty(2, **dd).uniform_(-jitter_max, jitter_max).exp()
                coords *= jitter_hw[None, :]
            if self.rescale_coords is not None:
                rescale_max = np.log(self.rescale_coords)
                rescale_hw = torch.empty(1, **dd).uniform_(-rescale_max, rescale_max).exp()
                coords *= rescale_hw

        periods = self.periods.to(device=device, dtype=dtype)
        angles = (2 * math.pi * coords[:, :, None]) / periods[None, None, :]  # [HW, 2, D/4]
        angles = angles.flatten(1, 2)  # [HW, D/2]
        angles = torch.cat((angles, angles), dim=-1)  # [HW, D]

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return sin, cos

    def prepare_cache(self, positions: torch.Tensor, head_dim: int, dtype: torch.dtype) -> NormalizedRotaryPositionEmbedding2DCache:
        """Sample coordinates and gather their cos/sin values once per forward."""
        if head_dim != self.head_dim:
            raise ValueError(f"Head dim {head_dim} doesn't match configured {self.head_dim}.")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("Positions must have shape (batch_size, n_tokens, 2).")

        batch_size, token_count = positions.shape[:2]
        height = int(positions[..., 0].max().item() + 1)
        width = int(positions[..., 1].max().item() + 1)
        sin, cos = self._get_sincos_for_grid(height, width, positions.device, dtype)

        indices = (positions[..., 0] * width + positions[..., 1]).long()
        flat_indices = indices.reshape(-1)
        gathered_sin = sin[flat_indices].view(batch_size, 1, token_count, head_dim)
        gathered_cos = cos[flat_indices].view(batch_size, 1, token_count, head_dim)
        return NormalizedRotaryPositionEmbedding2DCache(cos=gathered_cos, sin=gathered_sin)

    @staticmethod
    def _validate_cache(tokens: torch.Tensor, cache: NormalizedRotaryPositionEmbedding2DCache) -> None:
        if tokens.shape[0] != cache.cos.shape[0] or tokens.shape[-2] != cache.cos.shape[-2]:
            raise ValueError(f"RoPE cache shape does not match tokens: cache={cache.cos.shape}, tokens={tokens.shape}.")

    def apply_single(self, tokens: torch.Tensor, cache: NormalizedRotaryPositionEmbedding2DCache) -> torch.Tensor:
        self._validate_cache(tokens, cache)
        return _WORLDMIRROR_NORMALIZED_ROPE.apply_single(tokens, (cache.cos, cache.sin))

    def apply_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cache: NormalizedRotaryPositionEmbedding2DCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._validate_cache(q, cache)
        self._validate_cache(k, cache)
        return _WORLDMIRROR_NORMALIZED_ROPE.apply(q, k, (cache.cos, cache.sin))

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cache = self.prepare_cache(positions, head_dim=tokens.shape[-1], dtype=tokens.dtype)
        return self.apply_single(tokens, cache)
