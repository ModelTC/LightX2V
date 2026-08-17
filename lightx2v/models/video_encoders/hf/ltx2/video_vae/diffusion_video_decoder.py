# Copyright 2026 Lightricks Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Native LTX-2.5 diffusion video VAE decoder.

This is a dependency-free (with respect to ``ltx_core``) port of the official
LTX-2.5 eager/chunked-eager path.  The deterministic stages use full-volume 3D
neighborhood attention; the diffusion stage defers the final context upsample,
chunks attention along width, and tiles SwiGLU over tokens to bound peak memory.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from lightx2v.models.video_encoders.hf.ltx2.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from lightx2v.models.video_encoders.hf.ltx2.video_vae.timestep_embedding import (
    PixArtAlphaCombinedTimestepSizeEmbeddings,
)

logger = logging.getLogger(__name__)

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover - host dependent
    natten = None
    _NATTEN_AVAILABLE = False


def _triton_na_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


_DEFAULT_STAGE_CHANNELS = (1024, 512, 256, 256, 128)
_DEFAULT_STAGE_DEPTHS = (4, 6, 4, 2, 8)
_DEFAULT_STAGE_KERNELS = ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (3, 3, 3))
_DEFAULT_UPSAMPLES = (
    ((1, 2, 2), 2),
    ((2, 1, 1), 2),
    ((2, 2, 2), 1),
    ((2, 2, 2), 2),
)
_DEFAULT_STAGE5_KERNEL = (3, 7, 7)
_SWIGLU_TILE_TOKENS = 16_384
_CHUNKED_W_CHUNKS = 4


def _default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 8:
        raise ValueError(f"head_dim={head_dim} must be a multiple of 8")
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _rope_inv_freqs(dim: int, base: float = 10_000.0) -> torch.Tensor:
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return torch.from_numpy(1.0 / np.power(float(base), exponents)).to(torch.float32)


def _rotate_axis(
    x: torch.Tensor,
    positions: torch.Tensor,
    inv_freqs: torch.Tensor,
    axis: int,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    out_dtype = x.dtype
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even = pairs[..., 0].to(compute_dtype)
    odd = pairs[..., 1].to(compute_dtype)
    shape = [1] * x.ndim
    shape[axis] = positions.shape[0]
    shape[-1] = inv_freqs.shape[0]
    angle = (positions[:, None] * inv_freqs[None, :]).reshape(shape)
    cos, sin = angle.cos().to(compute_dtype), angle.sin().to(compute_dtype)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).reshape(x.shape)
    return rotated.to(out_dtype) if rotated.dtype != out_dtype else rotated


def _apply_abs_rope_slab(
    x: torch.Tensor,
    rope_dim_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    w_positions: torch.Tensor,
) -> torch.Tensor:
    d_t, d_h, _ = rope_dim_split
    inv_t, inv_h, inv_w = inv_freqs
    t_positions = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
    h_positions = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
    x_t = _rotate_axis(x[..., :d_t], t_positions, inv_t, axis=1)
    x_h = _rotate_axis(x[..., d_t : d_t + d_h], h_positions, inv_h, axis=2)
    x_w = _rotate_axis(x[..., d_t + d_h :], w_positions, inv_w, axis=3)
    return torch.cat((x_t, x_h, x_w), dim=-1)


def _apply_abs_rope(
    x: torch.Tensor,
    rope_dim_split: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_w_tiles: int = 4,
) -> torch.Tensor:
    chunks = torch.chunk(x, min(num_w_tiles, x.shape[3]), dim=3)
    result: list[torch.Tensor] = []
    offset = 0
    for chunk in chunks:
        positions = torch.arange(chunk.shape[3], dtype=torch.float32, device=x.device) + offset
        result.append(_apply_abs_rope_slab(chunk, rope_dim_split, inv_freqs, positions))
        offset += chunk.shape[3]
    return torch.cat(result, dim=3)


# The eager NA fallback below is derived from comfy-kitchen's Apache-2.0
# ``backends/eager/na.py``.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
_NA_SCORE_BUDGET = 2**25
_NA_KV_STACK_BUDGET = 2**28


def _window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    kernel = min(kernel, length)
    upper = length - kernel
    half = kernel // 2
    starts = [min(max(i - half, 0), upper) for i in range(length)]
    return starts, [start + kernel for start in starts]


def _pick_na_tiles(dims: tuple[int, int, int], kernels: tuple[int, int, int]) -> list[int]:
    tiles = list(dims)

    def cost(sizes: list[int]) -> int:
        queries = math.prod(sizes)
        keys = math.prod(min(dim, size + kernel - 1) for size, kernel, dim in zip(sizes, kernels, dims))
        return queries * keys

    while cost(tiles) > _NA_SCORE_BUDGET and max(tiles) > 1:
        axis = max(range(3), key=lambda i: tiles[i] / kernels[i])
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)
    return tiles


def _na_group_mask(
    relative_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    visible_axes = []
    for starts, ends in relative_bounds:
        starts_tensor = torch.tensor(starts, device=device)
        ends_tensor = torch.tensor(ends, device=device)
        key_index = torch.arange(int(ends_tensor.max()), device=device)
        visible_axes.append((key_index[None] >= starts_tensor[:, None]) & (key_index[None] < ends_tensor[:, None]))
    visible = visible_axes[0][:, None, None, :, None, None] & visible_axes[1][None, :, None, None, :, None] & visible_axes[2][None, None, :, None, None, :]
    n_queries = math.prod(visible.shape[:3])
    n_keys = math.prod(visible.shape[3:])
    mask = torch.zeros((n_queries, n_keys), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(n_queries, n_keys), torch.finfo(dtype).min)
    return mask.reshape(1, 1, n_queries, n_keys)


def _eager_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    batch, time, height, width, num_heads, head_dim = q.shape
    dims = (time, height, width)
    kernels = tuple(min(kernel, dim) for kernel, dim in zip(kernel_size, dims))
    bounds = [_window_bounds(dim, kernel) for dim, kernel in zip(dims, kernels)]
    tile_t, tile_h, tile_w = _pick_na_tiles(dims, kernels)
    groups: dict[tuple, list[tuple[tuple[slice, ...], tuple[slice, ...]]]] = {}

    for t0 in range(0, time, tile_t):
        t1 = min(t0 + tile_t, time)
        key_t0, key_t1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = (
            tuple(start - key_t0 for start in bounds[0][0][t0:t1]),
            tuple(end - key_t0 for end in bounds[0][1][t0:t1]),
        )
        for h0 in range(0, height, tile_h):
            h1 = min(h0 + tile_h, height)
            key_h0, key_h1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = (
                tuple(start - key_h0 for start in bounds[1][0][h0:h1]),
                tuple(end - key_h0 for end in bounds[1][1][h0:h1]),
            )
            for w0 in range(0, width, tile_w):
                w1 = min(w0 + tile_w, width)
                key_w0, key_w1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = (
                    tuple(start - key_w0 for start in bounds[2][0][w0:w1]),
                    tuple(end - key_w0 for end in bounds[2][1][w0:w1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(key_t0, key_t1), slice(key_h0, key_h1), slice(key_w0, key_w1)),
                    )
                )

    output = torch.empty_like(v)
    for relative, tiles in groups.items():
        mask = _na_group_mask(relative, q.dtype, q.device)
        n_queries, n_keys = mask.shape[2:]
        max_group = max(1, _NA_KV_STACK_BUDGET // max(1, batch * num_heads * n_keys * head_dim * 2)) if q.device.type == "cuda" else 1
        query_slices = tiles[0][0]
        query_shape = tuple(item.stop - item.start for item in query_slices)
        for group_start in range(0, len(tiles), max_group):
            chunk = tiles[group_start : group_start + max_group]
            group_size = len(chunk)
            q_group = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_group = torch.stack([k[:, ks[0], ks[1], ks[2]] for _, ks in chunk])
            v_group = torch.stack([v[:, ks[0], ks[1], ks[2]] for _, ks in chunk])
            q_group = q_group.permute(0, 1, 5, 2, 3, 4, 6).reshape(group_size * batch, num_heads, n_queries, head_dim)
            k_group = k_group.permute(0, 1, 5, 2, 3, 4, 6).reshape(group_size * batch, num_heads, n_keys, head_dim)
            v_group = v_group.permute(0, 1, 5, 2, 3, 4, 6).reshape(group_size * batch, num_heads, n_keys, head_dim)
            attended = F.scaled_dot_product_attention(q_group, k_group, v_group, attn_mask=mask, scale=1.0)
            attended = attended.view(group_size, batch, num_heads, *query_shape, head_dim).permute(0, 1, 3, 4, 5, 2, 6)
            for index, (qs, _) in enumerate(chunk):
                output[:, qs[0], qs[1], qs[2]] = attended[index]
    return output


class QKVProjections(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_q(x), self.to_k(x), self.to_v(x)


class NeighborhoodAttention3D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if dim % head_dim:
            raise ValueError(f"dim={dim} is not divisible by head_dim={head_dim}")
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = rope_dim_split or _default_rope_dim_split(head_dim)
        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.register_buffer("rope_inv_t", _rope_inv_freqs(self.rope_dim_split[0]), persistent=False)
        self.register_buffer("rope_inv_h", _rope_inv_freqs(self.rope_dim_split[1]), persistent=False)
        self.register_buffer("rope_inv_w", _rope_inv_freqs(self.rope_dim_split[2]), persistent=False)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q, k = q.to(v.dtype), k.to(v.dtype)
        if _NATTEN_AVAILABLE:
            try:
                return natten.na3d(
                    q,
                    k,
                    v,
                    kernel_size=self.kernel_size,
                    scale=1.0,
                    backend="cutlass-fna",
                )
            except TypeError:  # compatibility with older NATTEN builds
                return natten.na3d(q, k, v, kernel_size=self.kernel_size, scale=1.0)
        if _triton_na_available():
            from lightx2v.models.video_encoders.hf.ltx2.video_vae.triton_na import na3d as triton_na3d

            return triton_na3d(q, k, v, kernel_size=self.kernel_size, scale=1.0)
        return _eager_na3d(q, k, v, self.kernel_size)

    def _project_with_rope(self, x: torch.Tensor, w_positions: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, height, width, _ = x.shape
        shape = (batch, time, height, width, self.num_heads, self.head_dim)
        q, k, v = self.qkv(x)
        q = self.q_norm(q.view(shape)) * self.scale
        k = self.k_norm(k.view(shape))
        v = v.view(shape)
        inv_freqs = (
            self.rope_inv_t.to(x.device),
            self.rope_inv_h.to(x.device),
            self.rope_inv_w.to(x.device),
        )
        if w_positions is None:
            q = _apply_abs_rope(q, self.rope_dim_split, inv_freqs)
            k = _apply_abs_rope(k, self.rope_dim_split, inv_freqs)
        else:
            q = _apply_abs_rope_slab(q, self.rope_dim_split, inv_freqs, w_positions)
            k = _apply_abs_rope_slab(k, self.rope_dim_split, inv_freqs, w_positions)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def forward(self, x: torch.Tensor, w_positions: torch.Tensor | None = None) -> torch.Tensor:
        time, height, width = x.shape[1:4]
        if any(size < kernel for size, kernel in zip((time, height, width), self.kernel_size)):
            raise ValueError(f"Neighborhood attention input {(time, height, width)} is smaller than {self.kernel_size}")
        q, k, v = self._project_with_rope(x, w_positions)
        output = self._attention(q, k, v).reshape(*x.shape[:-1], self.dim)
        return self.proj(output)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from lightx2v.models.video_encoders.hf.ltx2.video_vae.triton_swiglu import swiglu_tiled

        return swiglu_tiled(
            x,
            self.w_gate.weight,
            self.w_up.weight,
            self.w_down.weight,
            _SWIGLU_TILE_TOKENS,
        )


class NABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int,
        rope_dim_split: tuple[int, int, int] | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim, rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden_dim = (int(dim * 4.0) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ChannelLinear(nn.Linear):
    @property
    def in_channels(self) -> int:
        return self.in_features

    @property
    def out_channels(self) -> int:
        return self.out_features


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int,
    ) -> None:
        super().__init__()
        self.stride = tuple(stride)
        projected_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.proj = nn.Linear(in_channels, projected_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        p_t, p_h, p_w = self.stride
        x = rearrange(
            self.proj(x),
            "b t h w (c pt ph pw) -> b (t pt) (h ph) (w pw) c",
            pt=p_t,
            ph=p_h,
            pw=p_w,
        )
        if p_t == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x


class AdaLNZero(nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.NUM_CHUNKS, dim=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class ChunkedDiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int,
        rope_dim_split: tuple[int, int, int] | None,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim, rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden_dim = (int(dim * 4.0) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden_dim)
        self.attn.proj.reset_parameters()

    def _modulation(self, modulation: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = [modulation[index] + self.scale_shift_table[index].view(1, 1, 1, 1, -1) for index in range(AdaLNZero.NUM_CHUNKS)]
        return values[0], values[1], values[3], values[4]

    def _inject_context(
        self,
        x: torch.Tensor,
        stage4_features: torch.Tensor,
        upsample: LinearPixelShuffleUpsample,
        drop_leading_frame: bool,
    ) -> None:
        p_t, p_h, p_w = upsample.stride
        chunks = torch.chunk(stage4_features, _CHUNKED_W_CHUNKS, dim=3)
        output_start = 0
        for chunk in chunks:
            context = F.linear(chunk, upsample.proj.weight, upsample.proj.bias)
            context = rearrange(
                context,
                "b t h w (c pt ph pw) -> b (t pt) (h ph) (w pw) c",
                pt=p_t,
                ph=p_h,
                pw=p_w,
            )
            if p_t == 2 and drop_leading_frame:
                context = context[:, 1:]
            context = self.context_proj(context)
            output_end = min(x.shape[3], output_start + chunk.shape[3] * p_w)
            x[:, :, :, output_start:output_end].add_(context[:, :, :, : output_end - output_start])
            output_start = output_end

    def _attention_residual(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> None:
        width = x.shape[3]
        chunk_width = (width + _CHUNKED_W_CHUNKS - 1) // _CHUNKED_W_CHUNKS
        halo = self.attn.kernel_size[2] // 2
        extent = chunk_width + 2 * halo
        left_halo: torch.Tensor | None = None

        for chunk_index in range(_CHUNKED_W_CHUNKS):
            core_start = chunk_index * chunk_width
            core_end = min(width, (chunk_index + 1) * chunk_width)
            core_length = core_end - core_start
            if core_length <= 0:
                continue
            buffer = x.new_zeros(*x.shape[:3], extent, x.shape[-1])
            if left_halo is not None:
                buffer[:, :, :, halo - left_halo.shape[3] : halo] = left_halo
            buffer[:, :, :, halo : halo + core_length] = x[:, :, :, core_start:core_end]

            right_filled = 0
            if core_end < width:
                right = x[:, :, :, core_end : min(width, core_end + halo)]
                right_filled = right.shape[3]
                buffer[:, :, :, halo + core_length : halo + core_length + right_filled] = right
            if chunk_index == 0 and halo:
                buffer[:, :, :, :halo] = buffer[:, :, :, halo : halo + 1].expand(*x.shape[:3], halo, x.shape[-1])
            missing_right = extent - (halo + core_length + right_filled)
            if missing_right:
                edge = buffer[:, :, :, halo + core_length - 1 : halo + core_length]
                buffer[:, :, :, halo + core_length + right_filled :] = edge.expand(*x.shape[:3], missing_right, x.shape[-1])
            if core_end < width:
                left_halo = x[:, :, :, max(core_start, core_end - halo) : core_end].clone()

            normalized = self.norm1(buffer)
            normalized = normalized * (1.0 + scale) + shift
            w_positions = torch.arange(extent, dtype=torch.float32, device=x.device) + core_start - halo
            attended = self.attn(normalized, w_positions=w_positions)
            x[:, :, :, core_start:core_end].add_(attended[:, :, :, halo : halo + core_length])

    def _mlp_residual(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        from lightx2v.models.video_encoders.hf.ltx2.video_vae.triton_swiglu import residual_modulating_mlp

        return residual_modulating_mlp(
            x,
            self.norm2.weight,
            scale,
            shift,
            self.mlp.w_gate.weight,
            self.mlp.w_up.weight,
            self.mlp.w_down.weight,
            _SWIGLU_TILE_TOKENS,
        )

    def forward(
        self,
        x: torch.Tensor,
        stage4_features: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        upsample: LinearPixelShuffleUpsample,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        self._inject_context(x, stage4_features, upsample, drop_leading_frame)
        self._attention_residual(x, scale_msa, shift_msa)
        return self._mlp_residual(x, scale_mlp, shift_mlp)


@dataclass(frozen=True)
class _AxisPad:
    before: int = 0
    after: int = 0


def _resize_axis(
    x: torch.Tensor,
    dim: int,
    size: int,
    mode: Literal["repeat_last", "symmetric"],
) -> tuple[torch.Tensor, _AxisPad]:
    length = x.shape[dim]
    if length == size:
        return x, _AxisPad()
    if length < size:
        needed = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            shape = list(x.shape)
            shape[dim] = needed
            return torch.cat((x, last.expand(shape)), dim=dim), _AxisPad(after=needed)
        before, after = needed // 2, needed - needed // 2
        parts = []
        shape = list(x.shape)
        if before:
            shape[dim] = before
            parts.append(x.narrow(dim, 0, 1).expand(shape))
        parts.append(x)
        if after:
            shape[dim] = after
            parts.append(x.narrow(dim, length - 1, 1).expand(shape))
        return torch.cat(parts, dim=dim), _AxisPad(before, after)
    needed = length - size
    before = 0 if mode == "repeat_last" else needed // 2
    return x.narrow(dim, before, size).contiguous(), _AxisPad(before, needed - before)


def _all_stages_min_size(
    stage_kernels: tuple[tuple[int, int, int], ...],
    upsamples: tuple[tuple[tuple[int, int, int], int], ...],
    stage5_kernel: tuple[int, int, int],
) -> tuple[int, int, int]:
    cumulative = [(1, 1, 1)]
    current = [1, 1, 1]
    for stride, _ in upsamples:
        current = [value * factor for value, factor in zip(current, stride)]
        cumulative.append(tuple(current))
    minimum = [1, 1, 1]
    for stage_index in range(len(upsamples)):
        for axis in range(3):
            minimum[axis] = max(minimum[axis], math.ceil(stage_kernels[stage_index][axis] / cumulative[stage_index][axis]))
    for axis in range(3):
        minimum[axis] = max(minimum[axis], math.ceil(stage5_kernel[axis] / cumulative[-1][axis]))
    return tuple(minimum)


class DiffusionVideoDecoder(nn.Module):
    """LTX-2.5 NADiffusionDecoder, defaulting to official ``chunked_eager``."""

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: tuple[int, int, int] | None = None,
        stage_channels: tuple[int, ...] = _DEFAULT_STAGE_CHANNELS,
        stage_depths: tuple[int, ...] = _DEFAULT_STAGE_DEPTHS,
        stage_kernels: tuple[tuple[int, int, int], ...] = _DEFAULT_STAGE_KERNELS,
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = _DEFAULT_UPSAMPLES,
        stage5_kernel: tuple[int, int, int] = _DEFAULT_STAGE5_KERNEL,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 2,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: Literal["v", "x0"] = "v",
    ) -> None:
        super().__init__()
        if not (len(stage_channels) == len(stage_depths) == len(stage_kernels)):
            raise ValueError("stage_channels, stage_depths, and stage_kernels must have equal length")
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.stage_channels = tuple(stage_channels)
        self.stage_depths = tuple(stage_depths)
        self.stage5_kernel = tuple(stage5_kernel)
        self.causal = False
        self.timestep_conditioning = True
        self.video_downscale_factors = (8, 32, 32)
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(
                1.0,
                1.0 / default_num_inference_steps,
                default_num_inference_steps,
                device="cpu",
            ),
            persistent=False,
        )
        self._trailing_latent_frames = (stage_kernels[0][0] // 2) * 2
        self._stage_min_sizes = _all_stages_min_size(stage_kernels, upsamples, self.stage5_kernel)
        upsample4_stride = upsamples[3][0]
        self._tile_min_sizes = tuple(max(stage_kernels[3][axis], math.ceil(self.stage5_kernel[axis] / upsample4_stride[axis])) for axis in range(3))

        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.conv_in = ChannelLinear(in_channels, stage_channels[0], bias=True)
        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_index in range(len(stage_channels) - 1):
            dim = stage_channels[stage_index]
            self.det_stages.append(nn.ModuleList([NABlock(dim, stage_kernels[stage_index], head_dim, rope_dim_split) for _ in range(stage_depths[stage_index])]))
            stride, reduction = upsamples[stage_index]
            self.upsamples.append(LinearPixelShuffleUpsample(dim, stride, reduction))

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(t_emb_dim, 0)
        context_channels = stage_channels[-1]
        diff_channels = stage5_channels or context_channels
        noised_pixel_channels = out_channels * patch_size**2
        self.conv_in_x_t = ChannelLinear(noised_pixel_channels, diff_channels, bias=True)
        self.shared_adaln = AdaLNZero(diff_channels, t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [
                ChunkedDiffusionNABlock(
                    diff_channels,
                    self.stage5_kernel,
                    context_channels,
                    head_dim,
                    rope_dim_split,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(diff_channels, eps=1e-6)
        self.conv_out = ChannelLinear(diff_channels, noised_pixel_channels, bias=True)

        if not _NATTEN_AVAILABLE and _triton_na_available():
            logger.warning("NATTEN is unavailable; LTX-2.5 DiffVAE will use the Triton na3d fallback.")
        elif not _NATTEN_AVAILABLE:
            logger.warning("NATTEN is unavailable; LTX-2.5 DiffVAE will use the numerically aligned pure-PyTorch neighborhood-attention fallback, which is substantially slower.")

    def _run_det_stage(self, x: torch.Tensor, stage_index: int, *, drop_leading_frame: bool = True) -> torch.Tensor:
        for block in self.det_stages[stage_index]:
            x = block(x)
        return self.upsamples[stage_index](x, drop_leading_frame=drop_leading_frame)

    def _forward_stages_1_to_3(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.per_channel_statistics.un_normalize(latent).permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_index in range(3):
            x = self._run_det_stage(x, stage_index)
        return x

    def _forward_stage_4(self, x: torch.Tensor, *, pad_trailing: bool = True) -> torch.Tensor:
        for block in self.det_stages[3]:
            x = block(x)
        if not pad_trailing:
            return x
        ghost_frames = self._trailing_latent_frames * (8 // self.upsamples[3].stride[0])
        content_frames = max(x.shape[1] - ghost_frames, 1)
        minimum_frames = max(1, math.ceil(self.stage5_kernel[0] / self.upsamples[3].stride[0]))
        keep = min(x.shape[1], max(content_frames, minimum_frames))
        return x[:, :keep].contiguous()

    def _diffusion_prediction(
        self,
        pixels: torch.Tensor,
        stage4_features: torch.Tensor,
        timestep: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        patched = patchify(pixels, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(patched.permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * timestep, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(
                x,
                stage4_features,
                modulation,
                self.upsamples[3],
                drop_leading_frame=drop_leading_frame,
            )
        x = self.conv_out(self.norm_out(x)).permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def _euler_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> torch.Tensor:
        shape = (-1,) + (1,) * (sample.ndim - 1)
        sigma = timestep.view(shape).float()
        dt = (timestep - next_timestep).view(shape).float()
        sample_fp32 = sample.float()
        velocity = model_output.float()
        if self.model_output_type == "x0":
            velocity = (sample_fp32 - velocity) / sigma
        return (sample_fp32 - dt * velocity).to(sample.dtype)

    def _ensure_min_shape(self, latent: torch.Tensor) -> tuple[torch.Tensor, _AxisPad, _AxisPad]:
        latent, _ = _resize_axis(latent, 2, max(latent.shape[2], self._stage_min_sizes[0]), "repeat_last")
        latent, h_pad = _resize_axis(latent, 3, max(latent.shape[3], self._stage_min_sizes[1]), "symmetric")
        latent, w_pad = _resize_axis(latent, 4, max(latent.shape[4], self._stage_min_sizes[2]), "symmetric")
        return latent, h_pad, w_pad

    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample.ndim != 5:
            raise ValueError(f"Expected DiffVAE latent [B,C,T,H,W], got {tuple(sample.shape)}")
        content_frames = (sample.shape[2] - 1) * 8 + 1
        content_height, content_width = sample.shape[3] * 32, sample.shape[4] * 32
        latent, h_pad, w_pad = self._ensure_min_shape(sample)

        trailing = latent[:, :, -1:].expand(*latent.shape[:2], self._trailing_latent_frames, *latent.shape[3:])
        stage4_features = self._forward_stages_1_to_3(torch.cat((latent, trailing), dim=2))
        stage4_features = self._forward_stage_4(stage4_features)

        stride_t, stride_h, stride_w = self.upsamples[3].stride
        pixel_shape = (
            stage4_features.shape[1] * stride_t - (1 if stride_t == 2 else 0),
            stage4_features.shape[2] * stride_h * self.patch_size,
            stage4_features.shape[3] * stride_w * self.patch_size,
        )
        pixel_shape = (max(pixel_shape[0], self.stage5_kernel[0]), pixel_shape[1], pixel_shape[2])
        random_device = generator.device if generator is not None else sample.device
        pixels = torch.randn(
            (sample.shape[0], self.out_channels, *pixel_shape),
            dtype=sample.dtype,
            device=random_device,
            generator=generator,
        ).to(sample.device)

        timesteps = self.default_inference_timesteps.to(sample.device).unsqueeze(0).expand(sample.shape[0], -1)
        for step in range(timesteps.shape[1] - 1):
            prediction = self._diffusion_prediction(pixels, stage4_features, timesteps[:, step])
            pixels = self._euler_step(pixels, prediction, timesteps[:, step], timesteps[:, step + 1])
        prediction = self._diffusion_prediction(pixels, stage4_features, timesteps[:, -1])
        if self.model_output_type == "x0":
            pixels = prediction
        else:
            pixels = self._euler_step(pixels, prediction, timesteps[:, -1], torch.zeros_like(timesteps[:, -1]))

        h_start, w_start = h_pad.before * 32, w_pad.before * 32
        pixels = pixels[
            :,
            :,
            :content_frames,
            h_start : h_start + content_height,
            w_start : w_start + content_width,
        ].contiguous()
        # The official untiled path still routes the sole tile through its
        # blend accumulator.  BF16 decoder activations are accumulated in FP16
        # and converted back on emit, so preserve that final rounding step.
        accumulator_dtype = torch.float16 if pixels.dtype == torch.bfloat16 else pixels.dtype
        return pixels.to(accumulator_dtype).to(sample.dtype)

    def _decode_one_tile(
        self,
        stage4_features: torch.Tensor,
        pixels: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        is_origin: bool,
        pad_trailing: bool,
    ) -> torch.Tensor:
        context = self._forward_stage_4(stage4_features, pad_trailing=pad_trailing)
        for step in range(timesteps.shape[1] - 1):
            prediction = self._diffusion_prediction(
                pixels,
                context,
                timesteps[:, step],
                drop_leading_frame=is_origin,
            )
            pixels = self._euler_step(pixels, prediction, timesteps[:, step], timesteps[:, step + 1])
        prediction = self._diffusion_prediction(
            pixels,
            context,
            timesteps[:, -1],
            drop_leading_frame=is_origin,
        )
        if self.model_output_type == "x0":
            return prediction
        return self._euler_step(pixels, prediction, timesteps[:, -1], torch.zeros_like(timesteps[:, -1]))

    @torch.no_grad()
    def tiled_decode(
        self,
        sample: torch.Tensor,
        tiling_config,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode DiffVAE stage 4/5 tiles and yield blended temporal chunks."""
        from lightx2v.models.video_encoders.hf.ltx2.video_vae import diffusion_tiling

        if sample.ndim != 5:
            raise ValueError(f"Expected DiffVAE latent [B,C,T,H,W], got {tuple(sample.shape)}")

        content_frames = (sample.shape[2] - 1) * 8 + 1
        content_height, content_width = sample.shape[3] * 32, sample.shape[4] * 32
        latent, h_pad, w_pad = self._ensure_min_shape(sample)
        work_frames = (latent.shape[2] - 1) * 8 + 1
        work_height, work_width = latent.shape[3] * 32, latent.shape[4] * 32

        strides = tuple(tuple(module.stride) for module in self.upsamples)
        stage4_t, stage4_h, stage4_w = diffusion_tiling.stage4_shape_from_latent(
            latent.shape[2],
            latent.shape[3],
            latent.shape[4],
            strides,
        )
        trailing = latent[:, :, -1:].expand(
            *latent.shape[:2],
            self._trailing_latent_frames,
            *latent.shape[3:],
        )
        stage4_features = self._forward_stages_1_to_3(torch.cat((latent, trailing), dim=2))
        tiles = diffusion_tiling.prepare_tile_schedule(
            torch.Size((latent.shape[0], stage4_t, stage4_h, stage4_w, stage4_features.shape[-1])),
            tiling_config,
            pixel_height=work_height,
            pixel_width=work_width,
            upsample_stride=tuple(self.upsamples[3].stride),
            patch_size=self.patch_size,
            min_tile_size=self._tile_min_sizes,
        )
        groups = diffusion_tiling.group_tiles_by_temporal_slice(tiles)
        timesteps = self.default_inference_timesteps.to(sample.device).unsqueeze(0).expand(sample.shape[0], -1)
        full_shape = (sample.shape[0], self.out_channels, work_frames, work_height, work_width)
        accumulator_dtype = torch.float16 if sample.dtype == torch.bfloat16 else sample.dtype
        random_device = generator.device if generator is not None else sample.device

        h_start, w_start = h_pad.before * 32, w_pad.before * 32

        def emit(
            buffer: torch.Tensor,
            weights: torch.Tensor,
            global_start: int,
        ) -> torch.Tensor | None:
            if global_start >= content_frames or buffer.shape[2] == 0:
                return None
            keep = min(buffer.shape[2], content_frames - global_start)
            weights = weights[:, :, :keep].clamp_min(torch.finfo(weights.dtype).tiny)
            chunk = (buffer[:, :, :keep] / weights).to(sample.dtype)
            return chunk[
                :,
                :,
                :,
                h_start : h_start + content_height,
                w_start : w_start + content_width,
            ].contiguous()

        previous_buffer: torch.Tensor | None = None
        previous_weights: torch.Tensor | None = None
        previous_slice: slice | None = None

        for group in groups:
            temporal_slice = group[0].out_coords[2]
            group_start, group_stop, _ = temporal_slice.indices(work_frames)
            group_frames = group_stop - group_start
            buffer = torch.zeros(
                sample.shape[0],
                self.out_channels,
                group_frames,
                work_height,
                work_width,
                device=sample.device,
                dtype=accumulator_dtype,
            )
            weights = torch.zeros(
                sample.shape[0],
                1,
                group_frames,
                work_height,
                work_width,
                device=sample.device,
                dtype=accumulator_dtype,
            )

            for tile in group:
                _, t_coord, h_coord, w_coord, _ = tile.in_coords
                t0, t1, _ = t_coord.indices(stage4_t)
                h0, h1, _ = h_coord.indices(stage4_h)
                w0, w1, _ = w_coord.indices(stage4_w)
                is_origin = t0 == 0
                pad_trailing = t1 == stage4_t
                feature_stop = stage4_features.shape[1] if pad_trailing else t1
                feature_tile = stage4_features[:, t0:feature_stop, h0:h1, w0:w1]

                output_shape = diffusion_tiling.tile_shape(full_shape, tile.out_coords)
                stride_t, stride_h, stride_w = self.upsamples[3].stride
                canvas_frames = (t1 - t0) * stride_t - (1 if is_origin and stride_t == 2 else 0)
                if pad_trailing:
                    canvas_frames = max(canvas_frames, self.stage5_kernel[0])
                canvas_height = (h1 - h0) * stride_h * self.patch_size
                canvas_width = (w1 - w0) * stride_w * self.patch_size
                pixels = torch.randn(
                    (sample.shape[0], self.out_channels, canvas_frames, canvas_height, canvas_width),
                    dtype=sample.dtype,
                    device=random_device,
                    generator=generator,
                ).to(sample.device)
                decoded = self._decode_one_tile(
                    feature_tile,
                    pixels,
                    timesteps,
                    is_origin=is_origin,
                    pad_trailing=pad_trailing,
                )
                decoded = decoded[:, :, : output_shape[2], : output_shape[3], : output_shape[4]]
                mask = diffusion_tiling.separable_mask(tile, device=sample.device, dtype=accumulator_dtype)

                out_t = tile.out_coords[2]
                out_h = tile.out_coords[3]
                out_w = tile.out_coords[4]
                local_t = slice(out_t.start - group_start, out_t.stop - group_start)
                coords = (slice(None), slice(None), local_t, out_h, out_w)
                buffer[coords] += decoded.to(accumulator_dtype) * mask
                weight_coords = (slice(None), slice(None), local_t, out_h, out_w)
                weights[weight_coords] += mask[:, :1]

            if previous_buffer is not None:
                assert previous_weights is not None and previous_slice is not None
                overlap = max(0, previous_slice.stop - group_start)
                if overlap:
                    previous_buffer[:, :, -overlap:] += buffer[:, :, :overlap]
                    previous_weights[:, :, -overlap:] += weights[:, :, :overlap]
                    buffer[:, :, :overlap] = previous_buffer[:, :, -overlap:]
                    weights[:, :, :overlap] = previous_weights[:, :, -overlap:]
                exclusive = max(0, group_start - previous_slice.start)
                chunk = emit(previous_buffer[:, :, :exclusive], previous_weights[:, :, :exclusive], previous_slice.start)
                if chunk is not None:
                    yield chunk

            previous_buffer = buffer
            previous_weights = weights
            previous_slice = slice(group_start, group_stop)

        if previous_buffer is not None:
            assert previous_weights is not None and previous_slice is not None
            chunk = emit(previous_buffer, previous_weights, previous_slice.start)
            if chunk is not None:
                yield chunk


__all__ = ["DiffusionVideoDecoder"]
