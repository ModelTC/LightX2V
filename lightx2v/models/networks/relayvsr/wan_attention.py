# SPDX-License-Identifier: Apache-2.0
# Adapted from kopperx/RelayVSR.
# Copyright RelayVSR contributors.

"""Exact boundary-preserving spatial attention for inference on CUDA.

Q/K already carry spatial RoPE and their resolved temporal RoPE. Each program handles a 2-D query
tile and directly loads its rectangular KV neighborhood. No replicated
KV windows or quadratic mask are materialized.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class SpatialAttentionSpec:
    frames: int
    height: int
    width: int
    window_height: int
    window_width: int
    temporal_window: int
    key_frames: int
    query_frame_offset: int = 0
    key_frame_offset: int = 0


@torch.library.custom_op("lightx2v::relayvsr_wan_attention", mutates_args=(), device_types="cuda")
def _compiled_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, geometry: list[int]) -> torch.Tensor:
    return spatial_attention(query, key, value, SpatialAttentionSpec(*geometry))


@_compiled_attention.register_fake
def _compiled_attention_fake(query, key, value, geometry):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@triton.jit
def _spatial_forward(
    Q,
    K,
    V,
    OUT,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    FRAMES: tl.constexpr,
    KEY_FRAMES: tl.constexpr,
    QUERY_OFFSET: tl.constexpr,
    KEY_OFFSET: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    WH: tl.constexpr,
    WW: tl.constexpr,
    TW: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    bh = tl.program_id(2)
    batch, head = bh // HEADS, bh % HEADS
    qt, row_tile = row // triton.cdiv(HEIGHT, BH), row % triton.cdiv(HEIGHT, BH)
    qy = row_tile * BH + tl.arange(0, BM) // BW
    qx = segment * BW + tl.arange(0, BM) % BW
    qi = (qt * HEIGHT + qy) * WIDTH + qx
    valid_q = (qx < WIDTH) & (qy < HEIGHT)
    d = tl.arange(0, DIM)
    top = tl.minimum(tl.maximum(qy - WH // 2, 0), HEIGHT - WH)
    left = tl.minimum(tl.maximum(qx - WW // 2, 0), WIDTH - WW)
    base_left = tl.minimum(tl.maximum(segment * BW - WW // 2, 0), WIDTH - WW)
    base_top = tl.minimum(tl.maximum(row_tile * BH - WH // 2, 0), HEIGHT - WH)
    q = tl.load(
        Q + batch * qs0 + qi[:, None] * qs1 + head * qs2 + d[None, :],
        valid_q[:, None],
        0,
    )
    acc = tl.full((BM, DIM), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    logical_qt = qt + QUERY_OFFSET - KEY_OFFSET
    first_t = tl.maximum(0, logical_qt - TW + 1)
    last_t = tl.minimum(KEY_FRAMES, logical_qt + 1)
    for kt in range(first_t, last_t):
        for offset in range(triton.cdiv(ROWS * COLS, BN)):
            p = offset * BN + tl.arange(0, BN)
            ky = base_top + p // COLS
            kx = base_left + p % COLS
            valid_k = (p < ROWS * COLS) & (kx < WIDTH) & (ky < HEIGHT)
            ki = kt * HEIGHT * WIDTH + ky * WIDTH + kx
            k = tl.load(
                K + batch * ks0 + ki[None, :] * ks1 + head * ks2 + d[:, None],
                valid_k[None, :],
                0,
            )
            scores = tl.dot(q, k).to(tl.float32) * SCALE
            allowed = valid_k[None, :] & (ky[None, :] >= top[:, None]) & (ky[None, :] < top[:, None] + WH) & (kx[None, :] >= left[:, None]) & (kx[None, :] < left[:, None] + WW)
            scores = tl.where(allowed, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.exp2(scores - safe_max[:, None])
            correction = tl.exp2(maximum - safe_max)
            v = tl.load(
                V + batch * vs0 + ki[:, None] * vs1 + head * vs2 + d[None, :],
                valid_k[:, None],
                0,
            )
            acc = acc * correction[:, None]
            acc += tl.dot(probabilities.to(v.dtype), v)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = new_max
    output = acc / denominator[:, None]
    tl.store(
        OUT + ((batch * FRAMES * HEIGHT * WIDTH + qi[:, None]) * HEADS + head) * DIM + d[None, :],
        output,
        valid_q[:, None],
    )


def spatial_attention(query, key, value, spec: SpatialAttentionSpec):
    """[B,T*H*W,heads,dim] -> same layout, exact shifted rectangular mask."""
    if torch.compiler.is_compiling():
        geometry = [spec.frames, spec.height, spec.width, spec.window_height, spec.window_width, spec.temporal_window, spec.key_frames, spec.query_frame_offset, spec.key_frame_offset]
        return _compiled_attention(query, key, value, geometry)
    if (
        key.shape != value.shape
        or query.shape[0] != key.shape[0]
        or query.shape[2:] != key.shape[2:]
        or query.dtype != key.dtype
        or key.dtype != value.dtype
        or query.device != key.device
        or key.device != value.device
    ):
        raise ValueError("Spatial attention requires matching batch, heads, dtype and device")
    key_frames = spec.key_frames
    if key.shape[1] != key_frames * spec.height * spec.width:
        raise ValueError("Spatial attention K/V length disagrees with key_frames")
    first_q = spec.query_frame_offset - spec.key_frame_offset
    if first_q < 0 or first_q + spec.frames > key_frames:
        raise ValueError("Every query must have a corresponding key frame")
    b, n, heads, dim = query.shape
    if n != spec.frames * spec.height * spec.width or dim not in (32, 64, 128):
        raise ValueError("Unsupported spatial attention shape")
    if any(t.stride(-1) != 1 for t in (query, key, value)):
        raise ValueError("Spatial attention requires contiguous head dimensions")
    wh, ww = min(spec.window_height, spec.height), min(spec.window_width, spec.width)
    if min(wh, ww, spec.temporal_window) <= 0:
        raise ValueError("Spatial and temporal windows must be positive")
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    bh, bw, bn = 4, 16, 64
    bm = bh * bw
    cols = triton.next_power_of_2(ww + bw - 1)
    rows = min(spec.height, wh + bh - 1)
    _spatial_forward[
        (
            spec.frames * triton.cdiv(spec.height, bh),
            triton.cdiv(spec.width, bw),
            b * heads,
        )
    ](
        query,
        key,
        value,
        output,
        *query.stride()[:3],
        *key.stride()[:3],
        *value.stride()[:3],
        spec.frames,
        key_frames,
        spec.query_frame_offset,
        spec.key_frame_offset,
        spec.height,
        spec.width,
        wh,
        ww,
        spec.temporal_window,
        heads,
        dim,
        dim**-0.5 * 1.4426950408889634,
        bh,
        bw,
        bm,
        bn,
        cols,
        rows,
        num_warps=4,
        num_stages=2,
    )
    return output
