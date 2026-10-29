# SPDX-License-Identifier: Apache-2.0
# Adapted from kopperx/RelayVSR.
# Copyright RelayVSR contributors.

"""Flash model local attention with Triton and PyTorch SDPA backends."""

from __future__ import annotations

from functools import cache

import torch

from .attention import _local_attention


@torch.library.custom_op("lightx2v::relayvsr_flash_model_triton_attention", mutates_args=(), device_types="cuda")
def _flash_model_compiled_triton_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, height: int, width: int, windows: list[int]) -> torch.Tensor:
    windows = tuple(zip(windows[::2], windows[1::2]))
    if len(windows) > 1:
        frames = key.shape[2] // (height * width)
        windows = windows[:2] + (windows[2],) * (frames - 2)
    return flash_model_triton_attention(query, key, value, height, width, windows)


@_flash_model_compiled_triton_attention.register_fake
def _flash_model_compiled_triton_attention_fake(query, key, value, height, width, windows):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@cache
def _get_flash_model_triton_kernel():
    # Triton resolves kernel symbols from module globals. Load them only when
    # the Triton backend executes so SDPA keeps no model-local Triton dependency.
    global triton, tl
    import triton
    import triton.language as tl

    return triton.jit(_flash_model_attention_forward)


def _flash_model_attention_forward(
    Q,
    K,
    V,
    OUT,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    WHS: tl.constexpr,
    WWS: tl.constexpr,
    COLS: tl.constexpr,
    NF: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    n: tl.constexpr = HEIGHT * WIDTH
    tile_y, tile_x, bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = bh // HQ, bh % HQ
    kv_head = head // (HQ // HK)
    row = tl.arange(0, BM)
    qy, qx = tile_y * BH + row // BW, tile_x * BW + row % BW
    qi = qy * WIDTH + qx
    valid_q = (qy < HEIGHT) & (qx < WIDTH)
    d = tl.arange(0, D)
    q = tl.load(
        Q + ((batch * HQ + head) * n + qi[:, None]) * D + d[None, :],
        valid_q[:, None],
        0,
    )
    acc = tl.full((BM, D), 0, tl.float32)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.full((BM,), 0, tl.float32)
    scale: tl.constexpr = D**-0.5 * 1.4426950408889634
    for source in tl.static_range(len(WHS)):
        wh = WHS[source]
        ww = WWS[source]
        rows = wh + BH - 1
        top, left = tile_y * BH - wh // 2, tile_x * BW - ww // 2
        for offset in range(tl.cdiv(rows * COLS, BN)):
            p = offset * BN + tl.arange(0, BN)
            ky, kx = top + p // COLS, left + p % COLS
            valid_k = (p < rows * COLS) & (ky >= 0) & (ky < HEIGHT) & (kx >= 0) & (kx < WIDTH)
            ki = source * n + ky * WIDTH + kx
            k = tl.load(
                K + ((batch * HK + kv_head) * NF * n + ki[None, :]) * D + d[:, None],
                valid_k[None, :],
                0,
            )
            scores = tl.dot(q, k).to(tl.float32) * scale
            allowed = valid_k[None, :] & (ky[None, :] >= qy[:, None] - wh // 2) & (ky[None, :] < qy[:, None] + wh // 2) & (kx[None, :] >= qx[:, None] - ww // 2) & (kx[None, :] < qx[:, None] + ww // 2)
            scores = tl.where(allowed, scores, -float("inf"))
            new_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.exp2(scores - safe_max[:, None])
            correction = tl.exp2(maximum - safe_max)
            v = tl.load(
                V + ((batch * HK + kv_head) * NF * n + ki[:, None]) * D + d[None, :],
                valid_k[:, None],
                0,
            )
            acc = acc * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
            denominator = denominator * correction + tl.sum(probabilities, 1)
            maximum = new_max
    result = acc / denominator[:, None]
    tl.store(
        OUT + ((batch * HQ + head) * n + qi[:, None]) * D + d[None, :],
        result,
        valid_q[:, None],
    )


@torch.no_grad()
def flash_model_triton_attention(query, key, value, height, width, windows):
    if torch.compiler.is_compiling():
        return _flash_model_compiled_triton_attention(query, key, value, height, width, [size for window in windows for size in window])
    kernel = _get_flash_model_triton_kernel()
    with torch.cuda.device(query.device):
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        b, hq, _, d = q.shape
        hk = k.shape[1]
        output = torch.empty_like(q)
        whs, wws = tuple(w[0] for w in windows), tuple(w[1] for w in windows)
        bh, bw, bn, warps, stages = 4, 16, 64, 4, 2
        grid = (triton.cdiv(height, bh), triton.cdiv(width, bw), b * hq)
        kernel[grid](
            q,
            k,
            v,
            output,
            height,
            width,
            hq,
            hk,
            d,
            whs,
            wws,
            triton.next_power_of_2(max(wws) + bw - 1),
            len(windows),
            bh,
            bw,
            bh * bw,
            bn,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=True,
        )
        return output


@torch.library.custom_op("lightx2v::relayvsr_flash_model_sdpa_attention", mutates_args=())
def _flash_model_compiled_sdpa_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, height: int, width: int, windows: list[int], query_start: int) -> torch.Tensor:
    windows = tuple(zip(windows[::2], windows[1::2]))
    if len(windows) > 1:
        frames = key.shape[2] // (height * width)
        windows = windows[:2] + (windows[2],) * (frames - 2)
    return flash_model_sdpa_attention(query, key, value, height, width, windows, query_start)


@_flash_model_compiled_sdpa_attention.register_fake
def _flash_model_compiled_sdpa_attention_fake(query, key, value, height, width, windows, query_start):
    b, heads, tokens, dim = query.shape
    return query.new_empty((b, tokens, heads, dim)).transpose(1, 2)


def flash_model_sdpa_attention(query, key, value, height, width, windows, query_start=0):
    if torch.compiler.is_compiling():
        return _flash_model_compiled_sdpa_attention(query, key, value, height, width, [size for window in windows for size in window], query_start)
    windows = tuple(tuple(window) for window in windows)
    if any(size <= 0 or size % 2 for window in windows for size in window):
        raise ValueError("RelayVSR Flash model requires positive even window dimensions")
    return _local_attention(query, key, value, height, width, windows, shifted=False, query_start=query_start)
