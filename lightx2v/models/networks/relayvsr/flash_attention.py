# SPDX-License-Identifier: Apache-2.0
# Adapted from kopperx/RelayVSR.
# Copyright RelayVSR contributors.

"""Single-query-frame local attention over condition and video KV planes."""

import torch
import triton
import triton.language as tl


@torch.library.custom_op("lightx2v::relayvsr_flash_attention", mutates_args=(), device_types="cuda")
def _compiled_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, height: int, width: int, windows: list[int]) -> torch.Tensor:
    windows = tuple(zip(windows[::2], windows[1::2]))
    if len(windows) > 1:
        frames = key.shape[2] // (height * width)
        windows = windows[:2] + (windows[2],) * (frames - 2)
    return triton_local_attention(query, key, value, height, width, windows)


@_compiled_attention.register_fake
def _compiled_attention_fake(query, key, value, height, width, windows):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@triton.jit
def _forward(
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
def triton_local_attention(query, key, value, height, width, windows):
    if torch.compiler.is_compiling():
        return _compiled_attention(query, key, value, height, width, [size for window in windows for size in window])
    with torch.cuda.device(query.device):
        q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
        b, hq, _, d = q.shape
        hk = k.shape[1]
        output = torch.empty_like(q)
        whs, wws = tuple(w[0] for w in windows), tuple(w[1] for w in windows)
        bh, bw, bn, warps, stages = 4, 16, 64, 4, 2
        grid = (triton.cdiv(height, bh), triton.cdiv(width, bw), b * hq)
        _forward[grid](
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
