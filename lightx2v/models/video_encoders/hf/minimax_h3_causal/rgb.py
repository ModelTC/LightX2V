"""Fused uint8 conversion with original sensitive-dtype rounding at each stage."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rgb(
    X,
    MEAN,
    STD,
    OUT,
    N,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    SB: tl.constexpr,
    SC: tl.constexpr,
    ST: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    SENSITIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % 3
    pixel = i // 3
    w = pixel % W
    h = pixel // W % H
    t = pixel // (W * H) % T
    b = pixel // (W * H * T)
    value = tl.load(X + b * SB + c * SC + t * ST + h * SH + w * SW, i < N, 0).to(SENSITIVE).to(tl.float32)
    scale = tl.load(STD + c).to(tl.float32)
    mean = tl.load(MEAN + c).to(tl.float32)
    value = (value * scale).to(SENSITIVE).to(tl.float32)
    value = (value + mean).to(SENSITIVE).to(tl.float32)
    value = tl.minimum(tl.maximum(value, 0.0), 1.0)
    value = libdevice.nearbyint(value * 255.0)
    tl.store(OUT + i, value.to(tl.uint8), i < N)


def to_uint8(video, mean, std, sensitive_dtype):
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("Expected [B,3,T,H,W] RGB")
    dtype = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[sensitive_dtype]
    output = torch.empty(video.shape, device=video.device, dtype=torch.uint8, memory_format=torch.channels_last_3d)
    _rgb[(triton.cdiv(output.numel(), 1024),)](video, mean.to(video.device), std.to(video.device), output, output.numel(), *video.shape[2:], *video.stride(), dtype, 1024, enable_fp_fusion=False)
    return output
