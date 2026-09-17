"""Tensor kernels ported from OpenVDN/vdn-minimax-h3 (Apache-2.0).

Source e02ff077: models/ops/temporal_conv.py and
models/linear_attention/kernels.py. Only VDN-specific operations live here;
block pointwise operations and QK norm/RoPE use the shared H3 implementation.
The functions consume LightX2V weight tensors; no upstream model is imported.
"""

from functools import cache

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


@cache
def _compiled(fn):
    return torch.compile(fn, dynamic=False)


def _run(fn, *args):
    return _compiled(fn)(*args) if args[0].is_cuda else fn(*args)


def _activate_body(x, normalize):
    y = F.silu(x)
    return F.normalize(y, dim=-1, eps=1e-6).to(y.dtype) if normalize else y


def activate(x, normalize):
    return _run(_activate_body, x, normalize)


def _epilogue_body(readout, weight, gate):
    ms = torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + 1e-6).to(readout.dtype) * weight.to(readout.dtype)
    frames, heads, spatial, dim = normed.shape
    return normed.permute(0, 2, 1, 3).reshape(frames * spatial, heads, dim) * gate


def linear_epilogue(readout, weight, gate):
    return _run(_epilogue_body, readout, weight, gate)


if triton is not None:

    @triton.jit
    def _temporal_kernel(X, W, OUT, T, S, C, D: tl.constexpr, L2: tl.constexpr, BLOCK_T: tl.constexpr):
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        channels = tl.program_id(2) * D + tl.arange(0, D)
        spatial = tl.program_id(1)
        acc = tl.zeros((BLOCK_T, D), tl.float32)
        for tap in tl.static_range(5):
            frames = rows + tap - 2
            x = tl.load(X + (frames[:, None] * S + spatial) * C + channels[None], mask=(rows[:, None] < T) & (frames[:, None] >= 0) & (frames[:, None] < T), other=0).to(tl.float32)
            w = tl.load(W + channels * 5 + tap).to(tl.float32)
            acc += x * w[None]
        y = acc * tl.sigmoid(acc)
        if L2:
            y *= tl.rsqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))[:, None]
        tl.store(OUT + (rows[:, None] * S + spatial) * C + channels[None], y, mask=rows[:, None] < T)


def temporal_activate(x, weight, heads, dim, normalize):
    """Symmetric five-tap depthwise temporal conv, SiLU and optional L2 norm."""
    frames, spatial, channels = x.shape
    if x.is_cuda:
        if triton is None:
            raise ImportError("VDN CUDA inference requires Triton")
        if not x.is_contiguous() or not weight.is_contiguous():
            raise ValueError("VDN temporal convolution requires contiguous inputs")
        out = torch.empty_like(x)
        _temporal_kernel[(triton.cdiv(frames, 16), spatial, heads)](x, weight, out, frames, spatial, channels, D=dim, L2=normalize, BLOCK_T=16, num_warps=4, num_stages=2)
        return out.reshape(-1, heads, dim)
    padded = F.pad(x.float(), (0, 0, 0, 0, 2, 2))
    out = sum(padded[tap : tap + frames] * weight[:, tap].float() for tap in range(5))
    return _activate_body(out.reshape(-1, heads, dim), normalize).to(x.dtype)
