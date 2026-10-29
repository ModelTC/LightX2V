"""CUDA L2 normalization with FP32 accumulation and a selected output dtype."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _l2_normalize_kernel(x, output, rows: tl.constexpr, width: tl.constexpr, xs0: tl.constexpr, xs1: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, block)
    value = tl.load(x + row * xs0 + cols * xs1, mask=cols < width, other=0).to(tl.float32)
    norm = libdevice.sqrt_rn(tl.sum(value * value, axis=0))
    denominator = tl.where(norm < eps, eps, norm)
    result = libdevice.div_rn(value, denominator)
    tl.store(output + row * width + cols, result, mask=cols < width)


def _validate_normalize(x, out_dtype, eps):
    if x.ndim != 2 or x.shape[1] == 0 or x.shape[1] > 8192 or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("L2 normalization requires a floating [rows, width<=8192] input with nonempty width")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("L2 normalization output must use BF16, FP16 or FP32")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("L2 normalization epsilon must be finite and positive")


@torch.library.custom_op("lightx2v::l2_normalize", mutates_args=(), device_types="cuda")
def l2_normalize(x: torch.Tensor, out_dtype: torch.dtype = torch.float32, eps: float = 1e-12) -> torch.Tensor:
    """Fuse the final FP32 conversion, L2 reduction and division.

    Accumulates in FP32 and writes directly in the requested output dtype.
    Its reduction tree can differ from torch.linalg.vector_norm.
    """
    _validate_normalize(x, out_dtype, eps)
    output = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    if x.shape[0]:
        block = triton.next_power_of_2(x.shape[1])
        _l2_normalize_kernel[(x.shape[0],)](x, output, x.shape[0], x.shape[1], *x.stride(), eps, block, num_warps=4 if block <= 2048 else 8, enable_fp_fusion=False)
    return output


@l2_normalize.register_fake
def _l2_normalize_fake(x, out_dtype=torch.float32, eps=1e-12):
    _validate_normalize(x, out_dtype, eps)
    return x.new_empty(x.shape, dtype=out_dtype)
