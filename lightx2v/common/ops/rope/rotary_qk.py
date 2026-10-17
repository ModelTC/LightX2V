"""Fused half-split rotary embeddings for batched Q/K tensors.

The CUDA implementation accepts [batch, sequence, heads, width] tensors and
preserves unrotated prefix tokens. Attention backends remain independent.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _separate_fp32_products(a, b, c, d):
    # Keep each FP32 operation scalar and explicitly rounded. On SM120 the
    # compiler can otherwise vectorize the expression into f32x2 operations
    # whose lowered arithmetic changes rare final BF16 tie cases.
    first = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b], dtype=tl.float32, is_pure=True, pack=1)
    second = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[c, d], dtype=tl.float32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[first, second], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rotary_qk_kernel(
    q,
    k,
    cos,
    sin,
    q_out,
    k_out,
    total: tl.constexpr,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    width: tl.constexpr,
    prefix: tl.constexpr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    qs2: tl.constexpr,
    qs3: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    ks3: tl.constexpr,
    cs0: tl.constexpr,
    cs1: tl.constexpr,
    cs2: tl.constexpr,
    ss0: tl.constexpr,
    ss1: tl.constexpr,
    ss2: tl.constexpr,
    cos_broadcast: tl.constexpr,
    sin_broadcast: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    col = offsets % width
    head = offsets // width % heads
    token = offsets // (width * heads) % sequence
    batch = offsets // (width * heads * sequence)
    is_patch = token >= prefix
    half = width // 2
    paired_col = (col + half) % width
    q_row = batch * qs0 + token * qs1 + head * qs2
    k_row = batch * ks0 + token * ks1 + head * ks2
    raw_q = tl.load(q + q_row + col * qs3, mask=active, other=0)
    raw_k = tl.load(k + k_row + col * ks3, mask=active, other=0)
    paired_q = tl.load(q + q_row + paired_col * qs3, mask=active & is_patch, other=0).to(tl.float32)
    paired_k = tl.load(k + k_row + paired_col * ks3, mask=active & is_patch, other=0).to(tl.float32)
    cosine = tl.load(cos + (token - prefix) * cs0 + (0 if cos_broadcast else head) * cs1 + col * cs2, mask=active & is_patch, other=0)
    sine = tl.load(sin + (token - prefix) * ss0 + (0 if sin_broadcast else head) * ss1 + col * ss2, mask=active & is_patch, other=0)
    rotated_q = tl.where(col < half, -paired_q, paired_q)
    rotated_k = tl.where(col < half, -paired_k, paired_k)
    q_value = _separate_fp32_products(raw_q.to(tl.float32), cosine, rotated_q, sine).to(raw_q.dtype)
    k_value = _separate_fp32_products(raw_k.to(tl.float32), cosine, rotated_k, sine).to(raw_k.dtype)
    # Selecting in the source dtype preserves even NaN payloads in prefix tokens.
    tl.store(q_out + offsets, tl.where(is_patch, q_value, raw_q), mask=active)
    tl.store(k_out + offsets, tl.where(is_patch, k_value, raw_k), mask=active)


def _validate_rope(q, k, cos, sin, prefix):
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("Half-split RoPE expects matching [batch, sequence, heads, width] Q/K")
    if q.dtype != k.dtype or q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Half-split RoPE requires matching floating Q/K dtypes")
    batch, sequence, heads, width = q.shape
    if not 0 <= prefix <= sequence or width == 0 or width % 2 or heads == 0:
        raise ValueError("Half-split RoPE requires an even width and a valid prefix")
    for angle in (cos, sin):
        if angle.ndim != 3 or angle.shape[0] != sequence - prefix or angle.shape[1] not in (1, heads) or angle.shape[2] != width:
            raise ValueError("Half-split RoPE angles must have shape [sequence - prefix, 1 or heads, width]")
        if angle.dtype != torch.float32:
            raise ValueError("Half-split RoPE angles must use FP32")
    if any(tensor.device != q.device for tensor in (k, cos, sin)):
        raise ValueError("Half-split RoPE tensors must share a device")


@torch.library.custom_op("lightx2v::apply_rotary_qk", mutates_args=(), device_types="cuda")
def apply_rotary_qk(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, prefix: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q/K in one launch, preserving the original FP32/BF16 boundaries.

    Inputs may be strided views, including slices of a packed QKV projection.
    Outputs are new contiguous tensors, compatible with tracing and CUDA graphs.
    Prefix tokens are copied bit for bit. No attention approximation is used.
    """
    _validate_rope(q, k, cos, sin, prefix)
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    if q.numel():
        _rotary_qk_kernel[(triton.cdiv(q.numel(), 1024),)](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            q.numel(),
            q.shape[1],
            q.shape[2],
            q.shape[3],
            prefix,
            *q.stride(),
            *k.stride(),
            *cos.stride(),
            *sin.stride(),
            cos.shape[1] == 1,
            sin.shape[1] == 1,
            1024,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return q_out, k_out


@apply_rotary_qk.register_fake
def _rotary_qk_fake(q, k, cos, sin, prefix=0):
    _validate_rope(q, k, cos, sin, prefix)
    return q.new_empty(q.shape), k.new_empty(k.shape)
