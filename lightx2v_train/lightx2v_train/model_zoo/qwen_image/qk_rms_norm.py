"""Paired frozen-weight RMSNorm for the image Q/K heads on MetaX."""

import torch
import triton
import triton.language as tl
from diffusers.models.normalization import RMSNorm


@triton.jit
def _reduction_input(Q, K, DQ, DK, WQ, WK, Work, n, WIDTH: tl.constexpr, BACKWARD: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    stream = tl.program_id(1)
    valid = offsets < n
    x = tl.load(tl.where(stream == 0, Q, K) + offsets, mask=valid, other=0).to(tl.float32)
    if BACKWARD:
        grad = tl.load(tl.where(stream == 0, DQ, DK) + offsets, mask=valid, other=0).to(tl.float32)
        weight = tl.load(tl.where(stream == 0, WQ, WK) + offsets % WIDTH).to(tl.float32)
        normalized_grad = (grad * weight).to(tl.bfloat16).to(tl.float32)
        value = normalized_grad * x
    else:
        value = x * x
    tl.store(Work + stream * n + offsets, value, mask=valid)


@triton.jit
def _apply(Q, K, DQ, DK, WQ, WK, Inv, Coeff, Out, n, WIDTH: tl.constexpr, BACKWARD: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    stream = tl.program_id(1)
    valid = offsets < n
    row = (stream * n + offsets) // WIDTH
    x = tl.load(tl.where(stream == 0, Q, K) + offsets, mask=valid, other=0).to(tl.float32)
    weight = tl.load(tl.where(stream == 0, WQ, WK) + offsets % WIDTH).to(tl.float32)
    inv = tl.load(Inv + row, mask=valid, other=0)
    if BACKWARD:
        grad = tl.load(tl.where(stream == 0, DQ, DK) + offsets, mask=valid, other=0).to(tl.float32)
        g = (grad * weight).to(tl.bfloat16).to(tl.float32)
        coefficient = tl.load(Coeff + row, mask=valid, other=0)
        # Each of the two eager paths reaches the BF16 input before accumulation.
        direct = (g * inv).to(tl.bfloat16).to(tl.float32)
        through_variance = (coefficient * (2.0 * x)).to(tl.bfloat16).to(tl.float32)
        value = direct + through_variance
    else:
        normalized = (x * inv).to(tl.bfloat16).to(tl.float32)
        value = normalized * weight
    tl.store(Out + stream * n + offsets, value, mask=valid)


class _PairedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, wq, wk, eps):
        shape = (2, *q.shape)
        n, width = q.numel(), q.shape[-1]
        grid = (triton.cdiv(n, 512), 2)
        squares = torch.empty(shape, device=q.device, dtype=torch.float32)
        _reduction_input[grid](q, k, q, k, wq, wk, squares, n, width, False, 512, enable_fp_fusion=False)
        inv = (squares.mean(dim=-1, keepdim=True) + eps).rsqrt()
        out = torch.empty(shape, device=q.device, dtype=q.dtype)
        _apply[grid](q, k, q, k, wq, wk, inv, inv, out, n, width, False, 512, enable_fp_fusion=False)
        ctx.save_for_backward(q, k, wq, wk, inv)
        return out[0], out[1]

    @staticmethod
    def backward(ctx, dq, dk):
        q, k, wq, wk, inv = ctx.saved_tensors
        dq, dk = dq.contiguous(), dk.contiguous()
        shape = (2, *q.shape)
        n, width = q.numel(), q.shape[-1]
        grid = (triton.cdiv(n, 512), 2)
        products = torch.empty(shape, device=q.device, dtype=torch.float32)
        _reduction_input[grid](q, k, dq, dk, wq, wk, products, n, width, True, 512, enable_fp_fusion=False)
        grad_inv = products.sum(dim=-1, keepdim=True)
        coefficient = (-0.5 * grad_inv) * inv.pow(3) / width
        out = torch.empty(shape, device=q.device, dtype=q.dtype)
        _apply[grid](q, k, dq, dk, wq, wk, inv, coefficient, out, n, width, True, 512, enable_fp_fusion=False)
        return out[0], out[1], None, None, None


def _frozen_head_norm(norm, width):
    return (
        type(norm) is RMSNorm
        and norm.bias is None
        and norm.weight is not None
        and norm.weight.shape == (width,)
        and not norm.weight.requires_grad
        and not (norm._forward_hooks or norm._forward_pre_hooks or norm._backward_hooks or norm._backward_pre_hooks)
    )


def normalize_image_qk(query, key, norm_q, norm_k, *, fused):
    width = query.shape[-1]
    if (
        fused
        and width == 128
        and query.shape == key.shape
        and query.numel() > 0
        and _frozen_head_norm(norm_q, width)
        and _frozen_head_norm(norm_k, width)
        and norm_q.eps == norm_k.eps
        and all(t.dtype == torch.bfloat16 and t.is_contiguous() and t.device == query.device for t in (query, key, norm_q.weight, norm_k.weight))
    ):
        return _PairedRMSNorm.apply(query, key, norm_q.weight, norm_k.weight, norm_q.eps)
    query = norm_q(query) if norm_q is not None else query
    key = norm_k(key) if norm_k is not None else key
    return query, key
