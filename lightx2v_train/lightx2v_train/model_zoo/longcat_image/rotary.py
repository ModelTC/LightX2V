"""Real-valued Q/K RoPE with optional text/image concatenation for training."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rotate(
    IQ,
    IK,
    TQ,
    TK,
    Q,
    K,
    Cos,
    Sin,
    text_tokens,
    total_tokens,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BACKWARD: tl.constexpr,
    TILE: tl.constexpr,
):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = offset // (HEADS * WIDTH)
    head = offset // WIDTH % HEADS
    dim = offset % WIDTH
    mate = dim ^ 1
    valid = token < total_tokens
    query = tl.program_id(1) == 0
    text = token < text_tokens
    separate = tl.where(text, tl.where(query, TQ, TK) + offset, tl.where(query, IQ, IK) + offset - text_tokens * HEADS * WIDTH)
    joint = tl.where(query, Q + token * q_token_stride + head * q_head_stride, K + token * k_token_stride + head * k_head_stride) + dim
    cosine = tl.load(Cos + token * WIDTH + dim, valid, 0)
    if BACKWARD:
        a = tl.load(joint, valid, 0).to(tl.float32)
        b = tl.load(joint + mate - dim, valid, 0).to(tl.float32)
        sine = tl.load(Sin + token * WIDTH + mate, valid, 0)
        # Eager adds the two paths after each has returned to BF16.
        direct = (a * cosine).to(tl.bfloat16).to(tl.float32)
        crossed = (b * sine).to(tl.bfloat16).to(tl.float32)
        tl.store(separate, direct + tl.where(dim % 2 == 0, crossed, -crossed), valid)
    else:
        a = tl.load(separate, valid, 0).to(tl.float32)
        b = tl.load(separate + mate - dim, valid, 0).to(tl.float32)
        sine = tl.load(Sin + token * WIDTH + dim, valid, 0)
        direct = libdevice.mul_rn(a, cosine)
        crossed = libdevice.mul_rn(tl.where(dim % 2 == 0, -b, b), sine)
        tl.store(joint, libdevice.add_rn(direct, crossed), valid)


def _launch(image_q, image_k, text_q, text_k, q, k, cos, sin, backward):
    heads, width = q.shape[-2:]
    total_tokens = q.size(1)
    text_tokens = 0 if text_q is None else text_q.size(1)
    grid = (triton.cdiv(total_tokens * heads * width, 512), 2)
    _rotate[grid](
        image_q,
        image_k,
        image_q if text_q is None else text_q,
        image_k if text_k is None else text_k,
        q,
        k,
        cos,
        sin,
        text_tokens,
        total_tokens,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        heads,
        width,
        backward,
        512,
        enable_fp_fusion=False,
    )


class LongCatRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_q, image_k, text_q, text_k, cos, sin):
        q, k = image_q.new_empty((2, 1, cos.size(0), *image_q.shape[-2:])).unbind(0)
        _launch(image_q, image_k, text_q, text_k, q, k, cos, sin, False)
        ctx.save_for_backward(cos, sin)
        ctx.text_tokens = None if text_q is None else text_q.size(1)
        return q, k

    @staticmethod
    def backward(ctx, q, k):
        cos, sin = ctx.saved_tensors
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        text_count = ctx.text_tokens
        image_q, image_k = q.new_empty((2, 1, q.size(1) - (text_count or 0), *q.shape[-2:])).unbind(0)
        text_q, text_k = (None, None) if text_count is None else q.new_empty((2, 1, text_count, *q.shape[-2:])).unbind(0)
        _launch(image_q, image_k, text_q, text_k, q, k, cos, sin, True)
        return image_q, image_k, text_q, text_k, None, None


def supports_rotary(image_q, image_k, text_q, text_k, frequencies):
    if frequencies is None or len(frequencies) != 2 or not image_q.is_cuda:
        return False
    if image_q.shape != image_k.shape or (text_q is None) != (text_k is None) or (text_q is not None and text_q.shape != text_k.shape):
        return False
    tensors = (image_q, image_k) if text_q is None else (image_q, image_k, text_q, text_k)
    if not all(t.ndim == 4 and t.size(0) == 1 and t.shape[-2:] == (24, 128) and t.dtype == torch.bfloat16 and t.is_contiguous() and t.device == image_q.device for t in tensors):
        return False
    tokens = image_q.size(1) + (0 if text_q is None else text_q.size(1))
    return tokens > 0 and all(freq.shape == (tokens, 128) and freq.dtype == torch.float32 and freq.device == image_q.device and freq.is_contiguous() and not freq.requires_grad for freq in frequencies)
