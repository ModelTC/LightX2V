"""MetaX training kernel for Flux2 real-valued RoPE."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _rotate_pair(
    IQ, IK, TQ, TK, Q, K, Cos, Sin, total_tokens, text_tokens,
    iq_token_stride, iq_head_stride, ik_token_stride, ik_head_stride,
    q_token_stride, q_head_stride, k_token_stride, k_head_stride,
    HEADS: tl.constexpr, WIDTH: tl.constexpr, BACKWARD: tl.constexpr, TILE: tl.constexpr,
):
    offset = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = offset // (HEADS * WIDTH)
    head = offset // WIDTH % HEADS
    dim = offset % WIDTH
    mate = dim ^ 1
    valid = token < total_tokens
    is_q = tl.program_id(1) == 0
    img_token = token - text_tokens
    img_q = IQ + img_token * iq_token_stride + head * iq_head_stride + dim
    img_k = IK + img_token * ik_token_stride + head * ik_head_stride + dim
    image = tl.where(is_q, img_q, img_k)
    text = tl.where(is_q, TQ, TK) + offset
    separate = tl.where(token < text_tokens, text, image)
    joint_q = Q + token * q_token_stride + head * q_head_stride + dim
    joint_k = K + token * k_token_stride + head * k_head_stride + dim
    joint = tl.where(is_q, joint_q, joint_k)
    cosine = tl.load(Cos + token * WIDTH + dim, valid, 0)
    if BACKWARD:
        a = tl.load(joint, valid, 0).to(tl.float32)
        b = tl.load(joint + mate - dim, valid, 0).to(tl.float32)
        sine = tl.load(Sin + token * WIDTH + mate, valid, 0)
        # The eager path rounds each gradient to BF16 before accumulation.
        direct = (a * cosine).to(tl.bfloat16).to(tl.float32)
        cross = (b * sine).to(tl.bfloat16).to(tl.float32)
        tl.store(separate, direct + tl.where(dim % 2 == 0, cross, -cross), valid)
    else:
        a = tl.load(separate, valid, 0).to(tl.float32)
        b = tl.load(separate + mate - dim, valid, 0).to(tl.float32)
        sine = tl.load(Sin + token * WIDTH + dim, valid, 0)
        direct = libdevice.mul_rn(a, cosine)
        cross = libdevice.mul_rn(tl.where(dim % 2 == 0, -b, b), sine)
        tl.store(joint, libdevice.add_rn(direct, cross), valid)


def _launch_rope(image_q, image_k, text_q, text_k, q, k, cos, sin, backward):
    heads, width = q.shape[-2:]
    total_tokens = q.size(1)
    text_tokens = 0 if text_q is None else text_q.size(1)
    _rotate_pair[(triton.cdiv(total_tokens * heads * width, 512), 2)](
        image_q, image_k, image_q if text_q is None else text_q, image_k if text_k is None else text_k,
        q, k, cos, sin, total_tokens, text_tokens,
        image_q.stride(1), image_q.stride(2), image_k.stride(1), image_k.stride(2),
        q.stride(1), q.stride(2), k.stride(1), k.stride(2),
        heads, width, backward, 512, enable_fp_fusion=False,
    )


class JointRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_q, image_k, text_q, text_k, cos, sin):
        q, k = image_q.new_empty((2, 1, cos.size(0), *image_q.shape[-2:])).unbind(0)
        _launch_rope(image_q, image_k, text_q, text_k, q, k, cos, sin, False)
        ctx.save_for_backward(cos, sin)
        ctx.text_tokens = 0 if text_q is None else text_q.size(1)
        return q, k

    @staticmethod
    def backward(ctx, q, k):
        cos, sin = ctx.saved_tensors
        q = q if q.stride(-1) == 1 else q.contiguous()
        k = k if k.stride(-1) == 1 else k.contiguous()
        text_tokens = ctx.text_tokens
        image_q, image_k = q.new_empty((2, 1, q.size(1) - text_tokens, *q.shape[-2:])).unbind(0)
        text_q, text_k = (None, None)
        if text_tokens:
            text_q, text_k = q.new_empty((2, 1, text_tokens, *q.shape[-2:])).unbind(0)
        _launch_rope(image_q, image_k, text_q, text_k, q, k, cos, sin, True)
        return image_q, image_k, text_q, text_k, None, None


def can_rotate(image_q, image_k, text_q, text_k, frequencies):
    if frequencies is None or len(frequencies) != 2 or not image_q.is_cuda:
        return False
    if image_q.shape != image_k.shape or (text_q is None) != (text_k is None):
        return False
    if text_q is not None and (text_q.shape != text_k.shape or text_q.size(1) == 0):
        return False
    tensors = (image_q, image_k) if text_q is None else (image_q, image_k, text_q, text_k)
    if not all(
        t.ndim == 4 and t.size(0) == 1 and t.shape[-2:] == image_q.shape[-2:]
        and t.dtype == torch.bfloat16 and t.stride(-1) == 1 and t.device == image_q.device
        for t in tensors
    ):
        return False
    if text_q is not None and not (text_q.is_contiguous() and text_k.is_contiguous()):
        return False
    tokens = image_q.size(1) + (0 if text_q is None else text_q.size(1))
    return tokens > 0 and image_q.size(-1) == 128 and all(
        freq.shape == (tokens, image_q.size(-1)) and freq.dtype == torch.float32
        and freq.device == image_q.device and freq.is_contiguous() and not freq.requires_grad
        for freq in frequencies
    )
