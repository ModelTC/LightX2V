"""Rotate text and image Q/K directly into the layout consumed by joint attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate_pairs(
    TQ,
    TK,
    IQ,
    IK,
    JQ,
    JK,
    TextFreq,
    ImageFreq,
    text_tokens,
    image_tokens,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    INVERSE: tl.constexpr,
    TILE: tl.constexpr,
):
    pair = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = pair // (HEADS * HALF)
    head = pair // HALF % HEADS
    dim = pair % HALF
    is_query = tl.program_id(1) == 0
    is_text = token < text_tokens
    valid = token < text_tokens + image_tokens

    text = tl.where(is_query, TQ, TK) + 2 * pair
    image = tl.where(is_query, IQ, IK) + 2 * (pair - text_tokens * HEADS * HALF)
    separate = tl.where(is_text, text, image)
    joint = tl.where(is_query, JQ + token * q_token_stride + head * q_head_stride, JK + token * k_token_stride + head * k_head_stride) + 2 * dim
    freq = tl.where(is_text, TextFreq + 2 * (token * HALF + dim), ImageFreq + 2 * ((token - text_tokens) * HALF + dim))
    cosine = tl.load(freq, mask=valid, other=0)
    sine = tl.load(freq + 1, mask=valid, other=0)
    if INVERSE:
        source, dest = joint, separate
        sine = -sine
    else:
        source, dest = separate, joint
    a = tl.load(source, mask=valid, other=0).to(tl.float32)
    b = tl.load(source + 1, mask=valid, other=0).to(tl.float32)
    # Match the FP32 complex multiply used by eager MetaX RoPE in both directions.
    tl.store(dest, a * cosine - b * sine, mask=valid)
    tl.store(dest + 1, tl.fma(b, cosine, a * sine), mask=valid)


def _rotate(text_q, text_k, image_q, image_k, joint_q, joint_k, text_freq, image_freq, inverse):
    text_tokens, image_tokens = text_freq.shape[0], image_freq.shape[0]
    heads, width = image_q.shape[-2:]
    grid = (triton.cdiv((text_tokens + image_tokens) * heads * (width // 2), 256), 2)
    _rotate_pairs[grid](
        text_q,
        text_k,
        image_q,
        image_k,
        joint_q,
        joint_k,
        torch.view_as_real(text_freq),
        torch.view_as_real(image_freq),
        text_tokens,
        image_tokens,
        joint_q.stride(1),
        joint_q.stride(2),
        joint_k.stride(1),
        joint_k.stride(2),
        heads,
        width // 2,
        inverse,
        256,
        enable_fp_fusion=False,
    )


class JointRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, text_q, text_k, image_q, image_k, text_freq, image_freq):
        text_freq, image_freq = text_freq.contiguous(), image_freq.contiguous()
        joint_q, joint_k = image_q.new_empty((2, 1, text_q.size(1) + image_q.size(1), *image_q.shape[-2:])).unbind()
        _rotate(text_q.contiguous(), text_k.contiguous(), image_q.contiguous(), image_k.contiguous(), joint_q, joint_k, text_freq, image_freq, False)
        ctx.save_for_backward(text_freq, image_freq)
        return joint_q, joint_k

    @staticmethod
    def backward(ctx, joint_q, joint_k):
        text_freq, image_freq = ctx.saved_tensors
        if joint_q.stride(-1) != 1:
            joint_q = joint_q.contiguous()
        if joint_k.stride(-1) != 1:
            joint_k = joint_k.contiguous()
        text_q, text_k = joint_q.new_empty((2, 1, text_freq.size(0), *joint_q.shape[-2:])).unbind()
        image_q, image_k = joint_q.new_empty((2, 1, image_freq.size(0), *joint_q.shape[-2:])).unbind()
        _rotate(text_q, text_k, image_q, image_k, joint_q, joint_k, text_freq, image_freq, True)
        return text_q, text_k, image_q, image_k, None, None


def supports_joint_rope(image_q, image_k, text_q, text_k, image_freq, text_freq):
    return (
        image_q.shape == image_k.shape
        and text_q.shape == text_k.shape
        and all(t.ndim == 4 and t.size(0) == 1 and t.shape[-2:] == (24, 128) and t.dtype == torch.bfloat16 and t.device == image_q.device for t in (image_q, image_k, text_q, text_k))
        and all(
            freq.shape == (length, 64) and freq.dtype == torch.complex64 and freq.device == image_q.device and not freq.requires_grad
            for freq, length in ((image_freq, image_q.size(1)), (text_freq, text_q.size(1)))
        )
    )
