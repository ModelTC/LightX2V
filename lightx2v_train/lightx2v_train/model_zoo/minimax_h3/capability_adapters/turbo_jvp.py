"""Exact input/time JVP support for H3 consistency training.

The JVP is a detached target, evaluated separately from the ordinary student
forward used for parameter gradients. Forward AD keeps FSDP's normal module
calls and does not differentiate or functionalize its sharded parameters.
"""

from contextlib import contextmanager

import torch
from torch.autograd import forward_ad

from lightx2v_train.model_zoo.native.minimax_h3.packing import KEYFRAME_NOISE_AUG


def differentiable_row_timesteps(layout, video_sigma, audio_sigma):
    """Build the same row times without scalar extraction or ``unique``.

    A fixed table keeps equal-valued video/audio times independently
    differentiable. Text rows use the target-video time, just as the released
    H3 packing does; reference visual rows use max(video time, augmentation).
    """
    video_time = 1.0 - video_sigma.reshape(())
    audio_time = 1.0 - audio_sigma.reshape(())
    condition_video_time = torch.maximum(video_time, video_time.new_tensor(KEYFRAME_NOISE_AUG))
    times = torch.stack((video_time, audio_time, condition_video_time, video_time.new_ones(())))
    indices = torch.zeros(layout.sequence_length, dtype=torch.long, device=video_time.device)
    indices[layout.audio_indices[layout.num_condition_audio_rows :]] = 1
    if layout.num_condition_video_rows:
        indices[layout.video_indices[: layout.num_condition_video_rows]] = 2
    if layout.num_condition_audio_rows:
        indices[layout.audio_indices[: layout.num_condition_audio_rows]] = 3
    return times, indices


def _math_attention(query, key, value, attention_mask):
    """Small CPU reference, never selected for production CUDA sequences."""
    query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    weights = query @ key.transpose(-1, -2) * (query.shape[-1] ** -0.5)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            weights = weights.masked_fill(~attention_mask, float("-inf"))
        else:
            weights = weights + attention_mask
    return (weights.softmax(dim=-1) @ value).transpose(1, 2)


def _attention_jvp(query, key, value, attention_mask):
    if query.device.type == "cpu":
        return _math_attention(query, key, value, attention_mask)
    if query.device.type != "cuda":
        raise ValueError("H3 consistency JVP supports CPU reference or CUDA Triton attention.")
    if attention_mask is not None:
        raise ValueError("H3 CUDA JVP requires an unmasked packed sequence.")
    from .turbo_flash_attention_jvp import attention_with_jvp

    primals, tangents = [], []
    for tensor in (query, key, value):
        primal, tangent = forward_ad.unpack_dual(tensor)
        primals.append(primal)
        tangents.append(torch.zeros_like(primal) if tangent is None else tangent)
    primal, tangent = attention_with_jvp(*primals, *tangents)
    return forward_ad.make_dual(primal, tangent)


class MiniMaxH3JVPAttnProcessor:
    def __init__(self, previous):
        self.previous = previous

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if forward_ad.unpack_dual(hidden_states).tangent is None:
            return self.previous(attn, hidden_states, rotary_emb, attention_mask)
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = (projection(hidden_states) for projection in (attn.to_q, attn.to_k, attn.to_v))
        query, key, value = (tensor.unflatten(-1, (attn.heads, -1)) for tensor in (query, key, value))
        query, key = attn.norm_q(query), attn.norm_k(key)
        if rotary_emb is not None:
            query, key = (_apply_rotary_emb(tensor, *rotary_emb) for tensor in (query, key))
        hidden_states = _attention_jvp(query, key, value, attention_mask).flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


@contextmanager
def exact_jvp_attention(transformer):
    """Temporarily install JVP attention; restore processors even on failure."""
    blocks = [*transformer.token_refiner.refiner_blocks, *transformer.transformer_blocks]
    previous = [(block.attn, block.attn.processor) for block in blocks]
    try:
        for attention, processor in previous:
            attention.set_processor(MiniMaxH3JVPAttnProcessor(processor))
        yield
    finally:
        for attention, processor in previous:
            attention.set_processor(processor)
