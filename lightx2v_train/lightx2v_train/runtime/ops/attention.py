"""Attention in BSND layout, with explicit padding and visibility semantics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _lengths(lengths, batch: int, maximum: int) -> list[int]:
    if lengths is None:
        return [maximum] * batch
    values = lengths.detach().cpu().tolist() if isinstance(lengths, torch.Tensor) else list(lengths)
    values = [int(value) for value in values]
    if len(values) != batch or any(value < 0 or value > maximum for value in values):
        raise ValueError(f"Attention lengths must contain {batch} values between 0 and {maximum}.")
    return values


def _visibility_mask(query, key, causal: bool, window_size: tuple[int, int]):
    """Return allowed positions using FlashAttention's bottom-right alignment."""
    left, right = window_size
    if not causal and left == -1 and right == -1:
        return None
    rows = torch.arange(query.size(1), device=query.device).unsqueeze(1)
    cols = torch.arange(key.size(1), device=query.device).unsqueeze(0)
    center = rows + key.size(1) - query.size(1)
    allowed = torch.ones((query.size(1), key.size(1)), device=query.device, dtype=torch.bool)
    if causal:
        allowed = allowed & (cols <= center)
    if left >= 0:
        allowed = allowed & (cols >= center - left)
    if right >= 0:
        allowed = allowed & (cols <= center + right)
    return allowed.unsqueeze(0).unsqueeze(0)


def training_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens=None,
    k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dtype: torch.dtype = torch.bfloat16,
):
    """Execute differentiable attention, retaining padded output shape.

    Samples are sliced before execution so padded keys never participate in
    softmax and padded queries produce zeros. The same contract covers GQA,
    rectangular causal attention, local windows, and explicit QK scaling.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Attention expects q, k, and v in [batch, sequence, heads, channels] layout.")
    if q.size(0) != k.size(0) or k.shape[:3] != v.shape[:3] or q.size(-1) != k.size(-1):
        raise ValueError("Incompatible attention batch, sequence, or channel dimensions.")
    if q.size(2) % k.size(2):
        raise ValueError("The number of query heads must be divisible by the number of key/value heads.")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Attention q, k, and v must use the same device.")
    if any(bound < -1 for bound in window_size) or len(window_size) != 2:
        raise ValueError("Attention window_size must contain two integers >= -1.")
    output_dtype = q.dtype
    query_lengths = _lengths(q_lens, q.size(0), q.size(1))
    key_lengths = _lengths(k_lens, k.size(0), k.size(1))
    outputs = []
    for index, (query_length, key_length) in enumerate(zip(query_lengths, key_lengths)):
        query = q[index : index + 1, :query_length]
        key = k[index : index + 1, :key_length]
        value = v[index : index + 1, :key_length]
        if query_length == 0 or key_length == 0:
            # Keep zero gradients connected to all three inputs.
            output = query[..., :1] * 0 + (key.sum() + value.sum()) * 0
            output = output.expand(1, query_length, q.size(2), v.size(-1))
        else:
            if q_scale is not None:
                query = query * q_scale
            repeats = q.size(2) // k.size(2)
            if repeats != 1:
                key = key.repeat_interleave(repeats, dim=2)
                value = value.repeat_interleave(repeats, dim=2)
            mask = _visibility_mask(query, key, causal, window_size)
            if query.device.type == "npu":
                # Vendor extension remains lazy and confined to the operator.
                from torch_npu import npu_fusion_attention

                compute_dtype = value.dtype if value.dtype in (torch.float16, torch.bfloat16) else dtype
                query, key, value = (tensor.to(compute_dtype).contiguous() for tensor in (query, key, value))
                output = npu_fusion_attention(
                    query,
                    key,
                    value,
                    query.size(2),
                    input_layout="BSND",
                    atten_mask=None if mask is None else (~mask).contiguous(),
                    scale=softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(query.size(-1)),
                    pre_tockens=2147483647,
                    next_tockens=2147483647,
                    keep_prob=1.0 - dropout_p,
                )[0]
                if mask is not None:
                    # Fully masked rows occur in rectangular causal attention.
                    output = output.masked_fill(~mask.any(dim=-1).transpose(1, 2).unsqueeze(-1), 0)
            else:
                query, key = query.to(value.dtype), key.to(value.dtype)
                output = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    attn_mask=mask,
                    dropout_p=dropout_p,
                    scale=softmax_scale,
                ).transpose(1, 2)
            output = output.to(output_dtype)
        if query_length < q.size(1):
            output = torch.cat([output, output.new_zeros(1, q.size(1) - query_length, q.size(2), v.size(-1))], dim=1)
        outputs.append(output)
    return torch.cat(outputs, dim=0)
