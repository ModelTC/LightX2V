"""Exact FP32 Wan forward AD, sharing the attributed Turbo attention kernel."""

from contextlib import contextmanager

import torch
from torch.autograd import forward_ad


def fp32_attention_jvp(query, key, value, query_tangent, key_tangent, value_tangent, scale=None):
    """FP32 CUDA attention/JVP without half casts or TF32 arithmetic.

    Uses the same attributed Apache-2.0 upstream streaming JVP algorithm, with
    explicit IEEE FP32 matmuls. H3's half-only implementation stays unchanged.
    """
    import triton

    from .wan_turbo_flash_attention_jvp import _attn_fwd

    tensors = [tensor.detach().transpose(1, 2).contiguous() for tensor in (query, key, value, query_tangent, key_tangent, value_tangent)]
    if any(tensor.dtype != torch.float32 or tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("Wan Turbo CUDA attention/JVP requires all six tensors in FP32.")
    q, k, v, tq, tk, tv = tensors
    if q.shape[0:2] != k.shape[0:2] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Wan Turbo QKV batch/head/channel dimensions must match.")
    if any(a.shape != b.shape for a, b in ((q, tq), (k, tk), (v, tv))):
        raise ValueError("Wan Turbo QKV tangents must match primal shapes.")
    batch, heads, query_length, channels = q.shape
    key_length = k.shape[2]
    if channels not in {16, 32, 64, 128, 256}:
        raise ValueError("Wan Turbo CUDA JVP requires head dimension 16,32,64,128,256.")
    output, tangent = torch.empty_like(q), torch.empty_like(q)
    lse = torch.empty((batch, heads, query_length), device=q.device, dtype=torch.float32)
    # Small FP32 tiles avoid the BF16 kernel's large shared-memory autotuning
    # choices. IEEE dot is explicit: no hidden BF16/FP16 or TF32 approximation.
    grid = (triton.cdiv(query_length, 16), batch * heads, 1)
    _attn_fwd[grid](
        q,
        k,
        v,
        tq,
        tk,
        tv,
        channels**-0.5 if scale is None else scale,
        lse,
        output,
        tangent,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        batch,
        heads,
        query_length,
        key_length,
        channels,
        channels,
        BLOCK_M=16,
        BLOCK_N=16,
        num_warps=4,
        num_stages=1,
        default_dot_input_precision="ieee",
    )
    return output.transpose(1, 2).contiguous(), tangent.transpose(1, 2).contiguous()


def exact_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), **kwargs):
    del kwargs
    if dropout_p or causal or tuple(window_size) != (-1, -1):
        raise ValueError("Wan Turbo JVP supports non-causal global attention without dropout.")
    for lengths, expected in ((q_lens, q.shape[1]), (k_lens, k.shape[1])):
        if lengths is not None and not bool((torch.as_tensor(lengths) == expected).all()):
            raise ValueError("Wan Turbo JVP requires unpadded singleton attention sequences.")
    if q_scale is not None:
        q = q * q_scale
    if q.device.type == "cpu":
        query, key, value = (tensor.transpose(1, 2) for tensor in (q, k, v))
        scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        return ((query @ key.transpose(-1, -2) * scale).softmax(-1) @ value).transpose(1, 2).contiguous()
    primals, tangents = [], []
    for tensor in (q, k, v):
        primal, tangent = forward_ad.unpack_dual(tensor)
        primals.append(primal)
        tangents.append(torch.zeros_like(primal) if tangent is None else tangent)
    output, tangent = fp32_attention_jvp(*primals, *tangents, scale=softmax_scale)
    return forward_ad.make_dual(output, tangent)


@contextmanager
def exact_wan_jvp_attention():
    # Wan's native self/cross modules resolve this shared function at call
    # time. Restore it even on failure; ordinary student backward stays on the
    # requested FP32 SDPA backend and the FSDP root call is never bypassed.
    from lightx2v_train.model_zoo.native.wan.modules import model as wan_model

    previous = wan_model.flash_attention
    try:
        wan_model.flash_attention = exact_attention
        yield
    finally:
        wan_model.flash_attention = previous
