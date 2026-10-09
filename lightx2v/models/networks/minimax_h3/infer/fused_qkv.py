"""Fused MiniMax-H3 QKV normalization and RoPE kernels."""

import math

import torch

from lightx2v.utils.registry_factory import QKV_NORM_ROPE_REGISTER

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _split_qkv_norm_rope_kernel(
        q_input,
        k_input,
        v_input,
        qw,
        kw,
        cos,
        sin,
        q,
        k,
        v,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        ROTARY: tl.constexpr,
        Q_STRIDE: tl.constexpr,
        K_STRIDE: tl.constexpr,
        V_STRIDE: tl.constexpr,
        COS_STRIDE: tl.constexpr,
        SIN_STRIDE: tl.constexpr,
        Q_EPS: tl.constexpr,
        K_EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        component = tl.program_id(1)
        token = row // HEADS
        head = row % HEADS
        d = tl.arange(0, BLOCK)
        if component == 0:
            input_ptr = q_input + token * Q_STRIDE + head * DIM
        elif component == 1:
            input_ptr = k_input + token * K_STRIDE + head * DIM
        else:
            input_ptr = v_input + token * V_STRIDE + head * DIM
        x = tl.load(input_ptr + d, d < DIM, other=0)
        if component == 2:
            tl.store(v + row * DIM + d, x, d < DIM)
        else:
            xf = x.to(tl.float32)
            if component == 0:
                weight = tl.load(qw + d, d < DIM, other=0).to(tl.float32)
                eps = Q_EPS
            else:
                weight = tl.load(kw + d, d < DIM, other=0).to(tl.float32)
                eps = K_EPS
            # Preserve the intermediate norm output cast without a memory roundtrip.
            y = (xf * tl.rsqrt(tl.sum(xf * xf, 0) / DIM + eps) * weight).to(x.dtype).to(tl.float32)
            pair = tl.where(d < ROTARY, tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2), d)
            paired = tl.gather(y, pair, axis=0)
            c = tl.load(cos + token * COS_STRIDE + d, d < ROTARY, other=0)
            s = tl.load(sin + token * SIN_STRIDE + d, d < ROTARY, other=0)
            a = y * c.to(tl.float32)
            b = paired * s.to(tl.float32)
            rotated = tl.where(d < ROTARY // 2, a - b, a + b)
            output = tl.where(d < ROTARY, rotated, y)
            if component == 0:
                tl.store(q + row * DIM + d, output, d < DIM)
            else:
                tl.store(k + row * DIM + d, output, d < DIM)


def _norms_are_compatible(q, k, v, norm_q, norm_k):
    """Check that the fused kernel preserves the configured RMSNorm semantics."""
    return (
        q.dtype in _SUPPORTED_DTYPES
        and all(t.device == q.device and t.dtype == q.dtype for t in (k, v))
        and all(
            getattr(norm, "weight", None) is not None
            and norm.weight.device == q.device
            and norm.weight.dtype == q.dtype
            and norm.weight.is_contiguous()
            and norm.sensitive_layer_dtype == norm.infer_dtype
            for norm in (norm_q, norm_k)
        )
    )


@torch.library.custom_op("lightx2v::minimax_h3_split_qkv_norm_rope", mutates_args=())
def split_qkv_norm_rope(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm and partial split-half RoPE with full-width cosine/sine caches."""
    if triton is None:
        raise RuntimeError("Fused QKV norm + RoPE requires Triton")
    dim = q_weight.numel()
    inputs = (q_input, k_input, v_input)
    if dim == 0 or any(t.ndim != 2 or t.shape != q_input.shape or t.shape[1] == 0 or t.shape[1] % dim or t.stride(1) != 1 for t in inputs):
        raise ValueError("Expected Q/K/V [tokens, heads * head_dim] tensors with contiguous channels")
    if q_weight.ndim != 1 or k_weight.shape != q_weight.shape or not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("Norm weights must be contiguous [head_dim] tensors")
    if q_input.dtype not in _SUPPORTED_DTYPES or any(t.dtype != q_input.dtype for t in (*inputs[1:], q_weight, k_weight)):
        raise ValueError("QKV and norm weights must have matching FP16/BF16/FP32 dtypes")
    if any(t.device != q_input.device for t in (*inputs[1:], q_weight, k_weight, cos, sin)):
        raise ValueError("All inputs must be on the same device")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != q_input.shape[0] or not 0 < cos.shape[1] <= dim or cos.shape[1] % 2:
        raise ValueError("Cos/sin must have shape [tokens, rotary_dim], with positive even rotary_dim <= head_dim")
    if cos.stride(1) != 1 or sin.stride(1) != 1 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Cos/sin must be FP32 with contiguous channels")
    if not all(math.isfinite(eps) and eps > 0 for eps in (q_eps, k_eps)):
        raise ValueError("Norm eps must be positive and finite")
    heads = q_input.shape[1] // dim
    outputs = tuple(q_input.new_empty((q_input.shape[0], heads, dim)) for _ in range(3))
    if q_input.shape[0]:
        with torch.get_device_module(q_input.device).device(q_input.device):
            _split_qkv_norm_rope_kernel[(q_input.shape[0] * heads, 3)](
                *inputs,
                q_weight,
                k_weight,
                cos,
                sin,
                *outputs,
                HEADS=heads,
                DIM=dim,
                ROTARY=cos.shape[1],
                Q_STRIDE=q_input.stride(0),
                K_STRIDE=k_input.stride(0),
                V_STRIDE=v_input.stride(0),
                COS_STRIDE=cos.stride(0),
                SIN_STRIDE=sin.stride(0),
                Q_EPS=q_eps,
                K_EPS=k_eps,
                BLOCK=triton.next_power_of_2(dim),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return outputs


@split_qkv_norm_rope.register_fake
def _split_qkv_norm_rope_fake(q, k, v, q_weight, k_weight, cos, sin, q_eps, k_eps):
    dim = q_weight.numel()
    shape = (q.shape[0], q.shape[1] // dim, dim)
    return tuple(q.new_empty(shape) for _ in range(3))


def prepare_qkv_norm_rope(q, k, v, norm_q, norm_k, rope, freqs):
    """Prepare fused inputs, or return None when semantics are incompatible."""
    if not _norms_are_compatible(q, k, v, norm_q, norm_k):
        return None
    if rope.layout != "split_half" or rope.compute_dtype != torch.float32:
        return None
    if not isinstance(freqs, tuple) or len(freqs) != 2:
        return None
    cos, sin = freqs
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != q.shape[0] or any(t.device != q.device or t.dtype != torch.float32 or t.stride(1) != 1 for t in freqs):
        return None
    return cos, sin


def run_triton_qkv_norm_rope(q, k, v, norm_q, norm_k, cos, sin):
    if triton is None:
        return None
    return split_qkv_norm_rope(
        q,
        k,
        v,
        norm_q.weight,
        norm_k.weight,
        cos,
        sin,
        norm_q.eps,
        norm_k.eps,
    )


@QKV_NORM_ROPE_REGISTER("triton")
class TritonQKVNormRope:
    @staticmethod
    def apply(q, k, v, norm_q, norm_k, rope, freqs):
        prepared = prepare_qkv_norm_rope(q, k, v, norm_q, norm_k, rope, freqs)
        if prepared is None:
            return None
        return run_triton_qkv_norm_rope(q, k, v, norm_q, norm_k, *prepared)


def try_split_qkv_norm_rope(q, k, v, norm_q, norm_k, rope, freqs, backend="triton"):
    """Apply a registered fused backend, returning None when it cannot run."""
    backend_cls = QKV_NORM_ROPE_REGISTER.get(backend)
    if backend_cls is None:
        return None
    return backend_cls().apply(q, k, v, norm_q, norm_k, rope, freqs)
