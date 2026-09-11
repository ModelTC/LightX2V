"""Split packed QKV and normalize Q/K without materializing their input copies."""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _split_qkv_norm_rope_kernel(
        packed,
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
        STRIDE: tl.constexpr,
        COS_STRIDE: tl.constexpr,
        SIN_STRIDE: tl.constexpr,
        Q_EPS: tl.constexpr,
        K_EPS: tl.constexpr,
        LOW_PRECISION_ROPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        component = tl.program_id(1)
        token = row // HEADS
        head = row % HEADS
        d = tl.arange(0, BLOCK)
        x = tl.load(packed + token * STRIDE + (component * HEADS + head) * DIM + d, d < DIM, other=0)
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
            if LOW_PRECISION_ROPE:
                c = c.to(x.dtype).to(tl.float32)
                s = s.to(x.dtype).to(tl.float32)
                a = (y * c).to(x.dtype).to(tl.float32)
                b = (paired * s).to(x.dtype).to(tl.float32)
            else:
                a = y * c.to(tl.float32)
                b = paired * s.to(tl.float32)
            rotated = tl.where(d < ROTARY // 2, a - b, a + b)
            output = tl.where(d < ROTARY, rotated, y)
            if component == 0:
                tl.store(q + row * DIM + d, output, d < DIM)
            else:
                tl.store(k + row * DIM + d, output, d < DIM)


def can_split_qkv_norm(packed, norm_q, norm_k):
    # Match RMSNorm's FP32 arithmetic and final output cast on any Triton backend.
    # Sensitive FP32 modes retain their own semantics.
    return packed.dtype in (torch.float16, torch.bfloat16, torch.float32) and all(
        getattr(norm, "weight", None) is not None
        and norm.weight.device == packed.device
        and norm.weight.dtype == packed.dtype
        and norm.weight.is_contiguous()
        and norm.sensitive_layer_dtype == norm.infer_dtype
        for norm in (norm_q, norm_k)
    )


@torch.library.custom_op("lightx2v::minimax_h3_split_qkv_norm_rope", mutates_args=())
def split_qkv_norm_rope(
    packed: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float, low_precision_rope: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMSNorm and partial split-half RoPE with full-width cosine/sine caches."""
    if triton is None or packed.device.type not in ("cuda", "xpu"):
        raise RuntimeError("Fused QKV norm + RoPE requires Triton on CUDA or XPU")
    dim = q_weight.numel()
    if packed.ndim != 2 or dim == 0 or packed.shape[1] == 0 or packed.shape[1] % (3 * dim) or packed.stride(1) != 1:
        raise ValueError("Expected packed [tokens, 3 * heads * head_dim] with contiguous channels")
    if q_weight.ndim != 1 or k_weight.shape != q_weight.shape or not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("Norm weights must be contiguous [head_dim] tensors")
    if packed.dtype not in (torch.float16, torch.bfloat16, torch.float32) or any(w.dtype != packed.dtype for w in (q_weight, k_weight)):
        raise ValueError("QKV and norm weights must have matching FP16/BF16/FP32 dtypes")
    if any(t.device != packed.device for t in (q_weight, k_weight, cos, sin)):
        raise ValueError("All inputs must be on the same device")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != packed.shape[0] or not 0 < cos.shape[1] <= dim or cos.shape[1] % 2:
        raise ValueError("Cos/sin must have shape [tokens, rotary_dim], with positive even rotary_dim <= head_dim")
    if cos.stride(1) != 1 or sin.stride(1) != 1 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError("Cos/sin must be FP32 with contiguous channels")
    if not all(math.isfinite(eps) and eps > 0 for eps in (q_eps, k_eps)):
        raise ValueError("Norm eps must be positive and finite")
    heads = packed.shape[1] // (3 * dim)
    outputs = tuple(packed.new_empty((packed.shape[0], heads, dim)) for _ in range(3))
    if packed.shape[0]:
        with getattr(torch, packed.device.type).device(packed.device):
            _split_qkv_norm_rope_kernel[(packed.shape[0] * heads, 3)](
                packed,
                q_weight,
                k_weight,
                cos,
                sin,
                *outputs,
                HEADS=heads,
                DIM=dim,
                ROTARY=cos.shape[1],
                STRIDE=packed.stride(0),
                COS_STRIDE=cos.stride(0),
                SIN_STRIDE=sin.stride(0),
                Q_EPS=q_eps,
                K_EPS=k_eps,
                LOW_PRECISION_ROPE=low_precision_rope,
                BLOCK=triton.next_power_of_2(dim),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return outputs


@split_qkv_norm_rope.register_fake
def _split_qkv_norm_rope_fake(packed, q_weight, k_weight, cos, sin, q_eps, k_eps, low_precision_rope=False):
    dim = q_weight.numel()
    shape = (packed.shape[0], packed.shape[1] // (3 * dim), dim)
    return tuple(packed.new_empty(shape) for _ in range(3))


def try_split_qkv_norm_rope(packed, norm_q, norm_k, rope, freqs, backend="triton"):
    """Return None when the configured norm/RoPE semantics require the original path."""
    if not can_split_qkv_norm(packed, norm_q, norm_k) or packed.device.type not in ("cuda", "xpu"):
        return None
    if rope.layout != "split_half" or rope.compute_dtype != torch.float32:
        return None
    if not isinstance(freqs, tuple) or len(freqs) != 2:
        return None
    cos, sin = freqs
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[0] != packed.shape[0] or any(t.device != packed.device or t.dtype != torch.float32 or t.stride(1) != 1 for t in freqs):
        return None
    rope_class = (type(rope).__module__, type(rope).__name__)
    low_precision = False
    if rope_class == ("lightx2v_platform.ops.rope.intel_xpu.minimax_h3_rope", "MiniMaxH3XpuRope"):
        view = packed[:, : packed.shape[1] // 3].unflatten(-1, (-1, norm_q.weight.numel()))
        low_precision = rope._can_use_xpu_kernel(view, cos, sin, cos.shape[-1])
    elif rope_class != ("lightx2v.common.ops.rope.torch_rope", "TorchRealRope"):
        return None
    if backend == "intel_xpu" and packed.device.type == "xpu" and norm_q.weight.numel() == 128 and cos.shape[1] == 96:
        try:
            import sycl_kernels
        except ImportError:
            sycl_kernels = None
        if sycl_kernels is not None and getattr(sycl_kernels, "has_minimax_h3_qkv_norm_rope", lambda: False)():
            return sycl_kernels.minimax_h3_qkv_norm_rope(packed, norm_q.weight, norm_k.weight, cos, sin, norm_q.eps, norm_k.eps, low_precision)
    if triton is None:
        return None
    return split_qkv_norm_rope(packed, norm_q.weight, norm_k.weight, cos, sin, norm_q.eps, norm_k.eps, low_precision)
