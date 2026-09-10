"""Split packed QKV and normalize Q/K without materializing their input copies."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _split_qkv_norm_kernel(
        packed, q_weight, k_weight, q_out, k_out, v_out,
        HEADS: tl.constexpr, DIM: tl.constexpr, ROW_STRIDE: tl.constexpr,
        Q_EPS: tl.constexpr, K_EPS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        component = tl.program_id(1)
        token = row // HEADS
        head = row % HEADS
        offsets = tl.arange(0, BLOCK)
        x = tl.load(packed + token * ROW_STRIDE + (component * HEADS + head) * DIM + offsets, offsets < DIM, other=0).to(tl.float32)
        if component == 0:
            weight = tl.load(q_weight + offsets, offsets < DIM, other=0).to(tl.float32)
            y = x * tl.rsqrt(tl.sum(x * x, 0) / DIM + Q_EPS) * weight
            tl.store(q_out + row * DIM + offsets, y, offsets < DIM)
        elif component == 1:
            weight = tl.load(k_weight + offsets, offsets < DIM, other=0).to(tl.float32)
            y = x * tl.rsqrt(tl.sum(x * x, 0) / DIM + K_EPS) * weight
            tl.store(k_out + row * DIM + offsets, y, offsets < DIM)
        else:
            tl.store(v_out + row * DIM + offsets, x, offsets < DIM)


@torch.library.custom_op("lightx2v::minimax_h3_split_qkv_norm", mutates_args=())
def split_qkv_norm(packed: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, q_eps: float, k_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Fused QKV normalization requires Triton")
    dim = q_weight.numel()
    if packed.ndim != 2 or dim == 0 or packed.shape[1] % (3 * dim) or packed.stride(1) != 1:
        raise ValueError("Expected packed [tokens, 3 * heads * head_dim] QKV with contiguous channels")
    if q_weight.ndim != 1 or k_weight.shape != q_weight.shape or not q_weight.is_contiguous() or not k_weight.is_contiguous():
        raise ValueError("Q/K RMSNorm weights must be contiguous [head_dim] tensors")
    if packed.device.type not in ("cuda", "xpu") or q_weight.device != packed.device or k_weight.device != packed.device:
        raise ValueError("QKV and norm weights must be on the same CUDA/XPU device")
    heads = packed.shape[1] // (3 * dim)
    shape = (packed.shape[0], heads, dim)
    q, k, v = (torch.empty(shape, device=packed.device, dtype=packed.dtype) for _ in range(3))
    if packed.shape[0]:
        with getattr(torch, packed.device.type).device(packed.device):
            _split_qkv_norm_kernel[(packed.shape[0] * heads, 3)](
                packed, q_weight, k_weight, q, k, v,
                HEADS=heads, DIM=dim, ROW_STRIDE=packed.stride(0),
                Q_EPS=q_eps, K_EPS=k_eps, BLOCK=triton.next_power_of_2(dim),
                num_warps=4,
            )
    return q, k, v


@split_qkv_norm.register_fake
def _split_qkv_norm_fake(packed, q_weight, k_weight, q_eps, k_eps):
    dim = q_weight.numel()
    shape = (packed.shape[0], packed.shape[1] // (3 * dim), dim)
    return tuple(packed.new_empty(shape) for _ in range(3))


def can_split_qkv_norm(packed, norm_q, norm_k):
    # Match Intel ESIMD RMSNorm's FP32 arithmetic and final output cast.
    # Other norm backends / sensitive FP32 modes retain their own semantics.
    return (
        triton is not None
        and packed.device.type == "xpu"
        and packed.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(
            getattr(norm, "weight", None) is not None
            and norm.weight.device == packed.device
            and norm.weight.dtype == packed.dtype
            and norm.weight.is_contiguous()
            and norm.sensitive_layer_dtype == norm.infer_dtype
            for norm in (norm_q, norm_k)
        )
    )
