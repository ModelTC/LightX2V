# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER

from .rms_norm_weight import RMSWeightTemplate

# Adapted from SGLang commit 8ef646a5c65bd2f8922483057dddc02e2b0de18c.


def _require_nvidia_triton(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda" and getattr(torch.version, "hip", None) is not None:
        raise RuntimeError("H3 Q/K normalization exact parity kernel supports NVIDIA CUDA only")


@triton.jit
def _mul_rn_f32(x, y):
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(x, y, z):
    return tl.inline_asm_elementwise(
        asm="fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[x, y, z],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rsqrt_approx_f32(x):
    return tl.inline_asm_elementwise(
        asm="rsqrt.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _h3_qknorm_128_kernel(
    x_ptr,
    weight_ptr,
    num_heads,
    token_stride,
    head_stride,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // num_heads
    head = row % num_heads
    lane = tl.arange(0, 32)
    base = x_ptr + token * token_stride + head * head_stride + lane * 4

    x0 = tl.load(base).to(tl.float32)
    x1 = tl.load(base + 1).to(tl.float32)
    x2 = tl.load(base + 2).to(tl.float32)
    x3 = tl.load(base + 3).to(tl.float32)
    accumulator = _fma_rn_f32(x0, x0, 0.0)
    accumulator = _fma_rn_f32(x1, x1, accumulator)
    accumulator = _fma_rn_f32(x2, x2, accumulator)
    accumulator = _fma_rn_f32(x3, x3, accumulator)

    # Match the CUDA warp's SHFL.BFLY reduction order: 16, 8, 4, 2, 1.
    accumulator = tl.sum(tl.reshape(accumulator, (2, 16), can_reorder=False), axis=0)
    accumulator = tl.sum(tl.reshape(accumulator, (2, 8), can_reorder=False), axis=0)
    accumulator = tl.sum(tl.reshape(accumulator, (2, 4), can_reorder=False), axis=0)
    accumulator = tl.sum(tl.reshape(accumulator, (2, 2), can_reorder=False), axis=0)
    accumulator = tl.sum(tl.reshape(accumulator, (2, 1), can_reorder=False), axis=0)
    sum_of_squares = tl.sum(accumulator)
    rstd = _rsqrt_approx_f32(_fma_rn_f32(sum_of_squares, 0.0078125, EPS))

    weight_base = weight_ptr + lane * 4
    w0 = tl.load(weight_base).to(tl.float32)
    w1 = tl.load(weight_base + 1).to(tl.float32)
    w2 = tl.load(weight_base + 2).to(tl.float32)
    w3 = tl.load(weight_base + 3).to(tl.float32)
    y0 = _mul_rn_f32(_mul_rn_f32(x0, rstd), w0)
    y1 = _mul_rn_f32(_mul_rn_f32(x1, rstd), w1)
    y2 = _mul_rn_f32(_mul_rn_f32(x2, rstd), w2)
    y3 = _mul_rn_f32(_mul_rn_f32(x3, rstd), w3)
    tl.store(base, y0)
    tl.store(base + 1, y1)
    tl.store(base + 2, y2)
    tl.store(base + 3, y3)


def _apply_qk_rms_norm_sglang(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    head_dim = 128
    if hidden_states.ndim != 3 or hidden_states.shape[-1] != head_dim:
        raise ValueError(f"H3 reference Q/K normalization expects [tokens, heads, 128], got {hidden_states.shape}")
    if hidden_states.device != weight.device:
        raise ValueError("H3 reference Q/K normalization tensors must be on one device")
    if hidden_states.device.type != "cuda":
        return F.rms_norm(hidden_states.float(), (head_dim,), weight.float(), eps).to(hidden_states.dtype)
    _require_nvidia_triton(hidden_states)
    if hidden_states.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise TypeError("H3 reference Q/K normalization requires BF16 activations and weights")
    if hidden_states.shape[1] <= 0:
        raise ValueError(f"H3 reference Q/K normalization requires at least one head, got {hidden_states.shape}")
    if hidden_states.stride(-1) != 1 or hidden_states.stride(-2) != head_dim:
        raise ValueError(f"Unsupported H3 reference Q/K strides: {hidden_states.stride()}")
    if hidden_states.shape[0] > 1 and hidden_states.stride(0) < hidden_states.shape[1] * head_dim:
        raise ValueError(f"Overlapping H3 reference Q/K token strides: {hidden_states.stride()}")
    if weight.shape != (head_dim,) or not weight.is_contiguous():
        raise ValueError("H3 reference Q/K normalization weights must be a contiguous [128] tensor")
    with torch.cuda.device(hidden_states.device):
        if hidden_states.numel():
            _h3_qknorm_128_kernel[(hidden_states.shape[0] * hidden_states.shape[1],)](
                hidden_states,
                weight,
                hidden_states.shape[1],
                hidden_states.stride(0),
                hidden_states.stride(1),
                EPS=float(eps),
                num_warps=1,
            )
    return hidden_states


@RMS_WEIGHT_REGISTER("h3_sgl_rms_norm")
class MiniMaxH3SGLQKRMSNorm(RMSWeightTemplate):
    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _apply_qk_rms_norm_sglang(input_tensor, self._get_actual_weight(), self.eps)


__all__ = ["MiniMaxH3SGLQKRMSNorm"]
