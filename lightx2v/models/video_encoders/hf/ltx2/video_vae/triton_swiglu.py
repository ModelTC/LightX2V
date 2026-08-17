# Copyright 2026 Lightricks Ltd.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 DiffVAE's CUDA SwiGLU kernels without an ``ltx_core`` dependency.

The released chunked-eager decoder selects these kernels for BF16 CUDA
activations.  Keeping that dispatch is important for source-level numerical
parity as well as for the decoder's peak-memory bound.
"""

from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - host dependent
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


_DUAL_GATE_UP_DIM_MIN: Final[int] = 256
_DUAL_GATE_UP_DIM_MAX: Final[int] = 2048


def triton_swiglu_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_gate_up_swiglu_kernel(
        x_ptr,
        w_gate_ptr,
        w_up_ptr,
        out_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_gate_n,
        stride_gate_k,
        stride_up_n,
        stride_up_k,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused gate/up GEMMs followed by BF16-rounded SwiGLU."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                w_gate_ptr + offs_n[None, :] * stride_gate_n + offs_k[:, None] * stride_gate_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            up_w = tl.load(
                w_up_ptr + offs_n[None, :] * stride_up_n + offs_k[:, None] * stride_up_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_w)
            up_acc += tl.dot(x, up_w)

        silu_bf16 = (gate_acc * tl.sigmoid(gate_acc)).to(tl.bfloat16)
        product = (silu_bf16.to(tl.float32) * up_acc).to(tl.bfloat16)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            product,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _dual_gate_up_eligible(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> bool:
    if not triton_swiglu_available() or not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or w_gate.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        return False
    dim = x.shape[-1]
    hidden = w_gate.shape[0]
    if w_up.shape[0] != hidden or w_gate.shape[1] != dim or w_up.shape[1] != dim:
        return False
    return hidden == 4 * dim and _DUAL_GATE_UP_DIM_MIN <= dim <= _DUAL_GATE_UP_DIM_MAX


def _fused_gate_up_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    m, k = x.shape
    n = w_gate.shape[0]
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_gate_up_swiglu_kernel[grid](
        x,
        w_gate,
        w_up,
        out,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        w_gate.stride(0),
        w_gate.stride(1),
        w_up.stride(0),
        w_up.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )


def _token_slices(n_tokens: int, tile_size: int) -> list[slice]:
    return [slice(start, min(n_tokens, start + tile_size)) for start in range(0, n_tokens, tile_size)]


def swiglu_tiled(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """Match upstream ``swiglu_tiled`` for the by-size tile policy."""
    if x.dtype != w_gate.dtype:
        x = x.to(dtype=w_gate.dtype)
    leading = x.shape[:-1]
    dim = x.shape[-1]
    flat = x.reshape(-1, dim).contiguous()
    if flat.shape[0] == 0:
        return x
    output = torch.empty_like(flat)
    workspace = torch.empty(
        (min(flat.shape[0], tile_size), w_gate.shape[0]),
        device=x.device,
        dtype=x.dtype,
    )
    use_dual = _dual_gate_up_eligible(flat, w_gate, w_up)
    w_gate_c = w_gate.contiguous() if use_dual else w_gate
    w_up_c = w_up.contiguous() if use_dual else w_up
    for interval in _token_slices(flat.shape[0], tile_size):
        source = flat[interval]
        work = workspace[: source.shape[0]]
        if use_dual:
            _fused_gate_up_swiglu(source, w_gate_c, w_up_c, work)
        else:
            torch.mm(source, w_gate.t(), out=work)
            F.silu(work, inplace=True)
            work.mul_(F.linear(source, w_up))
        torch.mm(work, w_down.t(), out=output[interval])
    return output.view(*leading, dim)


def residual_modulating_mlp(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tile_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match upstream chunked path's in-place modulated SwiGLU residual."""
    if not x.is_contiguous():
        x = x.contiguous()
    dim = x.shape[-1]
    flat = x.reshape(-1, dim)
    if flat.shape[0] == 0:
        return x
    affine_scale = scale.reshape(-1, dim)
    affine_shift = shift.reshape(-1, dim)
    max_tokens = min(flat.shape[0], tile_size)
    workspace = torch.empty((max_tokens, w_gate.shape[0]), device=x.device, dtype=x.dtype)
    normalized = torch.empty((max_tokens, dim), device=x.device, dtype=x.dtype)
    output = torch.empty((max_tokens, dim), device=x.device, dtype=x.dtype)
    use_dual = _dual_gate_up_eligible(flat, w_gate, w_up)
    w_gate_c = w_gate.contiguous() if use_dual else w_gate
    w_up_c = w_up.contiguous() if use_dual else w_up

    for interval in _token_slices(flat.shape[0], tile_size):
        source = flat[interval]
        count = source.shape[0]
        norm = normalized[:count]
        norm.copy_(F.rms_norm(source, (dim,), norm_weight, eps))
        norm.mul_(1.0 + affine_scale).add_(affine_shift)
        work = workspace[:count]
        if use_dual:
            _fused_gate_up_swiglu(norm, w_gate_c, w_up_c, work)
        else:
            torch.mm(norm, w_gate.t(), out=work)
            F.silu(work, inplace=True)
            work.mul_(F.linear(norm, w_up))
        torch.mm(work, w_down.t(), out=output[:count])
        source.add_(output[:count])
    return x


__all__ = ["residual_modulating_mlp", "swiglu_tiled", "triton_swiglu_available"]
