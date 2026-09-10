# SPDX-License-Identifier: Apache-2.0
"""Numerical kernels used by the MiniMax-H3 video VAE."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# The numerical kernels below are adapted from SGLang commit
# 8ef646a5c65bd2f8922483057dddc02e2b0de18c (Apache-2.0). Keeping the
# H3-specific subset here avoids importing an SGLang checkout at runtime.


def _supports_nvidia_triton(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "cuda" and getattr(torch.version, "hip", None) is None


def _require_nvidia_triton(tensor: torch.Tensor, operation: str) -> None:
    if tensor.device.type != "cuda":
        return
    if getattr(torch.version, "hip", None) is not None:
        raise RuntimeError(f"{operation} exact parity kernel supports NVIDIA CUDA only")


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round FP32 to BF16 precision while retaining an FP32 register."""
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _mul_rn_f32(x, y):
    """Correctly-rounded FP32 multiply which cannot contract into an FMA."""
    return tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, y],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _packed_silu_mul_kernel(
    output_ptr,
    x_ptr,
    num_rows,
    row_stride,
    output_row_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    mask = (row < num_rows) & (columns < D)
    row_base = row * row_stride
    gate = tl.load(x_ptr + row_base + columns, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(x_ptr + row_base + D + columns, mask=mask, other=0.0).to(tl.float32)
    activated = _round_bf16_to_fp32(gate * tl.sigmoid(gate))
    tl.store(output_ptr + row * output_row_stride + columns, activated * value, mask=mask)


@triton.jit
def _scaled_residual_add_exact_kernel(
    output_ptr,
    residual_ptr,
    x_ptr,
    scale_ptr,
    numel: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr + offsets % width, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, residual + _mul_rn_f32(x, scale), mask=mask)


def _try_scaled_residual_add_exact(
    residual: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor | None:
    if (
        torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or residual.dtype != torch.float32
        or x.dtype not in (torch.float16, torch.bfloat16)
        or scale.dtype != torch.float32
        or not residual.is_cuda
        or not _supports_nvidia_triton(residual)
        or residual.device != x.device
        or residual.device != scale.device
        or residual.shape != x.shape
        or scale.shape != (x.shape[-1],)
        or not residual.is_contiguous()
        or not x.is_contiguous()
        or not scale.is_contiguous()
        or x.numel() == 0
    ):
        return None
    output = torch.empty_like(residual)
    block_size = 1024
    with torch.cuda.device(x.device):
        _scaled_residual_add_exact_kernel[(triton.cdiv(x.numel(), block_size),)](
            output,
            residual,
            x,
            scale,
            numel=x.numel(),
            width=x.shape[-1],
            BLOCK_SIZE=block_size,
        )
    return output


def _silu_mul_with_activation_rounding(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.shape[-1] % 2:
        raise ValueError(f"SwiGLU input width must be even, got {hidden_states.shape[-1]}")
    hidden_size = hidden_states.shape[-1] // 2
    if hidden_states.is_cuda and hidden_states.dtype is torch.bfloat16 and hidden_states.is_contiguous() and hidden_states.numel():
        _require_nvidia_triton(hidden_states, "H3 VAE SwiGLU")
        rows = hidden_states.numel() // hidden_states.shape[-1]
        output = hidden_states.new_empty(*hidden_states.shape[:-1], hidden_size)
        with torch.cuda.device(hidden_states.device):
            _packed_silu_mul_kernel[(rows, triton.cdiv(hidden_size, 1024))](
                output,
                hidden_states,
                rows,
                hidden_states.shape[-1],
                hidden_size,
                D=hidden_size,
                BLOCK=1024,
            )
        return output

    gate, value = hidden_states.chunk(2, dim=-1)
    return F.silu(gate).mul_(value)


def sgl_exact_vae_silu_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    return _silu_mul_with_activation_rounding(hidden_states)


def sgl_exact_vae_scaled_residual_add(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    fused = _try_scaled_residual_add_exact(residual, hidden_states, scale)
    return residual + hidden_states * scale if fused is None else fused
