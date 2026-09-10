# SPDX-License-Identifier: Apache-2.0
"""Numerical kernels used by MiniMax-H3 DiT inference."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# The numerical kernels below are adapted from SGLang commit
# 8ef646a5c65bd2f8922483057dddc02e2b0de18c (Apache-2.0). Keeping the
# H3-specific subset here avoids importing an SGLang checkout at runtime.


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
def _indexed_scale_shift_bf16_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + index * stride_scale_row + columns, mask=mask, other=0.0).to(tl.float32)
    one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
    scaled = _round_bf16_to_fp32(x * one_plus_scale)
    tl.store(x_ptr + row * stride_x_row + columns, scaled + shift, mask=mask)


@triton.jit
def _indexed_gate_bf16_kernel(
    x_ptr,
    gate_ptr,
    other_ptr,
    indices_ptr,
    hidden_size,
    stride_x_row,
    stride_gate_row,
    stride_other_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0).to(tl.float32)
    other = tl.load(other_ptr + row * stride_other_row + columns, mask=mask, other=0.0).to(tl.float32)
    gated = _round_bf16_to_fp32(gate * other)
    tl.store(x_ptr + row * stride_x_row + columns, x + gated, mask=mask)


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


def _silu_mul_with_activation_rounding_inplace(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.shape[-1] % 2:
        raise ValueError(f"SwiGLU input width must be even, got {hidden_states.shape[-1]}")
    hidden_size = hidden_states.shape[-1] // 2
    if hidden_states.is_cuda and hidden_states.dtype is torch.bfloat16 and hidden_states.is_contiguous() and hidden_states.numel():
        _require_nvidia_triton(hidden_states, "H3 SwiGLU")
        rows = hidden_states.numel() // hidden_states.shape[-1]
        with torch.cuda.device(hidden_states.device):
            _packed_silu_mul_kernel[(rows, triton.cdiv(hidden_size, 1024))](
                hidden_states,
                hidden_states,
                rows,
                hidden_states.shape[-1],
                hidden_states.shape[-1],
                D=hidden_size,
                BLOCK=1024,
            )
        return hidden_states[..., :hidden_size]

    gate, value = hidden_states.chunk(2, dim=-1)
    F.silu(gate, inplace=True)
    return gate.mul_(value)


def _validate_indexed_modulation_inputs(
    operation: str,
    x: torch.Tensor,
    indices: torch.Tensor,
    lookup_tensors: tuple[tuple[str, torch.Tensor], ...],
    row_tensors: tuple[tuple[str, torch.Tensor], ...] = (),
) -> None:
    if x.ndim != 2:
        raise ValueError(f"{operation} expects a two-dimensional activation, got {x.shape}")
    if x.dtype is not torch.bfloat16:
        raise TypeError(f"{operation} requires BF16 activations, got {x.dtype}")
    if indices.ndim != 1 or indices.shape[0] != x.shape[0]:
        raise ValueError(f"{operation} indices must have shape ({x.shape[0]},), got {indices.shape}")
    if indices.dtype is not torch.long:
        raise TypeError(f"{operation} indices must use torch.long, got {indices.dtype}")
    if indices.device != x.device:
        raise ValueError(f"{operation} tensors must be on one device")

    hidden_size = x.shape[1]
    tensors = (("activation", x), *lookup_tensors, *row_tensors)
    for name, tensor in tensors:
        if tensor.device != x.device:
            raise ValueError(f"{operation} tensors must be on one device; {name} is on {tensor.device}")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{operation} requires BF16 {name}, got {tensor.dtype}")
        if tensor.ndim != 2 or tensor.shape[1] != hidden_size:
            raise ValueError(f"{operation} {name} must be two-dimensional with width {hidden_size}, got {tensor.shape}")
        if tensor.stride(-1) != 1 or (tensor.shape[0] > 1 and tensor.stride(0) < hidden_size):
            raise ValueError(f"{operation} {name} must have a non-overlapping contiguous last dimension, got stride {tensor.stride()}")

    lookup_rows = {tensor.shape[0] for _, tensor in lookup_tensors}
    if len(lookup_rows) > 1 or (x.shape[0] and lookup_rows == {0}):
        raise ValueError(f"{operation} lookup tensors must have the same nonzero row count")
    # H3 builds these indices from torch.unique(return_inverse=True); AdaLN
    # indices additionally include a bounded token tag. Their range is
    # guaranteed at the producer, while min/max here would synchronize the GPU
    # on every transformer block.
    for name, tensor in row_tensors:
        if tensor.shape != x.shape:
            raise ValueError(f"{operation} {name} must match activation shape {x.shape}, got {tensor.shape}")


def sgl_exact_indexed_scale_shift(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_indexed_modulation_inputs(
        "H3 indexed scale/shift",
        x,
        indices,
        (("shift", shift), ("scale", scale)),
    )
    if x.numel() == 0:
        return x
    if x.device.type != "cuda":
        selected_scale = scale.index_select(0, indices).float()
        selected_shift = shift.index_select(0, indices).float()
        one_plus_scale = (1.0 + selected_scale).to(torch.bfloat16).float()
        scaled = (x.float() * one_plus_scale).to(torch.bfloat16).float()
        return x.copy_((scaled + selected_shift).to(x.dtype))
    _require_nvidia_triton(x, "H3 indexed scale/shift")
    rows, hidden_size = x.shape
    with torch.cuda.device(x.device):
        _indexed_scale_shift_bf16_kernel[(rows,)](
            x,
            shift,
            scale,
            indices,
            hidden_size,
            x.stride(0),
            shift.stride(0),
            scale.stride(0),
            indices.stride(0),
            BLOCK_N=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return x


def sgl_exact_indexed_gate(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    _validate_indexed_modulation_inputs(
        "H3 indexed gate",
        x,
        indices,
        (("gate", gate),),
        (("other", other),),
    )
    if x.numel() == 0:
        return x
    if x.device.type != "cuda":
        selected_gate = gate.index_select(0, indices).float()
        gated = (selected_gate * other.float()).to(torch.bfloat16).float()
        return x.copy_((x.float() + gated).to(x.dtype))
    _require_nvidia_triton(x, "H3 indexed gate")
    rows, hidden_size = x.shape
    with torch.cuda.device(x.device):
        _indexed_gate_bf16_kernel[(rows,)](
            x,
            gate,
            other,
            indices,
            hidden_size,
            x.stride(0),
            gate.stride(0),
            other.stride(0),
            indices.stride(0),
            BLOCK_N=triton.next_power_of_2(hidden_size),
            num_warps=8,
        )
    return x
