# SPDX-License-Identifier: Apache-2.0
"""Local numerical kernels for MiniMax-H3's SGLang-compatible execution path."""

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


def apply_qk_rms_norm_sglang(
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
    _require_nvidia_triton(hidden_states, "H3 Q/K normalization")
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


def indexed_scale_shift_sglang(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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


def indexed_gate_sglang(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
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


def apply_vae_silu_mul_sglang(hidden_states: torch.Tensor) -> torch.Tensor:
    return _silu_mul_with_activation_rounding(hidden_states)


def scaled_residual_add_vae_sglang(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    fused = _try_scaled_residual_add_exact(residual, hidden_states, scale)
    return residual + hidden_states * scale if fused is None else fused
