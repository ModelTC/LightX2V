"""Fuse H3 SwiGLU and reduced-range FP8 activation quantization."""

import torch
import triton
import triton.language as tl

from lightx2v.common.ops.mm.fp8_f16_accum import validate_fp8_f16_accum_qmax


@triton.jit
def _swiglu_quant_kernel(
    value_ptr,
    gate_ptr,
    quantized_ptr,
    scale_ptr,
    value_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    width: tl.constexpr,
    qmax: tl.constexpr,
    activation_dtype: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    value = tl.load(value_ptr + row * value_row_stride + columns, columns < width, 0).to(tl.float32)
    gate = tl.load(gate_ptr + row * gate_row_stride + columns, columns < width, 0).to(tl.float32)

    # The source computes SiLU and then the product in BF16/FP16. Preserve
    # both rounding boundaries before applying the existing row quantizer.
    silu = (gate / (1.0 + tl.exp(-gate))).to(activation_dtype).to(tl.float32)
    activated = (value * silu).to(activation_dtype).to(tl.float32)
    absmax = tl.max(tl.where(columns < width, tl.abs(activated), 0.0), 0)
    scale = tl.maximum(absmax, 1e-8) / qmax
    quantized = tl.clamp(activated / scale, -qmax, qmax)
    tl.store(quantized_ptr + row * width + columns, quantized, columns < width)
    tl.store(scale_ptr + row, scale)


def _validate_inputs(value, gate, qmax):
    if value.ndim != 2 or value.dtype not in (torch.bfloat16, torch.float16) or value.shape[1] == 0:
        raise ValueError("SwiGLU FP8 fusion requires nonempty two-dimensional BF16 or FP16 inputs")
    if gate.shape != value.shape or gate.dtype != value.dtype or gate.device != value.device:
        raise ValueError("SwiGLU value and gate must have matching shape, dtype, and device")
    if value.stride(1) != 1 or gate.stride(1) != 1:
        raise ValueError("SwiGLU value and gate must have contiguous hidden dimensions")
    validate_fp8_f16_accum_qmax(qmax)


@torch.library.custom_op("lightx2v::fused_swiglu_quant_fp8_f16_accum", mutates_args=(), device_types="cuda")
def fused_swiglu_quant_fp8_f16_accum(value: torch.Tensor, gate: torch.Tensor, qmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize SwiGLU output without materializing its BF16/FP16 tensor."""
    _validate_inputs(value, gate, qmax)
    quantized = torch.empty(value.shape, dtype=torch.float8_e4m3fn, device=value.device)
    scales = torch.empty(value.shape[0], dtype=torch.float32, device=value.device)
    if value.shape[0]:
        _swiglu_quant_kernel[(value.shape[0],)](
            value,
            gate,
            quantized,
            scales,
            value.stride(0),
            gate.stride(0),
            value.shape[1],
            qmax,
            tl.bfloat16 if value.dtype == torch.bfloat16 else tl.float16,
            triton.next_power_of_2(value.shape[1]),
            num_warps=8,
            enable_fp_fusion=False,
        )
    return quantized, scales


@fused_swiglu_quant_fp8_f16_accum.register_fake
def _fused_swiglu_quant_fp8_f16_accum_fake(value, gate, qmax):
    _validate_inputs(value, gate, qmax)
    return value.new_empty(value.shape, dtype=torch.float8_e4m3fn), value.new_empty((value.shape[0],), dtype=torch.float32)
