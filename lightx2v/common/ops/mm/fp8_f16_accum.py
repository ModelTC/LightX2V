import math

import torch

from lightx2v.common.ops.mm.triton_kernels import fp8_quantize_range_triton

try:
    from lightx2v_kernel.gemm import FP8_F16_ACCUM_MM_AVAILABLE, cutlass_scaled_fp8_mm_f16_accum
except ImportError:
    FP8_F16_ACCUM_MM_AVAILABLE = False
    cutlass_scaled_fp8_mm_f16_accum = None


def fp8_f16_accum_mm_unavailable_reason():
    if not FP8_F16_ACCUM_MM_AVAILABLE:
        return "the lightx2v-kernel extension does not provide the FP8-F16 accumulation op"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        return f"SM120 is required, but the current CUDA capability is SM{capability[0]}{capability[1]}"
    return None


def fp8_f16_accum_mm_available():
    return fp8_f16_accum_mm_unavailable_reason() is None


def validate_fp8_f16_accum_qmax(activation_qmax):
    activation_qmax = float(activation_qmax)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    if not math.isfinite(activation_qmax) or not 0 < activation_qmax <= fp8_max:
        raise ValueError(f"FP8 activation qmax must be finite and in (0, {fp8_max}], got {activation_qmax}")
    return activation_qmax


def fp8_f16_accum_linear(input_tensor, weight, weight_scale, bias, activation_qmax):
    input_shape = input_tensor.shape
    input_matrix = input_tensor.reshape(-1, input_shape[-1])
    quantized, activation_scale = fp8_quantize_range_triton(input_matrix, activation_qmax)
    output = cutlass_scaled_fp8_mm_f16_accum(
        quantized,
        weight,
        activation_scale,
        weight_scale.float(),
        input_tensor.dtype,
        bias,
    )
    return output.view(*input_shape[:-1], weight.shape[1])
