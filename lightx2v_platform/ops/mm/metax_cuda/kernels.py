"""MetaX per-token INT8 quantization and per-channel scaled GEMM."""

from functools import lru_cache
from importlib import import_module

import torch


@lru_cache(maxsize=1)
def _load_metax_int8_ops():
    import_module("mcoplib._C")
    return torch.ops._C.dynamic_scaled_int8_quant, torch.ops._C.cutlass_scaled_mm


@torch.library.custom_op("lightx2v::metax_int8_linear", mutates_args=(), device_types="cuda")
def _metax_int8_linear(input_tensor: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    # Hide extension loading and mutable native buffers from Inductor tracing.
    quantize, scaled_mm = _load_metax_int8_ops()
    output = input_tensor.new_empty((input_tensor.shape[0], weight.shape[1]))
    quantized = torch.empty_like(input_tensor, dtype=torch.int8)
    scales = torch.empty((input_tensor.shape[0], 1), device=input_tensor.device, dtype=torch.float32)
    quantize(quantized, input_tensor, scales, None)
    scaled_mm(output, quantized, weight, scales, weight_scale, bias)
    return output


@_metax_int8_linear.register_fake
def _metax_int8_linear_fake(input_tensor, weight, weight_scale, bias):
    return input_tensor.new_empty((input_tensor.shape[0], weight.shape[1]))


def metax_int8_linear(input_tensor, weight, weight_scale, bias=None):
    """Apply an INT8 (K, N) weight while preserving leading input dimensions."""
    # C500 GEMM accepts incompatible dimensions/dtypes without rejecting them.
    if input_tensor.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("int8-metax requires FP16/BF16 activations")
    if input_tensor.shape[-1] != weight.shape[0]:
        raise ValueError("int8-metax input and weight contraction dimensions must match")
    if weight.dtype != torch.int8 or weight_scale.dtype != torch.float32:
        raise TypeError("int8-metax requires INT8 weights and FP32 channel scales")
    output_shape = (*input_tensor.shape[:-1], weight.shape[1])
    flat_input = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
    if flat_input.shape[0] == 0:
        return input_tensor.new_empty(output_shape)
    return _metax_int8_linear(flat_input, weight, weight_scale, bias).reshape(output_shape)
