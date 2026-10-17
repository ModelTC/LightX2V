# SPDX-License-Identifier: Apache-2.0
"""EdenLive BF16 GELU followed by SGL per-token FP8 quantization.

The original Triton kernel below preserves the numerical boundaries of the
existing Inductor GELU(tanh) and installed SGL quantizer. In particular, GELU
rounds to BF16 before reduction, and scale and reciprocal round separately.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gelu_quant_fp8_sgl_kernel(input_ptr, quantized_ptr, scale_ptr, width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    x = tl.load(input_ptr + row * width + columns, columns < width, 0.0).to(tl.float32)

    # Match the compiled GELU's operation order with FP fusion disabled.
    half_x = x * 0.5
    cubic = (x * x) * x
    argument = (x + cubic * 0.044715) * 0.7978845608028654
    activated = (half_x * (1.0 + libdevice.tanh(argument))).to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.where(columns < width, tl.abs(activated), 0.0), 0), 0.0)
    scale = tl.div_rn(absmax, 448.0)
    inverse_scale = tl.div_rn(1.0, scale)
    values = activated * inverse_scale
    # SGL has no positive epsilon floor. Preserve its NaN clamp order when
    # a zero row produces scale=0 and activated * inverse_scale is NaN.
    values = tl.maximum(tl.minimum(values, 448.0), -448.0)
    tl.store(quantized_ptr + row * width + columns, values, columns < width)
    tl.store(scale_ptr + row, scale)


def _validate_input(input_tensor):
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16:
        raise ValueError("fused_gelu_quant_fp8_sgl requires a two-dimensional BF16 input")
    if input_tensor.shape[1] == 0:
        raise ValueError("fused_gelu_quant_fp8_sgl requires a nonempty hidden dimension")


@torch.library.custom_op("lightx2v::fused_gelu_quant_fp8_sgl", mutates_args=(), device_types="cuda")
def fused_gelu_quant_fp8_sgl(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 activations and FP32 [tokens, 1] scales for BF16 GELU(tanh).

    This matches the EdenLive compiled path with ``emulate_precision_casts``
    enabled. It is intended for finite activations and does not change input.
    """
    _validate_input(input_tensor)
    input_tensor = input_tensor.contiguous()
    quantized = torch.empty(input_tensor.shape, device=input_tensor.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((input_tensor.shape[0], 1), device=input_tensor.device, dtype=torch.float32)
    if input_tensor.shape[0]:
        _gelu_quant_fp8_sgl_kernel[(input_tensor.shape[0],)](
            input_tensor,
            quantized,
            scales,
            input_tensor.shape[1],
            triton.next_power_of_2(input_tensor.shape[1]),
            num_warps=8,
            enable_fp_fusion=False,
        )
    return quantized, scales


@fused_gelu_quant_fp8_sgl.register_fake
def _fused_gelu_quant_fp8_sgl_fake(input_tensor):
    _validate_input(input_tensor)
    return (
        input_tensor.new_empty(input_tensor.shape, dtype=torch.float8_e4m3fn),
        input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32),
    )
