# SPDX-License-Identifier: Apache-2.0
"""Precision-preserving GELU/activation quantization for FP8-SGL and ConvRot.

Both backends preserve the compiled GELU(tanh) BF16 rounding boundary and
reuse their original quantization/GEMM arithmetic. FP8-SGL uses a Triton
quantizer; INT8 ConvRot lazily compiles its SM120 CUDA quantizer before capture.
Unsupported ConvRot shapes/backends retain the original GELU + linear path.
"""

import ctypes
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from lightx2v.common.ops.mm.projection_group import _backend_apis_available, convrot_projection_mm

try:
    import comfy_kitchen
    import comfy_kitchen.backends.cuda as kitchen_cuda
except ImportError:
    comfy_kitchen = kitchen_cuda = None


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


def _validate_fp8_input(input_tensor):
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16:
        raise ValueError("fused_gelu_quant_fp8_sgl requires a two-dimensional BF16 input")
    if input_tensor.shape[1] == 0:
        raise ValueError("fused_gelu_quant_fp8_sgl requires a nonempty hidden dimension")


@torch.library.custom_op("lightx2v::fused_gelu_quant_fp8_sgl", mutates_args=(), device_types="cuda")
def fused_gelu_quant_fp8_sgl(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 activations and FP32 [tokens, 1] scales for BF16 GELU(tanh).

    This matches the compiled path with ``emulate_precision_casts``
    enabled. It is intended for finite activations and does not change input.
    """
    _validate_fp8_input(input_tensor)
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
    _validate_fp8_input(input_tensor)
    return (
        input_tensor.new_empty(input_tensor.shape, dtype=torch.float8_e4m3fn),
        input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32),
    )


_extension = None
_extension_error = None
_prepared_devices = set()
_extension_lock = Lock()


def prepare_convrot_gelu_fusion(device=None):
    """Compile/load the SM120 quantizer outside CUDA capture; return readiness.

    A missing compiler or older backend leaves the original linear available.
    Build products use PyTorch's extension cache; no installed package changes.
    """
    global _extension, _extension_error
    if not torch.cuda.is_available() or not _backend_apis_available():
        return False
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    with torch.cuda.device(index):
        if torch.cuda.is_current_stream_capturing():
            return index in _prepared_devices
        with _extension_lock:
            if _extension_error is not None:
                return False
            if _extension is None:
                try:
                    from torch.utils.cpp_extension import load

                    library = load(
                        name="lightx2v_convrot_gelu_sm120",
                        sources=[str(Path(__file__).resolve().parents[4] / "lightx2v_kernel/nvidia/csrc/quantization/convrot_gelu_quant.cu")],
                        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
                        is_python_module=False,
                        verbose=False,
                    )
                    _extension = ctypes.CDLL(library)
                    _extension.lightx2v_prepare_convrot_gelu.argtypes = []
                    _extension.lightx2v_prepare_convrot_gelu.restype = ctypes.c_int
                    _extension.lightx2v_gelu_convrot_quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
                    _extension.lightx2v_gelu_convrot_quant.restype = ctypes.c_int
                except (OSError, RuntimeError) as error:
                    _extension_error = error
                    return False
            if index not in _prepared_devices:
                error = _extension.lightx2v_prepare_convrot_gelu()
                if error:
                    return False
                _prepared_devices.add(index)
    return True


def _validate_convrot(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize):
    if input_tensor.ndim != 2 or input_tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("fused_gelu_int8_convrot_linear requires a two-dimensional floating input")
    if convrot_groupsize not in (16, 64, 256):
        raise ValueError("ConvRot group size must be 16, 64, or 256")
    if input_tensor.shape[1] == 0 or input_tensor.shape[1] % convrot_groupsize:
        raise ValueError("ConvRot input width must be nonempty and divisible by its group size")
    if weight.ndim != 2 or weight.dtype != torch.int8 or weight.shape[1] != input_tensor.shape[1]:
        raise ValueError("ConvRot requires an INT8 weight with matching input width")
    if weight_scale.dtype != torch.float32 or weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError("ConvRot requires a scalar or per-output FP32 weight scale")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("ConvRot bias must match the output width")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("ConvRot requires floating output")


def _supports_convrot_fusion(input_tensor, weight, weight_scale, out_dtype, groupsize):
    if (
        input_tensor.dtype != torch.bfloat16
        or out_dtype != torch.bfloat16
        or input_tensor.shape[0] < 3
        or input_tensor.shape[1] != 14336
        or weight.shape[0] % 16
        or groupsize != 256
        or not _backend_apis_available()
        or torch.cuda.get_device_capability(input_tensor.device) != (12, 0)
    ):
        return False
    implementation = comfy_kitchen.registry.get_implementation("int8_linear", kwargs={"x": input_tensor, "weight": weight, "weight_scale": weight_scale})
    return implementation is kitchen_cuda.int8_linear and prepare_convrot_gelu_fusion(input_tensor.device)


@torch.library.custom_op("lightx2v::fused_gelu_convrot_int8_quant", mutates_args=(), device_types="cuda")
def fused_gelu_convrot_int8_quant(input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return original INT8 bytes and FP32 scales for BF16 [M>=3, 14336] GELU.

    Original GELU, rotated-value and scale BF16 rounding boundaries are retained.
    The supported hot path is SM120 with group size 256.
    """
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16 or input_tensor.shape[1] != 14336 or input_tensor.shape[0] < 3:
        raise ValueError("fused_gelu_convrot_int8_quant requires BF16 [M>=3, 14336]")
    if not prepare_convrot_gelu_fusion(input_tensor.device):
        return kitchen_cuda.quantize_int8_rowwise_convrot64(F.gelu(input_tensor.contiguous(), approximate="tanh"), 256)
    input_tensor = input_tensor.contiguous()
    quantized = torch.empty_like(input_tensor, dtype=torch.int8)
    scales = input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32)
    with torch.cuda.device(input_tensor.device):
        error = _extension.lightx2v_gelu_convrot_quant(
            input_tensor.data_ptr(), quantized.data_ptr(), scales.data_ptr(), input_tensor.shape[0], torch.cuda.current_stream(input_tensor.device).cuda_stream
        )
    if error:
        raise RuntimeError(f"GELU ConvRot quantizer CUDA error {error}")
    return quantized, scales


@fused_gelu_convrot_int8_quant.register_fake
def _fused_gelu_convrot_int8_quant_fake(input_tensor):
    if input_tensor.ndim != 2 or input_tensor.dtype != torch.bfloat16 or input_tensor.shape[1] != 14336 or input_tensor.shape[0] < 3:
        raise ValueError("fused_gelu_convrot_int8_quant requires BF16 [M>=3, 14336]")
    return input_tensor.new_empty(input_tensor.shape, dtype=torch.int8), input_tensor.new_empty((input_tensor.shape[0], 1), dtype=torch.float32)


@torch.library.custom_op("lightx2v::fused_gelu_int8_convrot_linear", mutates_args=(), device_types="cuda")
def fused_gelu_int8_convrot_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    convrot_groupsize: int,
) -> torch.Tensor:
    """Precisely fuse GELU/rotation/quantization and reuse the original GEMM."""
    _validate_convrot(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize)
    if comfy_kitchen is None:
        raise ImportError("ConvRot activation fusion requires comfy-kitchen")
    if input_tensor.shape[0] == 0:
        return input_tensor.new_empty((0, weight.shape[0]), dtype=out_dtype)
    if _supports_convrot_fusion(input_tensor, weight, weight_scale, out_dtype, convrot_groupsize):
        quantized, scales = fused_gelu_convrot_int8_quant(input_tensor)
        return convrot_projection_mm(quantized, weight.contiguous(), scales, weight_scale.contiguous(), bias, out_dtype)
    return comfy_kitchen.int8_linear(
        F.gelu(input_tensor.contiguous(), approximate="tanh"),
        weight.contiguous(),
        weight_scale.contiguous(),
        bias,
        out_dtype=out_dtype,
        convrot=True,
        convrot_groupsize=convrot_groupsize,
    )


@fused_gelu_int8_convrot_linear.register_fake
def _fused_gelu_int8_convrot_linear_fake(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize):
    _validate_convrot(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize)
    return input_tensor.new_empty((input_tensor.shape[0], weight.shape[0]), dtype=out_dtype)
