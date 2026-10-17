# SPDX-License-Identifier: Apache-2.0
"""Precision-preserving BF16 GELU + ConvRot INT8 FFN activation fusion.

Only the activation quantizer is locally compiled; GEMM and its epilogue remain
in the installed comfy-kitchen backend. Unsupported shapes/backends execute the
original GELU followed by the original ConvRot linear. Call
``prepare_convrot_gelu_fusion`` during warmup to compile before CUDA capture.
"""

import ctypes
from pathlib import Path
from threading import Lock

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.edenlive.convrot_projection import _backend_apis_available, convrot_projection_mm

try:
    import comfy_kitchen
    import comfy_kitchen.backends.cuda as kitchen_cuda
except ImportError:
    comfy_kitchen = kitchen_cuda = None


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
                        name="lightx2v_edenlive_convrot_gelu_sm120",
                        sources=[str(Path(__file__).with_name("convrot_gelu_quant.cu"))],
                        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_120"],
                        is_python_module=False,
                        verbose=False,
                    )
                    _extension = ctypes.CDLL(library)
                    _extension.edenlive_prepare_convrot_gelu.argtypes = []
                    _extension.edenlive_prepare_convrot_gelu.restype = ctypes.c_int
                    _extension.edenlive_gelu_convrot_quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
                    _extension.edenlive_gelu_convrot_quant.restype = ctypes.c_int
                    _extension.edenlive_rounded_gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
                    _extension.edenlive_rounded_gelu.restype = ctypes.c_int
                except (OSError, RuntimeError) as error:
                    _extension_error = error
                    return False
            if index not in _prepared_devices:
                error = _extension.edenlive_prepare_convrot_gelu()
                if error:
                    return False
                _prepared_devices.add(index)
    return True


def _validate(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize):
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


def _supported(input_tensor, weight, weight_scale, out_dtype, groupsize):
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
        error = _extension.edenlive_gelu_convrot_quant(
            input_tensor.data_ptr(), quantized.data_ptr(), scales.data_ptr(), input_tensor.shape[0], torch.cuda.current_stream(input_tensor.device).cuda_stream
        )
    if error:
        raise RuntimeError(f"EdenLive GELU ConvRot quantizer CUDA error {error}")
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
    _validate(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize)
    if comfy_kitchen is None:
        raise ImportError("ConvRot activation fusion requires comfy-kitchen")
    if input_tensor.shape[0] == 0:
        return input_tensor.new_empty((0, weight.shape[0]), dtype=out_dtype)
    if _supported(input_tensor, weight, weight_scale, out_dtype, convrot_groupsize):
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
    _validate(input_tensor, weight, weight_scale, bias, out_dtype, convrot_groupsize)
    return input_tensor.new_empty((input_tensor.shape[0], weight.shape[0]), dtype=out_dtype)
