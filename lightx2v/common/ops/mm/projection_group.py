"""Share input quantization across resident FP8-SGL or INT8 ConvRot projections.

Each backend retains its original quantizer, GEMM and rounding boundaries.
Optional packing shares weight storage with the original modules; unsupported
inputs, adapters or reloaded weights use those original module operations.
"""

import torch

from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl, MMWeightWint8ConvRot
from lightx2v.common.ops.mm.sgl_kernel import sgl_fp8_per_token_quant, sgl_fp8_scaled_mm, sgl_kernel

try:
    import comfy_kitchen
    import comfy_kitchen.backends.cuda as kitchen_cuda
except ImportError:
    comfy_kitchen = kitchen_cuda = None


def _backend_apis_available():
    """Older comfy-kitchen builds can keep the original ConvRot module path."""
    required = (
        "int8_linear",
        "quantize_int8_rowwise_convrot64",
        "_wrap_for_dlpack",
        "_int8_weight_scale_arg",
        "_gemm_vector_arg",
        "_empty_cuda_tensor",
        "_prefer_cublas_int8_fallback",
        "_DISABLE_CUTLASS_INT8",
        "_cuda_device_supports_cutlass_int8_dequant",
        "get_cublas_workspace",
        "DTYPE_TO_CODE",
    )
    kernels = ("cutlass_int8_dequant", "cublas_gemm_int8", "dequantize_int8_linear")
    return (
        comfy_kitchen is not None
        and kitchen_cuda is not None
        and all(hasattr(kitchen_cuda, name) for name in required)
        and all(hasattr(getattr(kitchen_cuda, "_C", None), name) for name in kernels)
        and hasattr(getattr(comfy_kitchen, "registry", None), "get_implementation")
    )


@torch.library.custom_op("lightx2v::convrot_projection_quant", mutates_args=(), device_types="cuda")
def convrot_projection_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Exactly the CUDA int8_linear path for group=256, K <= 4096, M >= 3.
    return kitchen_cuda.quantize_int8_rowwise_convrot64(x.contiguous(), 256)


@convrot_projection_quant.register_fake
def _convrot_projection_quant_fake(x):
    return x.new_empty(x.shape, dtype=torch.int8), x.new_empty((x.shape[0], 1), dtype=torch.float32)


@torch.library.custom_op("lightx2v::convrot_projection_mm", mutates_args=(), device_types="cuda")
def convrot_projection_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """The existing CUDA INT8 epilogue, with already quantized input.

    Keep CUTLASS/cuBLAS selection and the dequantization arithmetic identical
    to comfy-kitchen's CUDA int8_linear on the supported SM120 shapes.
    """
    cuda = kitchen_cuda
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    wrap = cuda._wrap_for_dlpack
    scale = cuda._int8_weight_scale_arg(weight_scale, x.device)
    bias_arg = cuda._gemm_vector_arg(bias, x.device, out_dtype) if bias is not None else cuda._empty_cuda_tensor(x.device, out_dtype)
    used_cutlass = False
    if not cuda._prefer_cublas_int8_fallback(m, n, k, x.get_device()) and not cuda._DISABLE_CUTLASS_INT8 and cuda._cuda_device_supports_cutlass_int8_dequant(x):
        channel_scale = scale if scale.numel() == n else scale.expand(n).contiguous()
        used_cutlass = cuda._C.cutlass_int8_dequant(wrap(x), wrap(weight), wrap(input_scale), wrap(channel_scale), wrap(bias_arg), wrap(out), cuda.DTYPE_TO_CODE[out_dtype], stream)
    if not used_cutlass:
        integer_out = torch.empty((m, n), dtype=torch.int32, device=x.device)
        cuda._C.cublas_gemm_int8(wrap(x), wrap(weight), wrap(integer_out), wrap(cuda.get_cublas_workspace()), stream)
        cuda._C.dequantize_int8_linear(wrap(integer_out), wrap(input_scale), wrap(scale), wrap(bias_arg), wrap(out), cuda.DTYPE_TO_CODE[out_dtype], stream)
    return out


@convrot_projection_mm.register_fake
def _convrot_projection_mm_fake(x, weight, input_scale, weight_scale, bias, out_dtype):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=out_dtype)


class ConvRotProjectionGroup:
    """Share a group=256 ConvRot input, optionally pack original INT8 weights.

    This intentionally supports only resident SM120 QKV/text-KV shapes. Small
    M, adapters, incompatible weights and changed module tensors use the
    original module implementations. Packed outputs are strided split views.
    Original modules share the packed weight storage, so packing adds no
    persistent copy of the QKV weights and their fallback remains usable.
    """

    @classmethod
    def from_modules(cls, modules, packed=False):
        modules = tuple(modules)
        if not _backend_apis_available() or not modules or any(type(module) is not MMWeightWint8ConvRot for module in modules):
            return None
        first = modules[0]
        weight = getattr(first, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.device.type != "cuda":
            return None
        device, input_dim, infer_dtype = weight.device, weight.shape[1], first.infer_dtype
        if torch.cuda.get_device_capability(device) != (12, 0) or input_dim % 256 or not 256 <= input_dim <= 4096 or infer_dtype not in (torch.bfloat16, torch.float16):
            return None
        has_bias = getattr(first, "bias", None) is not None
        for module in modules:
            if module.has_lora_branch or module.has_diff or hasattr(module, "weight_diff") or hasattr(module, "bias_diff"):
                return None
            if module.lazy_load or module.create_cuda_buffer or module.create_cpu_buffer or getattr(module, "_convrot_groupsize", None) != 256:
                return None
            weight, scale, bias = getattr(module, "weight", None), getattr(module, "weight_scale", None), getattr(module, "bias", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.device != device or weight.dtype != torch.int8 or not weight.is_contiguous():
                return None
            output_dim = weight.shape[0]
            if weight.shape[1] != input_dim or output_dim % 16 or module.infer_dtype != infer_dtype:
                return None
            if not isinstance(scale, torch.Tensor) or scale.device != device or scale.dtype != torch.float32 or scale.numel() != output_dim or not scale.is_contiguous():
                return None
            if scale.ndim not in (1, 2) or (bias is not None) != has_bias:
                return None
            if bias is not None and (not isinstance(bias, torch.Tensor) or bias.shape != (output_dim,) or bias.device != device or bias.dtype != infer_dtype or not bias.is_contiguous()):
                return None
        # Respect backend selection: do not replace Triton/eager math with CUDA.
        probe = torch.empty((3, input_dim), device=device, dtype=infer_dtype)
        implementation = comfy_kitchen.registry.get_implementation("int8_linear", kwargs={"x": probe, "weight": first.weight, "weight_scale": first.weight_scale})
        if implementation is not kitchen_cuda.int8_linear:
            return None
        return cls(modules, packed=packed)

    def __init__(self, modules, packed=False):
        self.modules = tuple(modules)
        self.weights = tuple(module.weight for module in modules)
        self.scales = tuple(module.weight_scale for module in modules)
        self.biases = tuple(getattr(module, "bias", None) for module in modules)
        self.infer_dtype = modules[0].infer_dtype
        self.device = self.weights[0].device
        self.input_dim = self.weights[0].shape[1]
        self.output_dims = tuple(weight.shape[0] for weight in self.weights)
        self.packed = packed
        if packed:
            # Finish every allocation before replacing any module tensor. Row
            # slices of these allocations are still contiguous and work with
            # the original module.apply, including its adapter/fallback paths.
            self.packed_weight = torch.cat(self.weights, dim=0)
            self.packed_scale = torch.cat([scale.reshape(-1) for scale in self.scales], dim=0)
            self.packed_bias = torch.cat(self.biases, dim=0) if self.biases[0] is not None else None
            weights = self.packed_weight.split(self.output_dims, dim=0)
            scales = self.packed_scale.split(self.output_dims, dim=0)
            biases = self.packed_bias.split(self.output_dims, dim=0) if self.packed_bias is not None else (None,) * len(self.modules)
            for module, weight, scale, bias, original_scale in zip(self.modules, weights, scales, biases, self.scales):
                module.weight = weight
                module.weight_scale = scale.view(original_scale.shape)
                module.bias = bias
            # Identity guards must track the new module views, and retaining
            # the previous tuple here would keep the duplicate storage alive.
            self.weights = tuple(module.weight for module in self.modules)
            self.scales = tuple(module.weight_scale for module in self.modules)
            self.biases = tuple(module.bias for module in self.modules)

    def apply(self, x):
        for module, weight, scale, bias in zip(self.modules, self.weights, self.scales, self.biases):
            if (
                module.has_lora_branch
                or module.has_diff
                or hasattr(module, "weight_diff")
                or hasattr(module, "bias_diff")
                or module.weight is not weight
                or module.weight_scale is not scale
                or getattr(module, "bias", None) is not bias
                or module.infer_dtype != self.infer_dtype
                or module._convrot_groupsize != 256
                or module.lazy_load
                or module.create_cuda_buffer
                or module.create_cpu_buffer
            ):
                return tuple(module.apply(x) for module in self.modules)
        if x.ndim != 2 or x.device != self.device or x.dtype != self.infer_dtype or x.shape[1] != self.input_dim or x.shape[0] < 3:
            return tuple(module.apply(x) for module in self.modules)
        quantized, input_scale = convrot_projection_quant(x)
        if self.packed:
            output = convrot_projection_mm(quantized, self.packed_weight, input_scale, self.packed_scale, self.packed_bias, self.infer_dtype)
            return output.split(self.output_dims, dim=-1)
        return tuple(convrot_projection_mm(quantized, weight, input_scale, scale, bias, self.infer_dtype) for weight, scale, bias in zip(self.weights, self.scales, self.biases))


class FP8ProjectionGroup:
    """Resident-only FP8-SGL projections, optionally combined into one GEMM.

    Packed outputs are views of one allocation. Consumers must respect their
    row stride instead of copying them back to contiguous tensors. Original
    modules remain available for adapters, reloaded weights, and fallback.
    """

    @classmethod
    def from_modules(cls, modules, packed=False):
        modules = tuple(modules)
        if sgl_kernel is None or not modules or any(type(module) is not MMWeightWfp8channelAfp8channeldynamicSgl for module in modules):
            return None
        first = modules[0]
        first_weight = getattr(first, "weight", None)
        if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2 or first_weight.device.type != "cuda":
            return None
        device, input_dim, infer_dtype = first_weight.device, first_weight.shape[0], first.infer_dtype
        if infer_dtype not in (torch.bfloat16, torch.float16):
            return None
        has_bias = getattr(first, "bias", None) is not None
        for module in modules:
            if module.has_lora_branch or module.has_diff or hasattr(module, "weight_diff") or hasattr(module, "bias_diff"):
                return None
            if module.lazy_load or module.create_cuda_buffer or module.create_cpu_buffer:
                return None
            weight, scale, bias = getattr(module, "weight", None), getattr(module, "weight_scale", None), getattr(module, "bias", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.device != device or weight.dtype != torch.float8_e4m3fn:
                return None
            if weight.shape[0] != input_dim or weight.stride() != (1, input_dim) or module.infer_dtype != infer_dtype:
                return None
            output_dim = weight.shape[1]
            if not isinstance(scale, torch.Tensor) or scale.device != device or scale.dtype != torch.float32:
                return None
            if scale.ndim not in (1, 2) or scale.numel() != output_dim or not scale.is_contiguous():
                return None
            if (bias is not None) != has_bias:
                return None
            if bias is not None and (not isinstance(bias, torch.Tensor) or bias.shape != (output_dim,) or bias.device != device or bias.dtype != infer_dtype or not bias.is_contiguous()):
                return None
        return cls(modules, packed=packed)

    def __init__(self, modules, packed=False):
        self.modules = tuple(modules)
        self.weights = tuple(module.weight for module in modules)
        self.scales = tuple(module.weight_scale for module in modules)
        self.biases = tuple(getattr(module, "bias", None) for module in modules)
        self.infer_dtype = modules[0].infer_dtype
        self.device = self.weights[0].device
        self.input_dim = self.weights[0].shape[0]
        self.output_dims = tuple(weight.shape[1] for weight in self.weights)
        self.packed = packed
        if packed:
            # FP8 cat is not available on every supported PyTorch version.
            # Reinterpret bytes to preserve the original FP8 encodings exactly.
            stored_weights = [weight.t().view(torch.uint8) for weight in self.weights]
            self.packed_weight = torch.cat(stored_weights, dim=0).view(torch.float8_e4m3fn).t()
            self.packed_scale = torch.cat([scale.reshape(-1, 1) for scale in self.scales], dim=0)
            self.packed_bias = torch.cat(self.biases, dim=0) if self.biases[0] is not None else None
            weights = self.packed_weight.split(self.output_dims, dim=1)
            scales = self.packed_scale.split(self.output_dims, dim=0)
            biases = self.packed_bias.split(self.output_dims, dim=0) if self.packed_bias is not None else (None,) * len(self.modules)
            for module, weight, scale, bias, original_scale in zip(self.modules, weights, scales, biases, self.scales):
                module.weight = weight
                module.weight_scale = scale.view(original_scale.shape)
                module.bias = bias
            # Keep the original module fallbacks, but let them share the packed
            # storage. Releasing the previous tuples avoids a second weight copy.
            self.weights = tuple(module.weight for module in self.modules)
            self.scales = tuple(module.weight_scale for module in self.modules)
            self.biases = tuple(module.bias for module in self.modules)

    def apply(self, x):
        # Identity checks prevent using a stale packed copy after model reloads.
        # Dynamo guards these Python conditions and retraces if adapters change.
        for module, weight, scale, bias in zip(self.modules, self.weights, self.scales, self.biases):
            if (
                module.has_lora_branch
                or module.has_diff
                or hasattr(module, "weight_diff")
                or hasattr(module, "bias_diff")
                or module.weight is not weight
                or module.weight_scale is not scale
                or getattr(module, "bias", None) is not bias
                or module.infer_dtype != self.infer_dtype
                or module.lazy_load
                or module.create_cuda_buffer
                or module.create_cpu_buffer
            ):
                return tuple(module.apply(x) for module in self.modules)
        if x.ndim != 2 or x.device != self.device or x.dtype != self.infer_dtype or x.shape[1] != self.input_dim or not x.is_contiguous():
            return tuple(module.apply(x) for module in self.modules)
        quantized, input_scale = sgl_fp8_per_token_quant(x)
        if self.packed:
            output = sgl_fp8_scaled_mm(quantized, self.packed_weight, input_scale, self.packed_scale, self.infer_dtype, self.packed_bias)
            return output.split(self.output_dims, dim=-1)
        return tuple(sgl_fp8_scaled_mm(quantized, weight, input_scale, scale, self.infer_dtype, bias) for weight, scale, bias in zip(self.weights, self.scales, self.biases))
