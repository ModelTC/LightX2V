"""Share FP8 input quantization across resident projections of one tensor."""

import torch

from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl
from lightx2v.common.ops.mm.sgl_kernel import sgl_fp8_per_token_quant, sgl_fp8_scaled_mm


class FP8ProjectionGroup:
    """Resident-only FP8-SGL projections, optionally combined into one GEMM.

    Packed outputs are views of one allocation. Consumers must respect their
    row stride instead of copying them back to contiguous tensors. Original
    modules remain available for adapters, reloaded weights, and fallback.
    """

    @classmethod
    def from_modules(cls, modules, packed=False):
        modules = tuple(modules)
        if not modules or any(type(module) is not MMWeightWfp8channelAfp8channeldynamicSgl for module in modules):
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
            ):
                return tuple(module.apply(x) for module in self.modules)
        if x.ndim != 2 or x.device != self.device or x.shape[1] != self.input_dim:
            return tuple(module.apply(x) for module in self.modules)
        quantized, input_scale = sgl_fp8_per_token_quant(x)
        if self.packed:
            output = sgl_fp8_scaled_mm(quantized, self.packed_weight, input_scale, self.packed_scale, self.infer_dtype, self.packed_bias)
            return output.split(self.output_dims, dim=-1)
        return tuple(sgl_fp8_scaled_mm(quantized, weight, input_scale, scale, self.infer_dtype, bias) for weight, scale, bias in zip(self.weights, self.scales, self.biases))
