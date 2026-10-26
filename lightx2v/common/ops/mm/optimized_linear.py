"""Backend-aware projection sharing and precision-preserving GELU/linear fusion.

These factories consume loaded linear modules, without model configuration or
model-specific tensor names. Rebuild the wrappers when loading model weights.
Callers choose whether their compile/offload/parallel execution permits fusion.
"""

import torch
import torch.nn.functional as F

from lightx2v.common.ops.mm.fused_gelu_linear import fused_gelu_int8_convrot_linear, fused_gelu_quant_fp8_sgl, prepare_convrot_gelu_fusion
from lightx2v.common.ops.mm.mm_weight import MMWeightWfp8channelAfp8channeldynamicSgl, MMWeightWint8ConvRot
from lightx2v.common.ops.mm.projection_group import ConvRotProjectionGroup, FP8ProjectionGroup, _backend_apis_available
from lightx2v.common.ops.mm.sgl_kernel import sgl_fp8_scaled_mm


def make_projection_group(modules, *, packed=False):
    """Share one input quantization, optionally packing projection GEMMs.

    Return ``None`` for unsupported backend/module layouts. A created group
    returns a tuple of projections from ``.apply(x)`` and falls back to the
    original linear modules if its input or module state becomes incompatible.
    """
    modules = tuple(modules)
    if not modules:
        return None
    if type(modules[0]) is MMWeightWfp8channelAfp8channeldynamicSgl:
        return FP8ProjectionGroup.from_modules(modules, packed=packed)
    if type(modules[0]) is MMWeightWint8ConvRot:
        return ConvRotProjectionGroup.from_modules(modules, packed=packed)
    return None


def _plain_resident(module):
    return not (
        getattr(module, "has_lora_branch", False)
        or getattr(module, "has_diff", False)
        or hasattr(module, "weight_diff")
        or hasattr(module, "bias_diff")
        or getattr(module, "lazy_load", False)
        or getattr(module, "create_cuda_buffer", False)
        or getattr(module, "create_cpu_buffer", False)
    )


class GeluLinear:
    """GELU(tanh) followed by the original linear, with an optional fused path.

    Fusion retains the BF16 GELU boundary and the backend's original activation
    quantizer. Adapters and reloaded/offloaded weights use ``module.apply``.
    """

    def __init__(self, module, fusion=None):
        self.module = module
        self.fusion = fusion
        self.weight = getattr(module, "weight", None)
        self.scale = getattr(module, "weight_scale", None)
        self.bias = getattr(module, "bias", None)
        self.infer_dtype = getattr(module, "infer_dtype", None)
        if fusion is not None:
            self.device = self.weight.device
            self.input_dim = self.weight.shape[0 if fusion == "fp8-sgl" else 1]

    def apply(self, hidden):
        module = self.module
        if (
            self.fusion is None
            or not _plain_resident(module)
            or module.weight is not self.weight
            or module.weight_scale is not self.scale
            or getattr(module, "bias", None) is not self.bias
            or module.infer_dtype != self.infer_dtype
            or hidden.ndim != 2
            or hidden.dtype != torch.bfloat16
            or hidden.device != self.device
            or hidden.shape[1] != self.input_dim
        ):
            return module.apply(F.gelu(hidden, approximate="tanh"))
        if self.fusion == "fp8-sgl":
            quantized, scale = fused_gelu_quant_fp8_sgl(hidden)
            return sgl_fp8_scaled_mm(quantized, self.weight, scale, self.scale, self.infer_dtype, self.bias)
        if hidden.shape[0] < 3 or module._convrot_groupsize != 256:
            return module.apply(F.gelu(hidden, approximate="tanh"))
        return fused_gelu_int8_convrot_linear(hidden, self.weight, self.scale, self.bias, self.infer_dtype, module._convrot_groupsize)


def make_gelu_linear(module, *, enabled=True):
    """Prepare a reusable GELU/linear wrapper before compilation or capture.

    ``enabled`` must reflect the caller's precision/compile contract: fused
    execution matches compiled GELU with ``emulate_precision_casts=True``.
    Unsupported cases always return a wrapper executing the original modules.
    """
    if not enabled or not _plain_resident(module):
        return GeluLinear(module)
    if type(module) is MMWeightWfp8channelAfp8channeldynamicSgl:
        group = FP8ProjectionGroup.from_modules((module,))
        # Bound a single-CTA reduction's resource use. The quantizer itself is
        # shape-generic, but larger hidden widths retain the original path.
        if group is not None and 0 < group.input_dim <= 16384:
            return GeluLinear(module, "fp8-sgl")
        return GeluLinear(module)
    if type(module) is not MMWeightWint8ConvRot or module.infer_dtype != torch.bfloat16 or getattr(module, "_convrot_groupsize", None) != 256:
        return GeluLinear(module)
    weight, scale, bias = getattr(module, "weight", None), getattr(module, "weight_scale", None), getattr(module, "bias", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.ndim != 2
        or weight.device.type != "cuda"
        or weight.dtype != torch.int8
        or not weight.is_contiguous()
        or weight.shape[1] != 14336
        or weight.shape[0] % 16
        or not isinstance(scale, torch.Tensor)
        or scale.device != weight.device
        or scale.dtype != torch.float32
        or scale.ndim not in (1, 2)
        or scale.numel() not in (1, weight.shape[0])
        or not scale.is_contiguous()
        or (bias is not None and (not isinstance(bias, torch.Tensor) or bias.device != weight.device or bias.dtype != torch.bfloat16 or bias.shape != (weight.shape[0],) or not bias.is_contiguous()))
        or not _backend_apis_available()
        or not prepare_convrot_gelu_fusion(weight.device)
    ):
        return GeluLinear(module)
    return GeluLinear(module, "int8-convrot")
