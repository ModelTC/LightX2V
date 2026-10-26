"""Backend-aware projection sharing and precision-preserving GELU/linear fusion.

These factories consume loaded linear modules, without model configuration or
model-specific tensor names. Rebuild the wrappers when loading model weights.
Callers choose whether their compile/offload/parallel execution permits fusion.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.mm.fp8_f16_accum import cutlass_scaled_fp8_mm_f16_accum
from lightx2v.common.ops.mm.fused_gelu_linear import fused_gelu_int8_convrot_linear, fused_gelu_quant_fp8_f16_accum, fused_gelu_quant_fp8_sgl, prepare_convrot_gelu_fusion
from lightx2v.common.ops.mm.fused_swiglu_linear import fused_swiglu_quant_fp8_f16_accum
from lightx2v.common.ops.mm.mm_weight import MMWeightTP, MMWeightWfp8channelAfp8channelF16Accum, MMWeightWfp8channelAfp8channeldynamicSgl, MMWeightWint8ConvRot
from lightx2v.common.ops.mm.projection_group import ConvRotProjectionGroup, FP8F16AccumProjectionGroup, FP8ProjectionGroup, _backend_apis_available
from lightx2v.common.ops.mm.sgl_kernel import sgl_fp8_scaled_mm


def make_projection_group(modules, *, packed=False, offload_buffers=False):
    """Share one input quantization, optionally packing projection GEMMs.

    Return ``None`` for unsupported backend/module layouts. A created group
    returns a tuple of projections from ``.apply(x)`` and falls back to the
    original linear modules if its input or module state becomes incompatible.
    """
    modules = tuple(modules)
    if not modules:
        return None
    if type(modules[0]) is MMWeightTP:
        first = modules[0]
        if any(
            type(module) is not MMWeightTP or module.split_dim != "col" or module.tp_group is not first.tp_group or module.tp_rank != first.tp_rank or module.tp_size != first.tp_size
            for module in modules
        ):
            return None
        return make_projection_group((module._mm for module in modules), packed=packed, offload_buffers=offload_buffers)
    if type(modules[0]) is MMWeightWfp8channelAfp8channelF16Accum:
        return FP8F16AccumProjectionGroup.from_modules(modules, packed=packed, offload_buffers=offload_buffers)
    if offload_buffers:
        return None
    if type(modules[0]) is MMWeightWfp8channelAfp8channeldynamicSgl:
        return FP8ProjectionGroup.from_modules(modules, packed=packed)
    if type(modules[0]) is MMWeightWint8ConvRot:
        return ConvRotProjectionGroup.from_modules(modules, packed=packed)
    return None


def apply_swiglu_linear(module, value, gate):
    """Fuse H3 SwiGLU with FP8-F16 quantization for an active local weight.

    The weight is read at each call, so block-offloaded weights and captured
    graph storage use the current CUDA buffer. Other backends and adapters
    retain the original activation and linear path.
    """
    tp = module if type(module) is MMWeightTP else None
    linear = tp._mm if tp is not None else module
    weight = getattr(linear, "weight", None)
    scale = getattr(linear, "weight_scale", None)
    qmax = getattr(linear, "fp8_activation_qmax", None)
    if (
        type(linear) is not MMWeightWfp8channelAfp8channelF16Accum
        or qmax is None
        or linear.has_lora_branch
        or linear.has_diff
        or hasattr(linear, "weight_diff")
        or hasattr(linear, "bias_diff")
        or not isinstance(weight, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or value.ndim != 2
        or value.shape != gate.shape
        or value.dtype != linear.infer_dtype
        or gate.dtype != value.dtype
        or value.device != weight.device
        or gate.device != value.device
        or value.stride(1) != 1
        or gate.stride(1) != 1
        or value.shape[1] != weight.shape[0]
        or (tp is not None and tp.split_dim != "row")
    ):
        return module.apply(value * F.silu(gate))
    quantized, input_scale = fused_swiglu_quant_fp8_f16_accum(value, gate, qmax)
    output = cutlass_scaled_fp8_mm_f16_accum(quantized, weight, input_scale, scale, linear.infer_dtype, linear._get_actual_bias())
    if tp is not None and tp.reduce_output and tp.tp_size > 1 and tp.tp_group is not None:
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp.tp_group)
        if tp._row_split_bias is not None:
            output = output + tp._row_split_bias
    return output


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

    Fusion retains the activation dtype's GELU boundary and the backend's
    original quantizer. Adapters and reloaded/offloaded weights use ``module.apply``.
    """

    def __init__(self, module, fusion=None):
        self.module = module
        self.fusion = fusion
        self.weight = getattr(module, "weight", None)
        self.scale = getattr(module, "weight_scale", None)
        self.bias = getattr(module, "bias", None)
        self.infer_dtype = getattr(module, "infer_dtype", None)
        self.activation_qmax = getattr(module, "fp8_activation_qmax", None)
        if fusion is not None:
            self.device = self.weight.device
            self.input_dim = self.weight.shape[1 if fusion == "int8-convrot" else 0]

    def apply(self, hidden):
        module = self.module
        if (
            self.fusion is None
            or not _plain_resident(module)
            or module.weight is not self.weight
            or module.weight_scale is not self.scale
            or getattr(module, "bias", None) is not self.bias
            or module.infer_dtype != self.infer_dtype
            or getattr(module, "fp8_activation_qmax", None) != self.activation_qmax
            or hidden.ndim != 2
            or hidden.dtype != (self.infer_dtype if self.fusion == "fp8-f16-accum" else torch.bfloat16)
            or hidden.device != self.device
            or hidden.shape[1] != self.input_dim
        ):
            return module.apply(F.gelu(hidden, approximate="tanh"))
        if self.fusion == "fp8-sgl":
            quantized, scale = fused_gelu_quant_fp8_sgl(hidden)
            return sgl_fp8_scaled_mm(quantized, self.weight, scale, self.scale, self.infer_dtype, self.bias)
        if self.fusion == "fp8-f16-accum":
            quantized, scale = fused_gelu_quant_fp8_f16_accum(hidden, self.activation_qmax)
            return cutlass_scaled_fp8_mm_f16_accum(quantized, self.weight, scale, self.scale, self.infer_dtype, self.bias)
        if hidden.shape[0] < 3 or module._convrot_groupsize != 256:
            return module.apply(F.gelu(hidden, approximate="tanh"))
        return fused_gelu_int8_convrot_linear(hidden, self.weight, self.scale, self.bias, self.infer_dtype, module._convrot_groupsize)


class OffloadedF16GeluLinear:
    """Fuse GELU into the active FP8-F16 upload slot without retaining a block."""

    def __init__(self, module, *, bias_after_reduce=False):
        self.module = module
        self.weight = module.weight_cuda_buffer
        self.scale = module.weight_scale_cuda_buffer
        self.bias = None if bias_after_reduce else getattr(module, "bias_cuda_buffer", None)
        self.infer_dtype = module.infer_dtype
        self.activation_qmax = module.fp8_activation_qmax

    def apply(self, hidden):
        module = self.module
        if (
            module.weight is not self.weight
            or module.weight_scale is not self.scale
            or getattr(module, "bias", None) is not self.bias
            or module.fp8_activation_qmax != self.activation_qmax
            or module.infer_dtype != self.infer_dtype
            or module.has_lora_branch
            or module.has_diff
            or hasattr(module, "weight_diff")
            or hasattr(module, "bias_diff")
            or hidden.ndim != 2
            or hidden.shape[1] != self.weight.shape[0]
            or hidden.dtype != self.infer_dtype
            or hidden.device != self.weight.device
        ):
            return module.apply(F.gelu(hidden, approximate="tanh"))
        quantized, scale = fused_gelu_quant_fp8_f16_accum(hidden, self.activation_qmax)
        return cutlass_scaled_fp8_mm_f16_accum(quantized, self.weight, scale, self.scale, self.infer_dtype, self.bias)


class TensorParallelGeluLinear:
    """Run the local fused linear before the original row-TP reduction."""

    def __init__(self, module, local_linear):
        self.module = module
        self.local_linear = local_linear

    def apply(self, hidden):
        output = self.local_linear.apply(hidden)
        module = self.module
        if module.split_dim == "row" and module.reduce_output and module.tp_size > 1 and module.tp_group is not None:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=module.tp_group)
            if module._row_split_bias is not None:
                output = output + module._row_split_bias
        return output


def make_gelu_linear(module, *, enabled=True, offload_buffers=False):
    """Prepare a reusable GELU/linear wrapper before compilation or capture.

    ``enabled`` must reflect the caller's precision/compile contract: fused
    execution matches compiled GELU with ``emulate_precision_casts=True``.
    Unsupported cases always return a wrapper executing the original modules.
    """
    if type(module) is MMWeightTP:
        local = make_gelu_linear(module._mm, enabled=enabled, offload_buffers=offload_buffers)
        if module.split_dim == "row" and isinstance(local, OffloadedF16GeluLinear):
            # Row TP removes the local bias after each staging upload. The
            # outer wrapper adds it once after all-reduce; retaining the upload
            # buffer here would permanently fail the fusion identity guard.
            local = OffloadedF16GeluLinear(module._mm, bias_after_reduce=True)
        return TensorParallelGeluLinear(module, local)
    if offload_buffers:
        if enabled and type(module) is MMWeightWfp8channelAfp8channelF16Accum:
            group = FP8F16AccumProjectionGroup.from_modules((module,), offload_buffers=True)
            if group is not None and group.input_dim <= 16384:
                return OffloadedF16GeluLinear(module)
        return GeluLinear(module)
    if not enabled or not _plain_resident(module):
        return GeluLinear(module)
    if type(module) is MMWeightWfp8channelAfp8channelF16Accum:
        group = FP8F16AccumProjectionGroup.from_modules((module,))
        if group is not None and 0 < group.input_dim <= 16384 and module.infer_dtype in (torch.bfloat16, torch.float16):
            return GeluLinear(module, "fp8-f16-accum")
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
