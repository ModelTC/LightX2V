"""Experiment with per-input NVFP4 scaling on the existing SM120 GEMM."""
from lightx2v.common.ops.mm.mm_weight import (
    MMWeightWnvfp4Anvfp4dynamic,
    cutlass_scaled_nvfp4_mm,
    scaled_nvfp4_quant,
)


def quantize(input_tensor):
    scale = 2688.0 / input_tensor.abs().amax().float().clamp_min(1e-12)
    packed, scales = scaled_nvfp4_quant(input_tensor, scale)
    return packed, scales, scale


def multiply(module, packed, scales, scale):
    return cutlass_scaled_nvfp4_mm(
        packed, module.weight, scales, module.weight_scale,
        alpha=module.alpha / scale, bias=module.bias,
    )


class SharedActivation:
    def __init__(self, modules):
        self.modules = modules

    def apply(self, input_tensor):
        packed, scales, scale = quantize(input_tensor)
        return tuple(multiply(module, packed, scales, scale) for module in self.modules)


def install(*, share_projection=False):
    def apply(self, input_tensor):
        output = multiply(self, *quantize(input_tensor))
        return self.add_lora(output, input_tensor) if self.has_lora_branch else output

    MMWeightWnvfp4Anvfp4dynamic.apply = apply
    if share_projection:
        from lightx2v.models.networks.wan import model

        original = model.make_projection_group

        def make_group(modules, *, packed=False, offload_buffers=False):
            modules = tuple(modules)
            if not offload_buffers and modules and all(
                type(module) is MMWeightWnvfp4Anvfp4dynamic and not module.has_lora_branch
                and module.weight is not None and module.weight.is_cuda for module in modules
            ):
                return SharedActivation(modules)
            return original(modules, packed=packed, offload_buffers=offload_buffers)

        model.make_projection_group = make_group
