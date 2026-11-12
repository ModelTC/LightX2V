import torch

from lightx2v_platform.ops.mm.template import MMWeightPerChannelQuantTemplate
from lightx2v_platform.registry_factory import PLATFORM_MM_WEIGHT_REGISTER

from .kernels import metax_int8_linear


@PLATFORM_MM_WEIGHT_REGISTER("int8-metax")
class MMWeightWint8channelAint8channeldynamicMetax(MMWeightPerChannelQuantTemplate):
    """Per-channel INT8 weights and dynamic per-token INT8 activations on MetaX."""

    checkpoint_dtype = torch.int8

    def apply(self, input_tensor):
        return metax_int8_linear(input_tensor, self.weight, self.weight_scale, self.bias)


__all__ = ["MMWeightWint8channelAint8channeldynamicMetax"]
