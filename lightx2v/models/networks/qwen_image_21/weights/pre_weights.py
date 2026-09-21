import torch

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeight
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class ZeroCenteredRMSWeight(RMSWeight):
    """Qwen-Image-2.1 stores RMSNorm's scale minus one."""

    def apply(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (self.weight.float() + 1)).to(x.dtype)


class QwenImage21PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        for name, key in {
            "img_in": "img_in.weight",
            "txt_in": "txt_in.in_layer.weight",
            "txt_out": "txt_in.out_layer.weight",
            "time_in": "time_text_embed.timestep_embedder.linear_1.weight",
            "time_out": "time_text_embed.timestep_embedder.linear_2.weight",
            "modulation": "modulation.1.weight",
        }.items():
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](key, bias_name=None))
        self.add_module("txt_norm", ZeroCenteredRMSWeight("txt_in.text_norm.weight", eps=config["eps"]))
