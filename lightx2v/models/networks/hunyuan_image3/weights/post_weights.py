from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.hunyuan_image3.weights.common import (
    HunyuanImage3TimestepEmbedderWeights,
    HunyuanImage3UNetUpWeights,
)
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class HunyuanImage3PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module("time_embed_2", HunyuanImage3TimestepEmbedderWeights("time_embed_2"))
        self.add_module(
            "final_norm",
            RMS_WEIGHT_REGISTER[config.get("rms_norm_type", "fp32_variance")](
                "model.ln_f.weight",
                eps=config.get("rms_norm_eps", 1e-5),
            ),
        )
        self.add_module("lm_head", MM_WEIGHT_REGISTER["Default"]("lm_head.weight"))
        self.add_module("final_layer", HunyuanImage3UNetUpWeights("final_layer", config))

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
