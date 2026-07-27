from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.hunyuan_image3.weights.common import (
    HunyuanImage3TimestepEmbedderWeights,
    HunyuanImage3UNetUpWeights,
    hunyuan_image3_mm_weight,
)
from lightx2v.utils.registry_factory import RMS_WEIGHT_REGISTER


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
        self.add_module(
            "lm_head",
            hunyuan_image3_mm_weight(
                config,
                "Default",
                "lm_head.weight",
                split_dim="col",
            ),
        )
        self.add_module("final_layer", HunyuanImage3UNetUpWeights("final_layer", config))

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
