from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.hunyuan_image3.weights.common import (
    HunyuanImage3TimestepEmbedderWeights,
    HunyuanImage3UNetDownWeights,
)
from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER


class HunyuanImage3PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module("token_embedding", EMBEDDING_WEIGHT_REGISTER["Default"]("model.wte.weight"))
        self.add_module("timestep_emb", HunyuanImage3TimestepEmbedderWeights("timestep_emb"))
        self.add_module("time_embed", HunyuanImage3TimestepEmbedderWeights("time_embed"))
        if config.get("cfg_distilled", False):
            self.add_module("guidance_emb", HunyuanImage3TimestepEmbedderWeights("guidance_emb"))
        if config.get("use_meanflow", False):
            self.add_module("timestep_r_emb", HunyuanImage3TimestepEmbedderWeights("timestep_r_emb"))
        self.add_module("patch_embed", HunyuanImage3UNetDownWeights("patch_embed", config))

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
