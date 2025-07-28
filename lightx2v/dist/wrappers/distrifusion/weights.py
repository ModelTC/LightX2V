import torch.distributed as dist
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)


class DistriFusionWanTransformerWeightsWrapper:
    def __init__(self, transformer_weights: WanTransformerWeights, config):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.transformer_weights = transformer_weights
        self.config = config
        
    def __getattr__(self, name: str):
        if name in self.__dict__:
            return getattr(self, name)
        else:
            return getattr(self.transformer_weights, name)
