from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import EMBEDDING_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class Cosmos3PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module("embed_tokens", EMBEDDING_WEIGHT_REGISTER["Default"]("embed_tokens.weight"))
        self.add_module("proj_in", MM_WEIGHT_REGISTER["Default"]("proj_in.weight", "proj_in.bias"))
        self.add_module(
            "time_embedder_linear_1",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.linear_1.weight", "time_embedder.linear_1.bias"),
        )
        self.add_module(
            "time_embedder_linear_2",
            MM_WEIGHT_REGISTER["Default"]("time_embedder.linear_2.weight", "time_embedder.linear_2.bias"),
        )
