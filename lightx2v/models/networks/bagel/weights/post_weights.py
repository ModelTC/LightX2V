from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class Qwen2PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module(
            "norm",
            RMS_WEIGHT_REGISTER["fp32_variance"]("language_model.model.norm.weight"),
        )
        self.add_module(
            "norm_moe_gen",
            RMS_WEIGHT_REGISTER["fp32_variance"]("language_model.model.norm_moe_gen.weight"),
        )
        # llm2vae
        self.add_module(
            "llm2vae",
            MM_WEIGHT_REGISTER["Default"]("llm2vae.weight", "llm2vae.bias"),
        )
