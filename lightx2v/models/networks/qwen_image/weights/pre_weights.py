from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class QwenImagePreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # img_in
        self.add_module(
            "img_in",
            MM_WEIGHT_REGISTER["Default"]("img_in.weight", "img_in.bias"),
        )
        # txt_in
        self.add_module(
            "txt_in",
            MM_WEIGHT_REGISTER["Default"]("txt_in.weight", "txt_in.bias"),
        )
        # txt_norm
        self.add_module(
            "txt_norm",
            RMS_WEIGHT_REGISTER["fp32_variance"]("txt_norm.weight")
        )
        # time_text_embed
        self.add_module(
            "time_text_embed_timestep_embedder_linear_1",
            MM_WEIGHT_REGISTER["Default"]("time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias")
        )
        self.add_module(
            "time_text_embed_timestep_embedder_linear_2",
            MM_WEIGHT_REGISTER["Default"]("time_text_embed.timestep_embedder.linear_2.weight", "time_text_embed.timestep_embedder.linear_2.bias")
        )
     