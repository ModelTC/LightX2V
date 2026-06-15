from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import CONV2D_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class ErnieImagePreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.add_module(
            "x_embedder",
            CONV2D_WEIGHT_REGISTER["Default"](
                "x_embedder.proj.weight",
                "x_embedder.proj.bias",
                stride=config.get("patch_size", 1),
                padding=0,
            ),
        )
        if config.get("text_in_dim") != config.get("hidden_size"):
            self.add_module(
                "text_proj",
                MM_WEIGHT_REGISTER[self.mm_type](
                    "text_proj.weight",
                    None,
                ),
            )
        else:
            self.text_proj = None
        self.add_module(
            "time_embedding_linear_1",
            MM_WEIGHT_REGISTER[self.mm_type](
                "time_embedding.linear_1.weight",
                "time_embedding.linear_1.bias",
            ),
        )
        self.add_module(
            "time_embedding_linear_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                "time_embedding.linear_2.weight",
                "time_embedding.linear_2.bias",
            ),
        )
        self.add_module(
            "adaln_modulation",
            MM_WEIGHT_REGISTER[self.mm_type](
                "adaLN_modulation.1.weight",
                "adaLN_modulation.1.bias",
            ),
        )
