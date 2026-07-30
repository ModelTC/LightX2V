from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class ErnieImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.layer_norm_type = config.get("layer_norm_type", "torch")
        self.add_module("final_norm", LN_WEIGHT_REGISTER[self.layer_norm_type](eps=config.get("eps", 1e-6)))
        self.add_module(
            "final_norm_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "final_norm.linear.weight",
                "final_norm.linear.bias",
            ),
        )
        self.add_module(
            "final_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "final_linear.weight",
                "final_linear.bias",
            ),
        )
