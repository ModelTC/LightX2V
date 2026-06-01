from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class Hunyuan3DPostWeights(WeightModule):
    """Post-processing weights for Hunyuan3D shape DiT."""

    def __init__(self, config):
        super().__init__()
        mm_type = config.get("dit_quant_scheme", "Default")
        ln_type = config.get("ln_norm_type", "torch")

        self.add_module(
            "norm_final",
            LN_WEIGHT_REGISTER[ln_type](
                "final_layer.norm_final.weight",
                "final_layer.norm_final.bias",
                eps=1e-6,
            ),
        )
        self.add_module(
            "linear",
            MM_WEIGHT_REGISTER[mm_type](
                "final_layer.linear.weight",
                "final_layer.linear.bias",
            ),
        )
