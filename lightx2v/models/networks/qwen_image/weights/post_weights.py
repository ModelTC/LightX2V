from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class QwenImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.task = config["task"]
        self.config = config
        self.lazy_load = self.config.get("lazy_load", False)
        if self.lazy_load:
            assert NotImplementedError
        self.lazy_load_file = False

        # norm_out
        self.add_module(
            "norm_out_linear",
            MM_WEIGHT_REGISTER["Default"](
                "norm_out.linear.weight",
                "norm_out.linear.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
        self.add_module("norm_out", LN_WEIGHT_REGISTER["Default"](eps=1e-6))

        # proj_out
        self.add_module(
            "proj_out_linear",
            MM_WEIGHT_REGISTER["Default"](
                "proj_out.weight",
                "proj_out.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
