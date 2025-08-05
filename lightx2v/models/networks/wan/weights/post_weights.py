from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanPostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_parameter(
            "norm",
            LN_WEIGHT_REGISTER["Default"](),
        )
        self.add_module("head", MM_WEIGHT_REGISTER["Default"]("head.head.weight", "head.head.bias"))
        self.register_parameter("head_modulation", TENSOR_REGISTER["Default"]("head.modulation"))
