from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class HunyuanPostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.add_module("final_layer_linear", MM_WEIGHT_REGISTER["Default-Force-FP32"]("final_layer.linear.weight", "final_layer.linear.bias"))
        self.add_module("final_layer_adaLN_modulation_1", MM_WEIGHT_REGISTER["Default"]("final_layer.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.bias"))
