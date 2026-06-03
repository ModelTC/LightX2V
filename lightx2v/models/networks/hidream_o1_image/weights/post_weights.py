from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class HidreamO1ImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("mm_type", config.get("dit_quant_scheme", "Default"))
        self.add_module("final_linear", MM_WEIGHT_REGISTER[self.mm_type]("model.final_layer2.linear.weight", "model.final_layer2.linear.bias"))
