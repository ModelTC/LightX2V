from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class QwenImage21PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("modulation", MM_WEIGHT_REGISTER["Default"]("norm_out.linear.weight", bias_name=None))
        self.add_module("norm", LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](eps=config["eps"]))
        self.add_module("proj", MM_WEIGHT_REGISTER["Default"]("proj_out.weight", bias_name=None))
