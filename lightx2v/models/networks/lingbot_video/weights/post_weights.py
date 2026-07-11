from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class LingBotVideoPostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("norm_out", LN_WEIGHT_REGISTER["torch"](eps=config.get("norm_eps", 1e-6)))
        self.add_module("norm_out_modulation", MM_WEIGHT_REGISTER["Default-ForceFp32"]("norm_out_modulation.1.weight", "norm_out_modulation.1.bias"))
        self.add_module("proj_out", MM_WEIGHT_REGISTER["Default"]("proj_out.weight", "proj_out.bias"))
