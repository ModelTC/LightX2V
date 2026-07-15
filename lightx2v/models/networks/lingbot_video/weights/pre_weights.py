from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class LingBotVideoPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("patch_embedder", MM_WEIGHT_REGISTER["Default"]("patch_embedder.weight", "patch_embedder.bias"))
        self.add_module("text_norm", RMS_WEIGHT_REGISTER["fp32_variance"]("text_embedder.norm.weight"))
        self.add_module("text_linear_1", MM_WEIGHT_REGISTER["Default"]("text_embedder.linear_1.weight", "text_embedder.linear_1.bias"))
        self.add_module("text_linear_2", MM_WEIGHT_REGISTER["Default"]("text_embedder.linear_2.weight", "text_embedder.linear_2.bias"))
        self.add_module("time_linear_1", MM_WEIGHT_REGISTER["Default-ForceFp32"]("time_embedder.linear_1.weight", "time_embedder.linear_1.bias"))
        self.add_module("time_linear_2", MM_WEIGHT_REGISTER["Default-ForceFp32"]("time_embedder.linear_2.weight", "time_embedder.linear_2.bias"))
        self.add_module("time_modulation", MM_WEIGHT_REGISTER["Default-ForceFp32"]("time_modulation.1.weight", "time_modulation.1.bias"))
