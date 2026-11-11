from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.minimax_h3.weights.pre_weights import MiniMaxH3PreWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class MiniMaxH3ImagePreWeights(MiniMaxH3PreWeights):
    def __init__(self, config):
        super().__init__(config)
        self._modules.pop("audio_proj_in")
        del self.audio_proj_in


class MiniMaxH3ImagePDDHead(WeightModule):
    def __init__(self, index):
        super().__init__()
        self.add_module("proj", MM_WEIGHT_REGISTER["Default-ForceFp32"](f"pdd_proj_out.{index}.weight", f"pdd_proj_out.{index}.bias"))


class MiniMaxH3ImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module("norm_out", RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")]("norm_out.norm.weight", eps=float(config.get("final_norm_eps", 1e-5))))
        if not config.get("use_adaln_cache", False):
            self.add_module("norm_out_linear", MM_WEIGHT_REGISTER["Default"]("norm_out.linear.weight", "norm_out.linear.bias"))
        self.add_module("pdd_heads", WeightModuleList([MiniMaxH3ImagePDDHead(index) for index in range(6)]))
