from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


def _rms(config, name, eps):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](name, eps=eps)


class MiniMaxH3PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.add_module(
            "norm_out",
            _rms(config, "norm_out.norm.weight", eps=float(config.get("final_norm_eps", 1e-5))),
        )
        self.add_module(
            "norm_out_linear",
            MM_WEIGHT_REGISTER["Default"]("norm_out.linear.weight", "norm_out.linear.bias"),
        )
        self.add_module(
            "proj_out",
            MM_WEIGHT_REGISTER["Default-ForceFp32"]("proj_out.weight", "proj_out.bias"),
        )
        self.add_module(
            "audio_proj_out",
            MM_WEIGHT_REGISTER["Default-ForceFp32"]("audio_proj_out.weight", "audio_proj_out.bias"),
        )
