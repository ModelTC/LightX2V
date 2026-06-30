from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.cosmos3.weights.pre_weights import Cosmos3DomainAwareLinearWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class Cosmos3PostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        eps = config.get("rms_norm_eps", 1e-6)
        rms_norm_type = config.get("rms_norm_type", "one-pass")
        self.add_module("norm", RMS_WEIGHT_REGISTER[rms_norm_type]("norm.weight", eps=eps))
        self.add_module("norm_moe_gen", RMS_WEIGHT_REGISTER[rms_norm_type]("norm_moe_gen.weight", eps=eps))
        self.add_module("proj_out", MM_WEIGHT_REGISTER["Default"]("proj_out.weight", "proj_out.bias"))
        if config.get("sound_gen", False):
            self.add_module("audio_proj_out", MM_WEIGHT_REGISTER["Default"]("audio_proj_out.weight", "audio_proj_out.bias"))
        if config.get("action_gen", False):
            self.add_module(
                "action_proj_out",
                Cosmos3DomainAwareLinearWeights(
                    "action_proj_out",
                    input_size=config["hidden_size"],
                    output_size=config.get("action_dim", config.get("max_action_dim", 64)),
                ),
            )
