from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class WanInfiniteTalkPreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)
        self.add_module(
            "audio_proj_proj1",
            MM_WEIGHT_REGISTER["Default"]("audio_proj.proj1.weight", "audio_proj.proj1.bias"),
        )
        self.add_module(
            "audio_proj_proj1_vf",
            MM_WEIGHT_REGISTER["Default"]("audio_proj.proj1_vf.weight", "audio_proj.proj1_vf.bias"),
        )
        self.add_module(
            "audio_proj_proj2",
            MM_WEIGHT_REGISTER["Default"]("audio_proj.proj2.weight", "audio_proj.proj2.bias"),
        )
        self.add_module(
            "audio_proj_proj3",
            MM_WEIGHT_REGISTER["Default"]("audio_proj.proj3.weight", "audio_proj.proj3.bias"),
        )
        if config.get("norm_output_audio", True):
            self.add_module(
                "audio_proj_norm",
                LN_WEIGHT_REGISTER["torch"]("audio_proj.norm.weight", "audio_proj.norm.bias"),
            )
