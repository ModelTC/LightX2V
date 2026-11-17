from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.pre_weights import WanPreWeights
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, TENSOR_REGISTER


class TBDubAudioProjectionLayer(WeightModule):
    """Checkpoint-native HuBERT projection block (attention followed by GEGLU)."""

    def __init__(self, config, index):
        super().__init__()
        prefix = f"audio_embedding.layers.{index}"
        for name in ("norm1", "norm2"):
            self.add_module(name, LN_WEIGHT_REGISTER[config.get("layer_norm_type", "torch")](f"{prefix}.{name}.weight", f"{prefix}.{name}.bias", eps=1e-5))
        for name in ("q", "k", "v"):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.attn.to_{name}.weight"))
        for name, suffix in (("out", "attn.to_out.0"), ("ff_in", "ff.net.0.proj"), ("ff_out", "ff.net.2")):
            self.add_module(name, MM_WEIGHT_REGISTER["Default"](f"{prefix}.{suffix}.weight", f"{prefix}.{suffix}.bias"))
        self.add_module("position", TENSOR_REGISTER["Default"](f"{prefix}.pos_embed.pe"))
        self.add_module("attn", ATTN_WEIGHT_REGISTER[config["cross_attn_1_type"]]())


class TBDubPreWeights(WanPreWeights):
    def __init__(self, config):
        super().__init__(config)
        for name in ("proj_in", "proj_out"):
            self.add_module(f"audio_{name}", MM_WEIGHT_REGISTER["Default"](f"audio_embedding.{name}.weight", f"audio_embedding.{name}.bias"))
        self.add_module("audio_layers", WeightModuleList([TBDubAudioProjectionLayer(config, index) for index in range(config.get("audio_projection_layers", 4))]))
