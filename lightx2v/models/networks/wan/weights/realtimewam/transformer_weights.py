from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.fastwam.transformer_weights import (
    FastWAMBlockWeights,
    FastWAMFFNWeights,
    FastWAMSelfAttentionWeights,
)
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


def condition_layers(config):
    layers = config.get("condition_layers")
    layers = list(range(int(config["num_layers"]))) if layers is None else list(layers)
    if not layers or layers != sorted(set(layers)) or any(type(i) is not int or not 0 <= i < config["num_layers"] for i in layers):
        raise ValueError("condition_layers must be nonempty, sorted, unique layer indices")
    return layers


class ActionOnlyWeights(WeightModule):
    def __init__(self, prefix, index, config):
        super().__init__()
        self.add_module("self_attn", FastWAMSelfAttentionWeights(prefix, index, config))
        self.add_module("ffn", FastWAMFFNWeights(prefix, index, config))


class RealtimeWAMTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        conditions = condition_layers(config)
        for name in ("video", "action"):
            expert = WeightModule()
            blocks = []
            for i in range(config["num_layers"]):
                cls = FastWAMBlockWeights if name == "video" or i in conditions else ActionOnlyWeights
                blocks.append(cls(f"mixtures.{name}", i, config))
            expert.add_module("blocks", WeightModuleList(blocks))
            self.add_module(name, expert)
        self.add_module("action_head", MM_WEIGHT_REGISTER["Default"]("mixtures.action.head.weight", "mixtures.action.head.bias"))
        if config.get("video_kv_fusion") == "interval_weighted_sum":
            self.add_module("fusion_logits", WeightModuleList([TENSOR_REGISTER["Default"](f"video_kv_fusion_logits.{i}") for i in range(len(conditions))]))
