import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.fastwam.transformer_weights import FastWAMBlockWeights, FastWAMFFNWeights, FastWAMSelfAttentionWeights, FastWAMTransformerWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


def condition_layers(config):
    interval = config.get("action_condition_interval") if config.get("kv_fusion", False) else 1
    if type(interval) is not int or interval <= 0:
        raise ValueError("action_condition_interval must be a positive integer when kv_fusion is enabled")
    return tuple(range(0, int(config["num_layers"]), interval))


class FasterWAMActionOnlyWeights(WeightModule):
    def __init__(self, prefix, index, config, *, rope_config_key):
        super().__init__()
        self.add_module("self_attn", FastWAMSelfAttentionWeights(prefix, index, config, rope_config_key=rope_config_key))
        self.add_module("ffn", FastWAMFFNWeights(prefix, index, config))


class RealtimeWAMPackedProjection:
    """Pack loaded Linear weights once, then return views of the combined output."""

    def __init__(self, modules):
        if any(module.has_lora_branch for module in modules):
            raise ValueError("RealtimeWAM packed projections require merged LoRA weights.")
        weights = [module._get_actual_weight().detach().t() for module in modules]
        self.widths = tuple(weight.shape[0] for weight in weights)
        biases = [module._get_actual_bias() for module in modules]
        bias = None if all(b is None for b in biases) else torch.cat([w.new_zeros(w.shape[0]) if b is None else b.detach() for w, b in zip(weights, biases)])
        self.linear = MM_WEIGHT_REGISTER["Default"]("realtimewam.packed.weight", "realtimewam.packed.bias" if bias is not None else None)
        self.linear.weight = torch.cat(weights).contiguous().t()
        self.linear.bias = bias

    def apply(self, x):
        return self.linear.apply(x).split(self.widths, dim=-1)


class RealtimeWAMTransformerWeights(FastWAMTransformerWeights):
    rope_config_key = "realtimewam_rope_type"

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path, lora_path)
        if config.get("kv_fusion", False):
            conditions = condition_layers(config)
            self.add_module("fusion_logits", WeightModuleList([TENSOR_REGISTER["Default"](f"video_kv_fusion_logits.{i}") for i in range(len(conditions))]))

    def _build_action_weights(self, config):
        if not config.get("kv_fusion", False):
            return super()._build_action_weights(config)
        conditions = condition_layers(config)
        action = WeightModule()
        blocks = []
        for index in range(int(config["num_layers"])):
            cls = FastWAMBlockWeights if index in conditions else FasterWAMActionOnlyWeights
            blocks.append(cls("mixtures.action", index, config, rope_config_key=self.rope_config_key))
        action.add_module("blocks", WeightModuleList(blocks))
        return action

    def pack_projections(self):
        # Derived caches: retain the original registered checkpoint modules.
        for expert in (self.video, self.action):
            for block in expert.blocks:
                attn = block.self_attn
                attn.qkv = RealtimeWAMPackedProjection((attn.q, attn.k, attn.v))
                if hasattr(block, "cross_attn"):
                    attn = block.cross_attn
                    attn.kv = RealtimeWAMPackedProjection((attn.k, attn.v))
