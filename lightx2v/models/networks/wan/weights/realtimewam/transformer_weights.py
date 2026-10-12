import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.wan.weights.fastwam.transformer_weights import FastWAMBlockWeights, FastWAMFFNWeights, FastWAMSelfAttentionWeights, FastWAMTransformerWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER, TENSOR_REGISTER


@torch.no_grad()
def replace_linears_with_fp8(weights):
    """Quantize loaded block linears once, after BF16 QKV/KV packing."""

    def quantize(linear):
        if isinstance(linear, MM_WEIGHT_REGISTER["fp8-sgl"]):
            return linear
        quant = MM_WEIGHT_REGISTER["fp8-sgl"](linear.weight_name, linear.bias_name)
        quant.set_config({"weight_auto_quant": True})
        # Loaded MMWeight stores [in, out]; weight quantization expects [out, in].
        quant.load_func({linear.weight_name: linear._get_actual_weight().detach().t().contiguous()})
        quant.bias = linear._get_actual_bias()
        quant.post_process()
        return quant

    for expert in (weights.video, weights.action):
        for block in expert.blocks:
            block.self_attn.qkv.linear = quantize(block.self_attn.qkv.linear)
            targets = [(block.self_attn, "o"), (block.ffn, "fc0"), (block.ffn, "fc2")]
            if hasattr(block, "cross_attn"):
                block.cross_attn.kv.linear = quantize(block.cross_attn.kv.linear)
                targets.extend([(block.cross_attn, "q"), (block.cross_attn, "o")])
            for owner, name in targets:
                owner.add_module(name, quantize(getattr(owner, name)))


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
    """Pack Linear weights once, release originals, and return output views."""

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
        for module in modules:
            module.weight = module.bias = None

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
        # Keep module registrations; release their weights after packing.
        for expert in (self.video, self.action):
            for block in expert.blocks:
                attn = block.self_attn
                attn.qkv = RealtimeWAMPackedProjection((attn.q, attn.k, attn.v))
                if hasattr(block, "cross_attn"):
                    attn = block.cross_attn
                    attn.kv = RealtimeWAMPackedProjection((attn.k, attn.v))
