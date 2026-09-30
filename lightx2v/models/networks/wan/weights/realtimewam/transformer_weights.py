import torch

from lightx2v.models.networks.wan.weights.fastwam.transformer_weights import FastWAMTransformerWeights
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


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

    def pack_projections(self):
        # Derived caches: retain the original registered checkpoint modules.
        for expert in (self.video, self.action):
            for block in expert.blocks:
                attn = block.self_attn
                attn.qkv = RealtimeWAMPackedProjection((attn.q, attn.k, attn.v))
                attn = block.cross_attn
                attn.kv = RealtimeWAMPackedProjection((attn.k, attn.v))
