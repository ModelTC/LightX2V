from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER, TENSOR_REGISTER

from ._shared import SequentialLinearWeights, load_prefixed_submodules, projector_layer_prefixes
from .pre_weights import build_motus_expert_configs


class MotusActionPostWeights(WeightModule):
    def __init__(self, config, action_config):
        super().__init__()
        self.action_config = action_config
        self.add_module("norm", LN_WEIGHT_REGISTER["torch"]())
        self.add_module(
            "action_head",
            SequentialLinearWeights(
                projector_layer_prefixes("mlp1x_silu", "decoder.action_head"),
                "silu",
                config,
            ),
        )
        self.register_parameter("modulation", TENSOR_REGISTER["Default"]("decoder.modulation"))

    def apply_output(self, action_tokens, time_emb):
        shift, scale = (self.modulation.tensor.unsqueeze(0) + time_emb.unsqueeze(2)).chunk(2, dim=2)
        hidden = self.norm.apply(action_tokens)
        hidden = hidden * (1 + scale.squeeze(2)) + shift.squeeze(2)
        return self.action_head.apply(hidden)


class MotusPostWeights(WeightModule):
    weight_prefixes = {
        "action": "action_expert.",
    }

    def __init__(self, config):
        super().__init__()
        expert_configs = build_motus_expert_configs(config)
        self.add_module("action", MotusActionPostWeights(config, expert_configs.action))

    def load(self, weight_dict):
        load_prefixed_submodules(self, weight_dict, self.weight_prefixes)
