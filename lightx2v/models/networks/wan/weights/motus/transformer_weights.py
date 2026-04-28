from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights

from ._shared import MotusJointExpertBlockWeights, MotusJointExpertTransformerWeights, load_prefixed_submodules
from .pre_weights import build_motus_expert_configs


class MotusWanTransformerWeights(WanTransformerWeights):
    pass


class MotusActionBlockWeights(MotusJointExpertBlockWeights):
    def __init__(self, block_idx, config, action_config, wan_num_heads, wan_head_dim):
        super().__init__(
            block_idx=block_idx,
            config=config,
            expert_dim=action_config.dim,
            wan_num_heads=wan_num_heads,
            wan_head_dim=wan_head_dim,
            attr_prefix="wan_action",
            norm_eps=action_config.eps,
            include_modulation=True,
        )


class MotusActionTransformerWeights(MotusJointExpertTransformerWeights):
    def __init__(self, config, action_config, wan_num_heads, wan_head_dim):
        super().__init__(
            action_config.num_layers,
            lambda block_idx: MotusActionBlockWeights(
                block_idx=block_idx,
                config=config,
                action_config=action_config,
                wan_num_heads=wan_num_heads,
                wan_head_dim=wan_head_dim,
            ),
        )


class MotusUndBlockWeights(MotusJointExpertBlockWeights):
    def __init__(self, block_idx, config, und_config, wan_num_heads, wan_head_dim):
        super().__init__(
            block_idx=block_idx,
            config=config,
            expert_dim=und_config.dim,
            wan_num_heads=wan_num_heads,
            wan_head_dim=wan_head_dim,
            attr_prefix="wan_und",
            norm_eps=und_config.eps,
            include_modulation=False,
        )


class MotusUndTransformerWeights(MotusJointExpertTransformerWeights):
    def __init__(self, config, und_config, wan_num_heads, wan_head_dim):
        super().__init__(
            und_config.num_layers,
            lambda block_idx: MotusUndBlockWeights(
                block_idx=block_idx,
                config=config,
                und_config=und_config,
                wan_num_heads=wan_num_heads,
                wan_head_dim=wan_head_dim,
            ),
        )


class MotusTransformerWeights(WeightModule):
    weight_prefixes = {
        "video": "video_model.wan_model.",
        "action": "action_expert.",
        "und": "und_expert.",
    }

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        del lazy_load_path, lora_path
        expert_configs = build_motus_expert_configs(config)
        wan_num_heads = config["num_heads"]
        wan_head_dim = config["dim"] // wan_num_heads
        self.add_module("video", MotusWanTransformerWeights(config))
        self.add_module(
            "action",
            MotusActionTransformerWeights(
                config,
                expert_configs.action,
                wan_num_heads=wan_num_heads,
                wan_head_dim=wan_head_dim,
            ),
        )
        self.add_module(
            "und",
            MotusUndTransformerWeights(
                config,
                expert_configs.und,
                wan_num_heads=wan_num_heads,
                wan_head_dim=wan_head_dim,
            ),
        )

    def load(self, weight_dict):
        load_prefixed_submodules(self, weight_dict, self.weight_prefixes)
