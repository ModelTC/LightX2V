from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class WorldPlayActionWeights(WeightModule):
    """
    Weight module for WorldPlay action conditioning.

    Contains:
    - action_in: TimestepEmbedder for discrete action encoding (81 classes)
    - img_attn_prope_proj: Linear projection for ProPE attention output
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.double_blocks_num = config["mm_double_blocks_depth"]

        # Action embedder (TimestepEmbedder style)
        self.add_module(
            "action_in_0",
            MM_WEIGHT_REGISTER["Default"](
                "action_in.mlp.0.weight",
                "action_in.mlp.0.bias",
            ),
        )
        self.add_module(
            "action_in_2",
            MM_WEIGHT_REGISTER["Default"](
                "action_in.mlp.2.weight",
                "action_in.mlp.2.bias",
            ),
        )

        # ProPE projection layers for each double block
        for i in range(self.double_blocks_num):
            self.add_module(
                f"img_attn_prope_proj_{i}",
                MM_WEIGHT_REGISTER["Default"](
                    f"double_blocks.{i}.img_attn_prope_proj.weight",
                    f"double_blocks.{i}.img_attn_prope_proj.bias",
                ),
            )
