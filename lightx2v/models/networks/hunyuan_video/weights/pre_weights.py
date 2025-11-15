from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import (
    CONV3D_WEIGHT_REGISTER,
    EMBEDDING_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class HunyuanVideo15PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.patch_size = config["patch_size"]  # (1, 1, 1)

        self.add_module(
            "img_in",
            CONV3D_WEIGHT_REGISTER["Default"](
                "img_in.proj.weight",
                "img_in.proj.bias",
                stride=self.patch_size,
            ),
        )
        self.add_module(
            "time_in_0",
            MM_WEIGHT_REGISTER["Default"](
                "time_in.mlp.0.weight",
                "time_in.mlp.0.bias",
            ),
        )
        self.add_module(
            "time_in_2",
            MM_WEIGHT_REGISTER["Default"](
                "time_in.mlp.2.weight",
                "time_in.mlp.2.bias",
            ),
        )

        self.add_module(
            "txt_in_t_embedder_0",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.t_embedder.mlp.0.weight",
                "txt_in.t_embedder.mlp.0.bias",
            ),
        )
        self.add_module(
            "txt_in_t_embedder_2",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.t_embedder.mlp.2.weight",
                "txt_in.t_embedder.mlp.2.bias",
            ),
        )

        self.add_module(
            "txt_in_c_embedder_0",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.c_embedder.linear_1.weight",
                "txt_in.c_embedder.linear_1.bias",
            ),
        )
        self.add_module(
            "txt_in_c_embedder_2",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.c_embedder.linear_2.weight",
                "txt_in.c_embedder.linear_2.bias",
            ),
        )

        self.add_module(
            "txt_in_input_embedder",
            MM_WEIGHT_REGISTER["Default"](
                "txt_in.input_embedder.weight",
                "txt_in.input_embedder.bias",
            ),
        )

        self.add_module(
            "individual_token_refiner",
            WeightModuleList(
                [
                    IndividualTokenRefinerBlock(
                        i,
                        self.mm_type,
                        self.config,
                        "txt_in.individual_token_refiner.blocks",
                    )
                    for i in range(2)  # 2 blocks
                ]
            ),
        )

        self.add_module(
            "cond_type_embedding",
            EMBEDDING_WEIGHT_REGISTER["Default"](
                "cond_type_embedding.weight",
            ),
        )


class IndividualTokenRefinerBlock(WeightModule):
    def __init__(self, block_idx, mm_type, config, block_prefix):
        super().__init__()
        self.config = config
        self.mm_type = mm_type
        self.add_module(
            "norm1",
            LN_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.norm1.weight", f"{block_prefix}.{block_idx}.norm1.bias"),
        )
        self.add_module(
            "self_attn_qkv",
            MM_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.self_attn_qkv.weight", f"{block_prefix}.{block_idx}.self_attn_qkv.bias"),
        )
        self.add_module(
            "self_attn_proj",
            MM_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.self_attn_proj.weight", f"{block_prefix}.{block_idx}.self_attn_proj.bias"),
        )
        self.add_module(
            "norm2",
            LN_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.norm2.weight", f"{block_prefix}.{block_idx}.norm2.bias"),
        )
        self.add_module(
            "mlp_fc1",
            MM_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.mlp.fc1.weight", f"{block_prefix}.{block_idx}.mlp.fc1.bias"),
        )
        self.add_module(
            "mlp_fc2",
            MM_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.mlp.fc2.weight", f"{block_prefix}.{block_idx}.mlp.fc2.bias"),
        )
        self.add_module(
            "adaLN_modulation",
            MM_WEIGHT_REGISTER["Default"](f"{block_prefix}.{block_idx}.adaLN_modulation.1.weight", f"{block_prefix}.{block_idx}.adaLN_modulation.1.bias"),
        )
