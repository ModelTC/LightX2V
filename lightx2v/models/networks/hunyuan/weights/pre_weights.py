from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, CONV3D_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER


class HunyuanPreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.add_module("img_in_proj", CONV3D_WEIGHT_REGISTER["Default"]("img_in.proj.weight", "img_in.proj.bias", stride=(1, 2, 2)))

        self.add_module("txt_in_input_embedder", MM_WEIGHT_REGISTER["Default"]("txt_in.input_embedder.weight", "txt_in.input_embedder.bias"))
        self.add_module("txt_in_t_embedder_mlp_0", MM_WEIGHT_REGISTER["Default"]("txt_in.t_embedder.mlp.0.weight", "txt_in.t_embedder.mlp.0.bias"))
        self.add_module("txt_in_t_embedder_mlp_2", MM_WEIGHT_REGISTER["Default"]("txt_in.t_embedder.mlp.2.weight", "txt_in.t_embedder.mlp.2.bias"))
        self.add_module("txt_in_c_embedder_linear_1", MM_WEIGHT_REGISTER["Default"]("txt_in.c_embedder.linear_1.weight", "txt_in.c_embedder.linear_1.bias"))
        self.add_module("txt_in_c_embedder_linear_2", MM_WEIGHT_REGISTER["Default"]("txt_in.c_embedder.linear_2.weight", "txt_in.c_embedder.linear_2.bias"))

        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_norm1",
            LN_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.norm1.weight", "txt_in.individual_token_refiner.blocks.0.norm1.bias", eps=1e-6),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_self_attn_qkv",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight", "txt_in.individual_token_refiner.blocks.0.self_attn_qkv.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_self_attn_proj",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.self_attn_proj.weight", "txt_in.individual_token_refiner.blocks.0.self_attn_proj.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_norm2",
            LN_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.norm2.weight", "txt_in.individual_token_refiner.blocks.0.norm2.bias", eps=1e-6),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_mlp_fc1",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.mlp.fc1.weight", "txt_in.individual_token_refiner.blocks.0.mlp.fc1.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_mlp_fc2",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.mlp.fc2.weight", "txt_in.individual_token_refiner.blocks.0.mlp.fc2.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_0_adaLN_modulation_1",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.weight", "txt_in.individual_token_refiner.blocks.0.adaLN_modulation.1.bias"),
        )

        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_norm1",
            LN_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.norm1.weight", "txt_in.individual_token_refiner.blocks.1.norm1.bias", eps=1e-6),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_self_attn_qkv",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_self_attn_proj",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.self_attn_proj.weight", "txt_in.individual_token_refiner.blocks.1.self_attn_proj.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_norm2",
            LN_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.norm2.weight", "txt_in.individual_token_refiner.blocks.1.norm2.bias", eps=1e-6),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_mlp_fc1",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.mlp.fc1.weight", "txt_in.individual_token_refiner.blocks.1.mlp.fc1.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_mlp_fc2",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.mlp.fc2.weight", "txt_in.individual_token_refiner.blocks.1.mlp.fc2.bias"),
        )
        self.add_module(
            "txt_in_individual_token_refiner_blocks_1_adaLN_modulation_1",
            MM_WEIGHT_REGISTER["Default"]("txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.weight", "txt_in.individual_token_refiner.blocks.1.adaLN_modulation.1.bias"),
        )

        self.add_module("time_in_mlp_0", MM_WEIGHT_REGISTER["Default"]("time_in.mlp.0.weight", "time_in.mlp.0.bias"))
        self.add_module("time_in_mlp_2", MM_WEIGHT_REGISTER["Default"]("time_in.mlp.2.weight", "time_in.mlp.2.bias"))
        self.add_module("vector_in_in_layer", MM_WEIGHT_REGISTER["Default"]("vector_in.in_layer.weight", "vector_in.in_layer.bias"))
        self.add_module("vector_in_out_layer", MM_WEIGHT_REGISTER["Default"]("vector_in.out_layer.weight", "vector_in.out_layer.bias"))
        self.add_module("guidance_in_mlp_0", MM_WEIGHT_REGISTER["Default"]("guidance_in.mlp.0.weight", "guidance_in.mlp.0.bias"))
        self.add_module("guidance_in_mlp_2", MM_WEIGHT_REGISTER["Default"]("guidance_in.mlp.2.weight", "guidance_in.mlp.2.bias"))

        # attention weights section
        self.add_module("txt_in_attn_1", ATTN_WEIGHT_REGISTER["torch_sdpa"]())
