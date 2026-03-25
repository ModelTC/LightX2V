from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.attn import FlashAttn2Weight, FlashAttn3Weight  # noqa: F401
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class NeoppTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        llm_config = config["llm_config"]
        self.blocks_num = llm_config["num_hidden_layers"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.attn_type = config.get("attn_type", "flash_attn2")

        blocks = WeightModuleList(
            NeoppDecoderLayerWeights(
                block_index=i,
                llm_config=llm_config,
                mm_type=self.mm_type,
                attn_type=self.attn_type,
            )
            for i in range(self.blocks_num)
        )
        self.add_module("blocks", blocks)

        self.add_module(
            "norm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"]("language_model.model.norm_mot_gen.weight", eps=1e-6),
        )

        self.add_module(
            "fm_head",
            NeoppFmHeadWeights(self.mm_type),
        )


class NeoppDecoderLayerWeights(WeightModule):
    def __init__(self, block_index, llm_config, mm_type, attn_type="flash_attn2"):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}"

        self.add_module(
            "input_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.input_layernorm_mot_gen.weight", eps=1e-6),
        )

        attn = NeoppAttentionWeights(block_index, llm_config, mm_type, attn_type)
        self.add_module("self_attn", attn)

        self.add_module(
            "post_attention_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.post_attention_layernorm_mot_gen.weight", eps=1e-6),
        )

        gen_num_experts = int(llm_config["gen_num_experts"])
        mlp_mot_gen = NeoppSparseMoeWeights(block_index, mm_type, "mlp_mot_gen", gen_num_experts)
        self.add_module("mlp_mot_gen", mlp_mot_gen)


class NeoppAttentionWeights(WeightModule):
    def __init__(self, block_index, llm_config, mm_type, attn_type="flash_attn2"):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.self_attn"

        self.add_module("q_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.q_proj_mot_gen.weight"))

        self.add_module("k_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.k_proj_mot_gen.weight"))

        self.add_module("v_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.v_proj_mot_gen.weight"))

        self.add_module("o_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.o_proj_mot_gen.weight"))

        self.add_module("q_norm_mot_gen", RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.q_norm_mot_gen.weight", eps=1e-6))
        self.add_module("q_norm_hw_mot_gen", RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.q_norm_hw_mot_gen.weight", eps=1e-6))
        self.add_module("k_norm_mot_gen", RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.k_norm_mot_gen.weight", eps=1e-6))
        self.add_module("k_norm_hw_mot_gen", RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.k_norm_hw_mot_gen.weight", eps=1e-6))

        self.add_module("cross_attn", ATTN_WEIGHT_REGISTER[attn_type]())


class NeoppSparseMoeWeights(WeightModule):
    def __init__(self, block_index, mm_type, subname, num_experts):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.{subname}"

        self.add_module("gate", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate.weight"))

        self.num_experts = num_experts
        experts = WeightModuleList(NeoppMoeSingleExpertWeights(block_index, mm_type, subname, j) for j in range(num_experts))
        self.add_module("experts", experts)


class NeoppMoeSingleExpertWeights(WeightModule):
    def __init__(self, block_index, mm_type, subname, expert_index):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.{subname}.experts.{expert_index}"
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate_proj.weight"))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.up_proj.weight"))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.down_proj.weight"))


class NeoppFmHeadWeights(WeightModule):
    def __init__(self, mm_type):
        super().__init__()
        self.add_module(
            "fm_head_0",
            MM_WEIGHT_REGISTER[mm_type](
                "fm_modules.fm_head.0.weight",
                "fm_modules.fm_head.0.bias",
            ),
        )

        self.add_module(
            "fm_head_2",
            MM_WEIGHT_REGISTER[mm_type](
                "fm_modules.fm_head.2.weight",
                "fm_modules.fm_head.2.bias",
            ),
        )
