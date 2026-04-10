import torch

try:
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
except ImportError:
    get_cutlass_fused_moe_module = None

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.attn import FlashAttn2Weight, FlashAttn3Weight  # noqa: F401
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFusedQKNorm3DRope
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
                config=self.config,
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
    def __init__(self, block_index, config, mm_type, attn_type="flash_attn2"):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}"

        self.add_module(
            "input_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.input_layernorm_mot_gen.weight", eps=1e-6),
        )

        attn = NeoppAttentionWeights(block_index, mm_type, attn_type)
        self.add_module("self_attn", attn)

        self.add_module(
            "post_attention_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.post_attention_layernorm_mot_gen.weight", eps=1e-6),
        )

        if config["version"] == "moe":
            gen_num_experts = int(config["llm_config"]["gen_num_experts"])
            mlp_mot_gen = NeoppSparseMoeWeights(block_index, mm_type, "mlp_mot_gen", gen_num_experts)
        elif config["version"] == "dense":
            mlp_mot_gen = NeoppMlpWeights(block_index, mm_type)
        else:
            raise ValueError(f"Unsupported version: {config['version']}")
        self.add_module("mlp_mot_gen", mlp_mot_gen)


class NeoppAttentionWeights(WeightModule):
    def __init__(self, block_index, mm_type, attn_type="flash_attn2"):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.self_attn"

        self.add_module("q_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.q_proj_mot_gen.weight", None))

        self.add_module("k_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.k_proj_mot_gen.weight", None))

        self.add_module("v_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.v_proj_mot_gen.weight", None))

        self.add_module("o_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.o_proj_mot_gen.weight", None))

        self.add_module(
            "qk_norm",
            RMSWeightFusedQKNorm3DRope(
                f"{prefix}.q_norm_mot_gen.weight",
                f"{prefix}.q_norm_hw_mot_gen.weight",
                f"{prefix}.k_norm_mot_gen.weight",
                f"{prefix}.k_norm_hw_mot_gen.weight",
            ),
        )

        self.add_module("cross_attn", ATTN_WEIGHT_REGISTER[attn_type]())


class NeoppSparseMoeWeights(WeightModule):
    def __init__(self, block_index, mm_type, subname, num_experts):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.{subname}"

        self.add_module("gate", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate.weight", None))

        self.num_experts = num_experts
        experts = WeightModuleList(NeoppMoeSingleExpertWeights(block_index, mm_type, subname, j) for j in range(num_experts))
        self.add_module("experts", experts)

    def load(self, weight_dict):
        super().load(weight_dict)
        self._build_flashinfer_weights()

    def _build_flashinfer_weights(self):
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            get_cutlass_fused_moe_module(f"{major * 10 + minor}")
        fc1_list, fc2_list = [], []
        for expert_w in self.experts:
            up_w = expert_w.up_proj._get_actual_weight().t().contiguous()
            gate_w = expert_w.gate_proj._get_actual_weight().t().contiguous()
            fc1_list.append(torch.cat([up_w, gate_w], dim=0))
            fc2_list.append(expert_w.down_proj._get_actual_weight().t().contiguous())
        self._fi_fc1_weight = torch.stack(fc1_list, dim=0)
        self._fi_fc2_weight = torch.stack(fc2_list, dim=0)


class NeoppMoeSingleExpertWeights(WeightModule):
    def __init__(self, block_index, mm_type, subname, expert_index):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.{subname}.experts.{expert_index}"
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate_proj.weight", None))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.up_proj.weight", None))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.down_proj.weight", None))


class NeoppMlpWeights(WeightModule):
    def __init__(self, block_index, mm_type):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.mlp_mot_gen"
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate_proj.weight", None))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.up_proj.weight", None))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.down_proj.weight", None))

    # def load(self, weight_dict):
    #     super().load(weight_dict)
    #     self._build_flashinfer_weights()

    # def _build_flashinfer_weights(self):
    #     gate_w = self.gate_proj._get_actual_weight()  # [hidden_size, intermediate_size]
    #     up_w = self.up_proj._get_actual_weight()      # [hidden_size, intermediate_size]
    #     self._fi_gate_up_weight = torch.cat([gate_w, up_w], dim=1).contiguous()


class NeoppFmHeadWeights(WeightModule):
    def __init__(self, mm_type):
        super().__init__()
        self.add_module(
            "fm_head_0",
            MM_WEIGHT_REGISTER["Default"](
                "fm_modules.fm_head.0.weight",
                "fm_modules.fm_head.0.bias",
            ),
        )

        self.add_module(
            "fm_head_2",
            MM_WEIGHT_REGISTER["Default"](
                "fm_modules.fm_head.2.weight",
                "fm_modules.fm_head.2.bias",
            ),
        )
