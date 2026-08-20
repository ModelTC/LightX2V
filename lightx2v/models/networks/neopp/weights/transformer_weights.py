import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.attn import FlashAttn2Weight, FlashAttn3Weight  # noqa: F401
from lightx2v.common.ops.moe.fused_moe import create_local_fused_moe
from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightFusedQKNorm3DRope
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    CONV2D_WEIGHT_REGISTER,
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
                lora_path=lora_path,
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
            NeoppFmHeadWeights(self.mm_type, config=self.config),
        )

    @staticmethod
    def _reject_routed_expert_adapter(weight_dict):
        if any(".mlp_mot_gen.experts." in key for key in weight_dict):
            raise NotImplementedError("NeoPP common fused MoE backends do not support routed-expert LoRA/diff adapters")

    def register_lora(self, weight_dict, strength):
        self._reject_routed_expert_adapter(weight_dict)
        super().register_lora(weight_dict, strength)

    def update_lora(self, weight_dict, strength):
        self._reject_routed_expert_adapter(weight_dict)
        super().update_lora(weight_dict, strength)

    def register_diff(self, weight_dict):
        self._reject_routed_expert_adapter(weight_dict)
        super().register_diff(weight_dict)


class NeoppDecoderLayerWeights(WeightModule):
    def __init__(self, block_index, config, mm_type, attn_type="flash_attn2", lora_path=None):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}"

        self.add_module(
            "input_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.input_layernorm_mot_gen.weight", eps=1e-6),
        )

        use_triton_qknorm_rope = config.get("use_triton_qknorm_rope", True)
        attn = NeoppAttentionWeights(config, block_index, mm_type, attn_type, use_triton_qknorm_rope, lora_path=lora_path)
        self.add_module("self_attn", attn)

        self.add_module(
            "post_attention_layernorm_mot_gen",
            RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.post_attention_layernorm_mot_gen.weight", eps=1e-6),
        )

        if config["version"] == "moe":
            if mm_type != "Default":
                raise NotImplementedError(f"NeoPP common fused MoE backends support only mm_type='Default', got {mm_type!r}")
            gen_num_experts = int(config["llm_config"]["gen_num_experts"])
            moe_backend = config.get("moe_backend", "flashinfer")
            supported_moe_backends = {"flashinfer", "torch_grouped_mm", "torch_expert_loop"}
            if moe_backend not in supported_moe_backends:
                raise ValueError(f"Invalid moe_backend={moe_backend!r}, expected one of {sorted(supported_moe_backends)}")
            fi_cfg = config.get("moe_flashinfer_setting") or {}
            if fi_cfg.get("autotune") and moe_backend != "flashinfer":
                raise ValueError("moe_flashinfer_setting.autotune=true requires moe_backend='flashinfer'")
            tune_max_num_tokens = int(fi_cfg.get("tune_max_num_tokens", 8192))
            mlp_mot_gen = NeoppSparseMoeWeights(
                block_index,
                mm_type,
                gen_num_experts,
                moe_backend=moe_backend,
                tune_max_num_tokens=tune_max_num_tokens,
                lora_path=lora_path,
            )
        elif config["version"] == "dense":
            mlp_mot_gen = NeoppMlpWeights(block_index, mm_type, lora_path=lora_path)
        else:
            raise ValueError(f"Unsupported version: {config['version']}")
        self.add_module("mlp_mot_gen", mlp_mot_gen)


class NeoppAttentionWeights(WeightModule):
    def __init__(self, config, block_index, mm_type, attn_type="flash_attn2", use_triton_qknorm_rope=True, lora_path=None):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.self_attn"
        lora_prefix = "language_model"

        self.add_module("q_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.q_proj_mot_gen.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

        self.add_module("k_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.k_proj_mot_gen.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

        self.add_module("v_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.v_proj_mot_gen.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

        self.add_module("o_proj_mot_gen", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.o_proj_mot_gen.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

        if use_triton_qknorm_rope:
            # Fused triton kernel: single module holds all 4 norm weights and applies
            # dual-RMSNorm + 3D Neox-RoPE for Q and K in one kernel launch.
            self.add_module(
                "qk_norm",
                RMSWeightFusedQKNorm3DRope(
                    f"{prefix}.q_norm_mot_gen.weight",
                    f"{prefix}.q_norm_hw_mot_gen.weight",
                    f"{prefix}.k_norm_mot_gen.weight",
                    f"{prefix}.k_norm_hw_mot_gen.weight",
                ),
            )
        else:
            # Pure torch: 4 separate RMSNorm modules, logic expanded in transformer_infer.py.
            self.add_module(
                "q_norm_mot_gen",
                RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.q_norm_mot_gen.weight", eps=1e-6),
            )
            self.add_module(
                "q_norm_hw_mot_gen",
                RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.q_norm_hw_mot_gen.weight", eps=1e-6),
            )
            self.add_module(
                "k_norm_mot_gen",
                RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.k_norm_mot_gen.weight", eps=1e-6),
            )
            self.add_module(
                "k_norm_hw_mot_gen",
                RMS_WEIGHT_REGISTER["fp32_variance_qwen"](f"{prefix}.k_norm_hw_mot_gen.weight", eps=1e-6),
            )

        self.add_module("cross_attn", ATTN_WEIGHT_REGISTER[attn_type]())
        if config["seq_parallel"]:
            self.add_module(
                "cross_attn_parallel",
                ATTN_WEIGHT_REGISTER[config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )


class NeoppSparseMoeWeights(WeightModule):
    def __init__(self, block_index, mm_type, num_experts, moe_backend, tune_max_num_tokens, lora_path=None):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.mlp_mot_gen"
        lora_prefix = "language_model"

        self.moe_backend = moe_backend
        self.tune_max_num_tokens = tune_max_num_tokens
        self.add_module("gate", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

        experts = WeightModuleList(NeoppMoeSingleExpertWeights(block_index, mm_type, j, lora_path=lora_path) for j in range(num_experts))
        self.add_module("experts", experts)

    def load(self, weight_dict):
        super().load(weight_dict)

        fc1_weights, fc2_weights = [], []
        for expert_idx, expert in enumerate(self.experts):
            up_weight = self._loaded_weight(expert.up_proj, f"experts.{expert_idx}.up_proj").t()
            gate_weight = self._loaded_weight(expert.gate_proj, f"experts.{expert_idx}.gate_proj").t()
            down_weight = self._loaded_weight(expert.down_proj, f"experts.{expert_idx}.down_proj").t()
            fc1_weights.append(torch.cat([up_weight, gate_weight], dim=0))
            fc2_weights.append(down_weight)

        self.add_module(
            "fused_moe",
            create_local_fused_moe(
                self.moe_backend,
                fc1_weights,
                fc2_weights,
                "swiglu",
                tune_max_num_tokens=self.tune_max_num_tokens,
            ),
        )

        for expert in self.experts:
            for projection in (expert.gate_proj, expert.up_proj, expert.down_proj):
                projection.weight = None
                projection.pin_weight = None

    @staticmethod
    def _loaded_weight(module, name):
        if module.weight is not None:
            return module._get_actual_weight()
        if module.pin_weight is not None:
            return module.pin_weight
        raise RuntimeError(f"NeoPP MoE weight {name} is not loaded")


class NeoppMoeSingleExpertWeights(WeightModule):
    def __init__(self, block_index, mm_type, expert_index, lora_path=None):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.mlp_mot_gen.experts.{expert_index}"
        lora_prefix = "language_model"
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.up_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.down_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))


class NeoppMlpWeights(WeightModule):
    def __init__(self, block_index, mm_type, lora_path=None):
        super().__init__()
        prefix = f"language_model.model.layers.{block_index}.mlp_mot_gen"
        lora_prefix = "language_model"
        self.add_module("gate_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.gate_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))
        self.add_module("up_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.up_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))
        self.add_module("down_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.down_proj.weight", None, lora_prefix=lora_prefix, lora_path=lora_path))

    # def load(self, weight_dict):
    #     super().load(weight_dict)
    #     self._build_flashinfer_weights()

    # def _build_flashinfer_weights(self):
    #     gate_w = self.gate_proj._get_actual_weight()  # [hidden_size, intermediate_size]
    #     up_w = self.up_proj._get_actual_weight()      # [hidden_size, intermediate_size]
    #     self._fi_gate_up_weight = torch.cat([gate_w, up_w], dim=1).contiguous()


class NeoppFmHeadWeights(WeightModule):
    def __init__(self, mm_type, config=None):
        super().__init__()
        lora_prefix = "fm_modules"
        # New "pixel head" variant (use_pixel_head=True): fm_head is a ConvDecoder
        # (PixelShuffle + Conv2d) that decodes hidden states directly to RGB pixels,
        # instead of the legacy per-token MLP head (fm_head.0 / fm_head.2).
        self.use_pixel_head = bool(config.get("use_pixel_head", False)) if config is not None else False
        if self.use_pixel_head:
            # Conv2d(k=3, p=1). in/out channels are inferred from PixelShuffle upstream,
            # weights carry the true shapes: conv1[1024,1024,3,3], conv2[192,256,3,3].
            self.add_module(
                "conv1",
                CONV2D_WEIGHT_REGISTER["Default"](
                    "fm_modules.fm_head.conv1.weight",
                    "fm_modules.fm_head.conv1.bias",
                    stride=1,
                    padding=1,
                ),
            )
            self.add_module(
                "conv2",
                CONV2D_WEIGHT_REGISTER["Default"](
                    "fm_modules.fm_head.conv2.weight",
                    "fm_modules.fm_head.conv2.bias",
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.add_module(
                "fm_head_0",
                MM_WEIGHT_REGISTER["Default"](
                    "fm_modules.fm_head.0.weight",
                    "fm_modules.fm_head.0.bias",
                    lora_prefix=lora_prefix,
                ),
            )

            self.add_module(
                "fm_head_2",
                MM_WEIGHT_REGISTER["Default"](
                    "fm_modules.fm_head.2.weight",
                    "fm_modules.fm_head.2.bias",
                    lora_prefix=lora_prefix,
                ),
            )
