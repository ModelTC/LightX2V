import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.common.ops.moe.fused_moe import create_local_fused_moe
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, ROPE_REGISTER, TENSOR_REGISTER


class LingBotVideoTransformerWeights(WeightModule):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__()
        self.config = config
        self.blocks_num = config["depth"]
        blocks = WeightModuleList([LingBotVideoBlockWeights(i, config) for i in range(self.blocks_num)])
        self.add_module("blocks", blocks)


class LingBotVideoBlockWeights(WeightModule):
    def __init__(self, block_index, config):
        super().__init__()
        prefix = f"blocks.{block_index}"
        self.block_index = block_index
        self.config = config
        self.add_module("scale_shift_table", TENSOR_REGISTER["Default"](f"{prefix}.scale_shift_table"))
        self.add_module("norm1", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.norm1.weight"))
        self.add_module("norm2", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.norm2.weight"))
        self.add_module("norm_post_attn", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.norm_post_attn.weight"))
        self.add_module("norm_post_ffn", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.norm_post_ffn.weight"))
        self.add_module("attn", LingBotVideoAttentionWeights(prefix, config))
        self.add_module("ffn", LingBotVideoFFNWeights(prefix, block_index, config))


class LingBotVideoAttentionWeights(WeightModule):
    def __init__(self, prefix, config):
        super().__init__()
        self.add_module(
            "rope",
            ROPE_REGISTER[config.get("rope_type", "torch_complex_rope")](layout="interleaved", compute_dtype=torch.float32),
        )
        mm_type = config.get("dit_quant_scheme", "Default")
        if mm_type != "Default":
            raise NotImplementedError("LingBot-Video currently supports original BF16 transformer weights only.")
        qkv_bias = config.get("qkv_bias", False)
        for name in ("to_q", "to_k", "to_v"):
            self.add_module(
                name,
                MM_WEIGHT_REGISTER[mm_type](
                    f"{prefix}.attn.{name}.weight",
                    f"{prefix}.attn.{name}.bias" if qkv_bias else None,
                    lora_prefix="blocks",
                ),
            )
        self.add_module("norm_q", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.attn.norm_q.weight"))
        self.add_module("norm_k", RMS_WEIGHT_REGISTER["fp32_variance"](f"{prefix}.attn.norm_k.weight"))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[config.get("attn_type", "torch_sdpa")]())
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.attn.to_out.weight",
                f"{prefix}.attn.to_out.bias",
                lora_prefix="blocks",
            ),
        )


class LingBotVideoFFNWeights(WeightModule):
    def __init__(self, prefix, block_index, config):
        super().__init__()
        num_experts = int(config.get("num_experts", 0))
        self.moe_backend = config.get("moe_backend", "torch_grouped_mm")
        decoder_sparse_step = int(config.get("decoder_sparse_step", 1))
        mlp_only_layers = tuple(config.get("mlp_only_layers", ()))
        use_moe = block_index not in mlp_only_layers and num_experts > 0 and (block_index + 1) % decoder_sparse_step == 0
        self.use_moe = use_moe
        if use_moe:
            self.add_module("router", LingBotVideoRouterWeights(prefix))
            self.add_module("experts", LingBotVideoExpertsWeights(prefix))
            if config.get("n_shared_experts", 0):
                self.add_module("shared_experts", LingBotVideoDenseMLPWeights(f"{prefix}.ffn.shared_experts"))
            else:
                self.shared_experts = None
        else:
            self.add_module("dense", LingBotVideoDenseMLPWeights(f"{prefix}.ffn"))

    def load(self, weight_dict):
        super().load(weight_dict)
        self._build_fused_moe()

    @staticmethod
    def _loaded_tensor(module, name):
        tensor = getattr(module, "tensor", None)
        if tensor is not None:
            return tensor
        tensor = getattr(module, "pin_tensor", None)
        if tensor is not None:
            return tensor
        raise RuntimeError(f"LingBot-Video MoE weight {name} is not loaded")

    def _build_fused_moe(self):
        if not self.use_moe:
            return

        w1 = self._loaded_tensor(self.experts.w1, "w1")
        w2 = self._loaded_tensor(self.experts.w2, "w2")
        w3 = self._loaded_tensor(self.experts.w3, "w3")
        self.add_module(
            "fused_moe",
            create_local_fused_moe(
                self.moe_backend,
                w3,
                w2,
                "swiglu",
                fc1_gate_weight=w1,
            ),
        )
        self._modules.pop("experts")
        del self.experts


class LingBotVideoRouterWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("weight", MM_WEIGHT_REGISTER["Default-ForceFp32"](f"{prefix}.ffn.router.weight", None))
        self.add_module("e_score_correction_bias", TENSOR_REGISTER["Default"](f"{prefix}.ffn.router.e_score_correction_bias"))


class LingBotVideoExpertsWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        self.add_module("w1", TENSOR_REGISTER["Default"](f"{prefix}.ffn.experts.w1"))
        self.add_module("w2", TENSOR_REGISTER["Default"](f"{prefix}.ffn.experts.w2"))
        self.add_module("w3", TENSOR_REGISTER["Default"](f"{prefix}.ffn.experts.w3"))


class LingBotVideoDenseMLPWeights(WeightModule):
    def __init__(self, prefix):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            self.add_module(
                name,
                MM_WEIGHT_REGISTER["Default"](
                    f"{prefix}.{name}.weight",
                    None,
                    lora_prefix="blocks",
                ),
            )
