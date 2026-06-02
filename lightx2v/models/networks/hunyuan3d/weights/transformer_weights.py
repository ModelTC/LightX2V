from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, LN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class Hunyuan3DFeedForwardWeights(WeightModule):
    """Diffusers FeedForward weights (gelu MLP)."""

    def __init__(self, prefix, mm_type):
        super().__init__()
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.net.0.proj.weight",
                f"{prefix}.net.0.proj.bias",
            ),
        )
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.net.2.weight",
                f"{prefix}.net.2.bias",
            ),
        )


class Hunyuan3DMLPWeights(WeightModule):
    def __init__(self, prefix, mm_type):
        super().__init__()
        self.add_module(
            "fc1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.fc1.weight",
                f"{prefix}.fc1.bias",
            ),
        )
        self.add_module(
            "fc2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.fc2.weight",
                f"{prefix}.fc2.bias",
            ),
        )


class Hunyuan3DMoEWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type):
        super().__init__()
        prefix = f"blocks.{block_idx}.moe"
        num_experts = config.get("num_experts", 8)
        self.num_experts = num_experts
        self.moe_top_k = config.get("moe_top_k", 2)
        self.add_module(
            "gate",
            MM_WEIGHT_REGISTER[mm_type](
                f"{prefix}.gate.weight",
                None,
            ),
        )
        self.add_module(
            "shared_experts",
            Hunyuan3DFeedForwardWeights(f"{prefix}.shared_experts", mm_type),
        )
        experts = WeightModuleList(Hunyuan3DFeedForwardWeights(f"{prefix}.experts.{expert_idx}", mm_type) for expert_idx in range(num_experts))
        self.add_module("experts", experts)


class Hunyuan3DSelfAttentionWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type, ln_type, rms_norm_type, attn_type, qkv_bias):
        super().__init__()
        prefix = f"blocks.{block_idx}.attn1"
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.qk_norm = config.get("qk_norm", True)

        self.add_module("to_q", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_q.weight", f"{prefix}.to_q.bias" if qkv_bias else None))
        self.add_module("to_k", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_k.weight", f"{prefix}.to_k.bias" if qkv_bias else None))
        self.add_module("to_v", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_v.weight", f"{prefix}.to_v.bias" if qkv_bias else None))
        if self.qk_norm:
            self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.q_norm.weight"))
            self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.k_norm.weight"))
        else:
            self.norm_q = None
            self.norm_k = None
        self.add_module("out_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.out_proj.weight", f"{prefix}.out_proj.bias"))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[attn_type]())


class Hunyuan3DCrossAttentionWeights(WeightModule):
    def __init__(self, config, block_idx, mm_type, rms_norm_type, attn_type, qkv_bias):
        super().__init__()
        prefix = f"blocks.{block_idx}.attn2"
        self.num_heads = config["num_heads"]
        self.head_dim = config["hidden_size"] // self.num_heads
        self.qk_norm = config.get("qk_norm", True)

        self.add_module("to_q", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_q.weight", f"{prefix}.to_q.bias" if qkv_bias else None))
        self.add_module("to_k", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_k.weight", f"{prefix}.to_k.bias" if qkv_bias else None))
        self.add_module("to_v", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.to_v.weight", f"{prefix}.to_v.bias" if qkv_bias else None))
        if self.qk_norm:
            self.add_module("norm_q", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.q_norm.weight"))
            self.add_module("norm_k", RMS_WEIGHT_REGISTER[rms_norm_type](f"{prefix}.k_norm.weight"))
        else:
            self.norm_q = None
            self.norm_k = None
        self.add_module("out_proj", MM_WEIGHT_REGISTER[mm_type](f"{prefix}.out_proj.weight", f"{prefix}.out_proj.bias"))
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[attn_type]())


class Hunyuan3DTransformerBlockWeights(WeightModule):
    """Weights for one HunYuanDiT block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.depth = config["depth"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.ln_type = config.get("ln_norm_type", "torch")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")
        qkv_bias = config.get("qkv_bias", False)
        prefix = f"blocks.{block_idx}"

        self.add_module("norm1", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm1.weight", f"{prefix}.norm1.bias", eps=1e-6))
        self.add_module("norm2", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm2.weight", f"{prefix}.norm2.bias", eps=1e-6))
        self.add_module("norm3", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.norm3.weight", f"{prefix}.norm3.bias", eps=1e-6))
        self.add_module(
            "attn1",
            Hunyuan3DSelfAttentionWeights(config, block_idx, self.mm_type, self.ln_type, self.rms_norm_type, self.attn_type, qkv_bias),
        )
        self.add_module(
            "attn2",
            Hunyuan3DCrossAttentionWeights(config, block_idx, self.mm_type, self.rms_norm_type, self.attn_type, qkv_bias),
        )

        use_moe = self.depth - block_idx <= config.get("num_moe_layers", 6)
        if use_moe:
            self.add_module("moe", Hunyuan3DMoEWeights(config, block_idx, self.mm_type))
            self.mlp = None
        else:
            self.add_module("mlp", Hunyuan3DMLPWeights(f"{prefix}.mlp", self.mm_type))
            self.moe = None

        if block_idx > self.depth // 2:
            self.add_module("skip_norm", LN_WEIGHT_REGISTER[self.ln_type](f"{prefix}.skip_norm.weight", f"{prefix}.skip_norm.bias", eps=1e-6))
            self.add_module(
                "skip_linear",
                MM_WEIGHT_REGISTER[self.mm_type](f"{prefix}.skip_linear.weight", f"{prefix}.skip_linear.bias"),
            )
        else:
            self.skip_norm = None
            self.skip_linear = None


class Hunyuan3DTransformerWeights(WeightModule):
    """Transformer weights for Hunyuan3D shape DiT."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.depth = config["depth"]
        blocks = WeightModuleList(Hunyuan3DTransformerBlockWeights(config, block_idx) for block_idx in range(self.depth))
        self.add_module("blocks", blocks)
