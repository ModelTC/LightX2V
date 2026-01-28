from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class LongCatImageDoubleBlockWeights(WeightModule):
    """Weights for a single double-stream transformer block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")

        p = f"transformer_blocks.{self.block_idx}"

        # Image stream norm1 (AdaLayerNormZero)
        self.add_module(
            "norm1_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.norm1.linear.weight",
                f"{p}.norm1.linear.bias",
            ),
        )

        # Context stream norm1 (AdaLayerNormZero)
        self.add_module(
            "norm1_context_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.norm1_context.linear.weight",
                f"{p}.norm1_context.linear.bias",
            ),
        )

        # Attention - image stream
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_q.weight",
                f"{p}.attn.to_q.bias",
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_k.weight",
                f"{p}.attn.to_k.bias",
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_v.weight",
                f"{p}.attn.to_v.bias",
            ),
        )
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_q.weight",
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_k.weight",
            ),
        )

        # Attention - context stream (added projections)
        self.add_module(
            "add_q_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_q_proj.weight",
                f"{p}.attn.add_q_proj.bias",
            ),
        )
        self.add_module(
            "add_k_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_k_proj.weight",
                f"{p}.attn.add_k_proj.bias",
            ),
        )
        self.add_module(
            "add_v_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_v_proj.weight",
                f"{p}.attn.add_v_proj.bias",
            ),
        )
        self.add_module(
            "norm_added_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_added_q.weight",
            ),
        )
        self.add_module(
            "norm_added_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_added_k.weight",
            ),
        )

        # Attention output projections
        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_out.0.weight",
                f"{p}.attn.to_out.0.bias",
            ),
        )
        self.add_module(
            "to_add_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_add_out.weight",
                f"{p}.attn.to_add_out.bias",
            ),
        )

        # Attention calculation module
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        # Image FFN
        self.add_module(
            "ff_net_0_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff.net.0.proj.weight",
                f"{p}.ff.net.0.proj.bias",
            ),
        )
        self.add_module(
            "ff_net_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff.net.2.weight",
                f"{p}.ff.net.2.bias",
            ),
        )

        # Context FFN
        self.add_module(
            "ff_context_net_0_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff_context.net.0.proj.weight",
                f"{p}.ff_context.net.0.proj.bias",
            ),
        )
        self.add_module(
            "ff_context_net_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff_context.net.2.weight",
                f"{p}.ff_context.net.2.bias",
            ),
        )

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class LongCatImageSingleBlockWeights(WeightModule):
    """Weights for a single single-stream transformer block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.rms_norm_type = config.get("rms_norm_type", "torch")
        self.attn_type = config.get("attn_type", "flash_attn3")

        p = f"single_transformer_blocks.{self.block_idx}"

        # AdaLayerNormZeroSingle
        self.add_module(
            "norm_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.norm.linear.weight",
                f"{p}.norm.linear.bias",
            ),
        )

        # MLP projection
        self.add_module(
            "proj_mlp",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.proj_mlp.weight",
                f"{p}.proj_mlp.bias",
            ),
        )

        # Output projection (combined attn + mlp)
        self.add_module(
            "proj_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.proj_out.weight",
                f"{p}.proj_out.bias",
            ),
        )

        # Attention
        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_q.weight",
                f"{p}.attn.to_q.bias",
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_k.weight",
                f"{p}.attn.to_k.bias",
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_v.weight",
                f"{p}.attn.to_v.bias",
            ),
        )
        self.add_module(
            "norm_q",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_q.weight",
            ),
        )
        self.add_module(
            "norm_k",
            RMS_WEIGHT_REGISTER[self.rms_norm_type](
                f"{p}.attn.norm_k.weight",
            ),
        )

        # Attention calculation module
        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class LongCatImageTransformerWeights(WeightModule):
    """Complete transformer weights for LongCat Image model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 10)
        self.num_single_layers = config.get("num_single_layers", 20)

        # Create weight containers for each block
        self.double_blocks = WeightModuleList([LongCatImageDoubleBlockWeights(config, i) for i in range(self.num_layers)])
        self.single_blocks = WeightModuleList([LongCatImageSingleBlockWeights(config, i) for i in range(self.num_single_layers)])

        self.add_module("double_blocks", self.double_blocks)
        self.add_module("single_blocks", self.single_blocks)

    def to_cuda(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cuda(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for block in self.double_blocks:
            block.to_cpu(non_blocking=non_blocking)
        for block in self.single_blocks:
            block.to_cpu(non_blocking=non_blocking)
