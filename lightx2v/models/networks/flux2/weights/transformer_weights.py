from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER


class Flux2DoubleBlockWeights(WeightModule):
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

        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_q.weight",
            ),
        )
        self.add_module(
            "to_k",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_k.weight",
            ),
        )
        self.add_module(
            "to_v",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_v.weight",
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

        self.add_module(
            "add_q_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_q_proj.weight",
            ),
        )
        self.add_module(
            "add_k_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_k_proj.weight",
            ),
        )
        self.add_module(
            "add_v_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.add_v_proj.weight",
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

        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_out.0.weight",
            ),
        )
        self.add_module(
            "to_add_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_add_out.weight",
            ),
        )

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

        self.add_module(
            "ff_net_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff.linear_in.weight",
            ),
        )
        self.add_module(
            "ff_net_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff.linear_out.weight",
            ),
        )

        self.add_module(
            "ff_context_net_0",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff_context.linear_in.weight",
            ),
        )
        self.add_module(
            "ff_context_net_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.ff_context.linear_out.weight",
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


class Flux2SingleBlockWeights(WeightModule):
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

        self.add_module(
            "to_qkv_mlp_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_qkv_mlp_proj.weight",
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

        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{p}.attn.to_out.weight",
            ),
        )

        self.add_module("calculate", ATTN_WEIGHT_REGISTER[self.attn_type]())

        if self.config.get("seq_parallel", False):
            self.add_module(
                "calculate_parallel",
                ATTN_WEIGHT_REGISTER[self.config["parallel"].get("seq_p_attn_type", "ulysses")](),
            )

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)


class Flux2TransformerWeights(WeightModule):
    """Complete transformer weights for Flux2 model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 10)
        self.num_single_layers = config.get("num_single_layers", 20)
        self.mm_type = config.get("dit_quant_scheme", "Default")

        inner_dim = config.get("num_attention_heads", 24) * config.get("attention_head_dim", 64)

        self.double_blocks = WeightModuleList([Flux2DoubleBlockWeights(config, i) for i in range(self.num_layers)])
        self.single_blocks = WeightModuleList([Flux2SingleBlockWeights(config, i) for i in range(self.num_single_layers)])

        self.add_module("double_blocks", self.double_blocks)
        self.add_module("single_blocks", self.single_blocks)

        self.add_module(
            "double_stream_modulation_img_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "double_stream_modulation_img.linear.weight",
            ),
        )
        self.add_module(
            "double_stream_modulation_txt_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "double_stream_modulation_txt.linear.weight",
            ),
        )
        self.add_module(
            "single_stream_modulation_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "single_stream_modulation.linear.weight",
            ),
        )

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


# Backward-compatible aliases
Flux2KleinDoubleBlockWeights = Flux2DoubleBlockWeights
Flux2KleinSingleBlockWeights = Flux2SingleBlockWeights
Flux2KleinTransformerWeights = Flux2TransformerWeights
