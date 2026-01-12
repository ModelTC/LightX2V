import torch

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList

from .pre_weights import _get_required_weight


class LongCatImageDoubleBlockWeights(WeightModule):
    """Weights for a single double-stream transformer block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]

    def load_from_state_dict(self, state_dict, prefix=""):
        """Load weights from state dict."""
        p = f"{prefix}transformer_blocks.{self.block_idx}."

        # Image stream norm1 (AdaLayerNormZero)
        self.norm1_linear_weight = _get_required_weight(state_dict, f"{p}norm1.linear.weight")
        self.norm1_linear_bias = _get_required_weight(state_dict, f"{p}norm1.linear.bias")

        # Context stream norm1 (AdaLayerNormZero)
        self.norm1_context_linear_weight = _get_required_weight(state_dict, f"{p}norm1_context.linear.weight")
        self.norm1_context_linear_bias = _get_required_weight(state_dict, f"{p}norm1_context.linear.bias")

        # Attention - image stream
        self.attn_to_q_weight = _get_required_weight(state_dict, f"{p}attn.to_q.weight")
        self.attn_to_q_bias = _get_required_weight(state_dict, f"{p}attn.to_q.bias")
        self.attn_to_k_weight = _get_required_weight(state_dict, f"{p}attn.to_k.weight")
        self.attn_to_k_bias = _get_required_weight(state_dict, f"{p}attn.to_k.bias")
        self.attn_to_v_weight = _get_required_weight(state_dict, f"{p}attn.to_v.weight")
        self.attn_to_v_bias = _get_required_weight(state_dict, f"{p}attn.to_v.bias")
        self.attn_norm_q_weight = _get_required_weight(state_dict, f"{p}attn.norm_q.weight")
        self.attn_norm_k_weight = _get_required_weight(state_dict, f"{p}attn.norm_k.weight")

        # Attention - context stream (added projections)
        self.attn_add_q_proj_weight = _get_required_weight(state_dict, f"{p}attn.add_q_proj.weight")
        self.attn_add_q_proj_bias = _get_required_weight(state_dict, f"{p}attn.add_q_proj.bias")
        self.attn_add_k_proj_weight = _get_required_weight(state_dict, f"{p}attn.add_k_proj.weight")
        self.attn_add_k_proj_bias = _get_required_weight(state_dict, f"{p}attn.add_k_proj.bias")
        self.attn_add_v_proj_weight = _get_required_weight(state_dict, f"{p}attn.add_v_proj.weight")
        self.attn_add_v_proj_bias = _get_required_weight(state_dict, f"{p}attn.add_v_proj.bias")
        self.attn_norm_added_q_weight = _get_required_weight(state_dict, f"{p}attn.norm_added_q.weight")
        self.attn_norm_added_k_weight = _get_required_weight(state_dict, f"{p}attn.norm_added_k.weight")

        # Attention output projections
        self.attn_to_out_0_weight = _get_required_weight(state_dict, f"{p}attn.to_out.0.weight")
        self.attn_to_out_0_bias = _get_required_weight(state_dict, f"{p}attn.to_out.0.bias")
        self.attn_to_add_out_weight = _get_required_weight(state_dict, f"{p}attn.to_add_out.weight")
        self.attn_to_add_out_bias = _get_required_weight(state_dict, f"{p}attn.to_add_out.bias")

        # Image FFN
        self.ff_net_0_proj_weight = _get_required_weight(state_dict, f"{p}ff.net.0.proj.weight")
        self.ff_net_0_proj_bias = _get_required_weight(state_dict, f"{p}ff.net.0.proj.bias")
        self.ff_net_2_weight = _get_required_weight(state_dict, f"{p}ff.net.2.weight")
        self.ff_net_2_bias = _get_required_weight(state_dict, f"{p}ff.net.2.bias")

        # Context FFN
        self.ff_context_net_0_proj_weight = _get_required_weight(state_dict, f"{p}ff_context.net.0.proj.weight")
        self.ff_context_net_0_proj_bias = _get_required_weight(state_dict, f"{p}ff_context.net.0.proj.bias")
        self.ff_context_net_2_weight = _get_required_weight(state_dict, f"{p}ff_context.net.2.weight")
        self.ff_context_net_2_bias = _get_required_weight(state_dict, f"{p}ff_context.net.2.bias")

    def to_cuda(self):
        """Move all weights to CUDA."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.cuda())

    def to_cpu(self):
        """Move all weights to CPU."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.cpu())


class LongCatImageSingleBlockWeights(WeightModule):
    """Weights for a single single-stream transformer block."""

    def __init__(self, config, block_idx):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]

    def load_from_state_dict(self, state_dict, prefix=""):
        """Load weights from state dict."""
        p = f"{prefix}single_transformer_blocks.{self.block_idx}."

        # AdaLayerNormZeroSingle
        self.norm_linear_weight = _get_required_weight(state_dict, f"{p}norm.linear.weight")
        self.norm_linear_bias = _get_required_weight(state_dict, f"{p}norm.linear.bias")

        # MLP projection
        self.proj_mlp_weight = _get_required_weight(state_dict, f"{p}proj_mlp.weight")
        self.proj_mlp_bias = _get_required_weight(state_dict, f"{p}proj_mlp.bias")

        # Output projection (combined attn + mlp)
        self.proj_out_weight = _get_required_weight(state_dict, f"{p}proj_out.weight")
        self.proj_out_bias = _get_required_weight(state_dict, f"{p}proj_out.bias")

        # Attention
        self.attn_to_q_weight = _get_required_weight(state_dict, f"{p}attn.to_q.weight")
        self.attn_to_q_bias = _get_required_weight(state_dict, f"{p}attn.to_q.bias")
        self.attn_to_k_weight = _get_required_weight(state_dict, f"{p}attn.to_k.weight")
        self.attn_to_k_bias = _get_required_weight(state_dict, f"{p}attn.to_k.bias")
        self.attn_to_v_weight = _get_required_weight(state_dict, f"{p}attn.to_v.weight")
        self.attn_to_v_bias = _get_required_weight(state_dict, f"{p}attn.to_v.bias")
        self.attn_norm_q_weight = _get_required_weight(state_dict, f"{p}attn.norm_q.weight")
        self.attn_norm_k_weight = _get_required_weight(state_dict, f"{p}attn.norm_k.weight")

    def to_cuda(self):
        """Move all weights to CUDA."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.cuda())

    def to_cpu(self):
        """Move all weights to CPU."""
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.cpu())


class LongCatImageTransformerWeights(WeightModule):
    """Complete transformer weights for LongCat Image model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 10)
        self.num_single_layers = config.get("num_single_layers", 20)

        # Create weight containers for each block
        self.double_blocks = WeightModuleList([
            LongCatImageDoubleBlockWeights(config, i) for i in range(self.num_layers)
        ])
        self.single_blocks = WeightModuleList([
            LongCatImageSingleBlockWeights(config, i) for i in range(self.num_single_layers)
        ])

    def load_from_state_dict(self, state_dict, prefix=""):
        """Load all block weights from state dict."""
        for block in self.double_blocks:
            block.load_from_state_dict(state_dict, prefix)
        for block in self.single_blocks:
            block.load_from_state_dict(state_dict, prefix)

    def to_cuda(self):
        """Move all weights to CUDA."""
        for block in self.double_blocks:
            block.to_cuda()
        for block in self.single_blocks:
            block.to_cuda()

    def to_cpu(self):
        """Move all weights to CPU."""
        for block in self.double_blocks:
            block.to_cpu()
        for block in self.single_blocks:
            block.to_cpu()
