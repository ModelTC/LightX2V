import torch

from lightx2v.common.modules.weight_module import WeightModule


def _get_required_weight(state_dict, key):
    """Get a required weight from state dict, raising error if missing."""
    if key not in state_dict:
        raise KeyError(f"Required weight '{key}' not found in state dict")
    return state_dict[key]


class LongCatImagePreWeights(WeightModule):
    """Pre-processing weights for LongCat Image Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.in_channels = config.get("transformer_in_channels", config.get("in_channels", 64))
        self.joint_attention_dim = config.get("joint_attention_dim", 3584)

    def load_from_state_dict(self, state_dict):
        """Load weights from state dict."""
        self.x_embedder_weight = _get_required_weight(state_dict, "x_embedder.weight")
        self.x_embedder_bias = _get_required_weight(state_dict, "x_embedder.bias")

        self.context_embedder_weight = _get_required_weight(state_dict, "context_embedder.weight")
        self.context_embedder_bias = _get_required_weight(state_dict, "context_embedder.bias")

        # Time embedding
        self.time_proj_weight = None  # Sinusoidal, no weight
        self.timestep_embedder_linear_1_weight = _get_required_weight(state_dict, "time_embed.timestep_embedder.linear_1.weight")
        self.timestep_embedder_linear_1_bias = _get_required_weight(state_dict, "time_embed.timestep_embedder.linear_1.bias")
        self.timestep_embedder_linear_2_weight = _get_required_weight(state_dict, "time_embed.timestep_embedder.linear_2.weight")
        self.timestep_embedder_linear_2_bias = _get_required_weight(state_dict, "time_embed.timestep_embedder.linear_2.bias")

    def to_cuda(self):
        """Move weights to CUDA."""
        from lightx2v.utils.envs import GET_DTYPE
        from lightx2v_platform.base.global_var import AI_DEVICE

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(AI_DEVICE, dtype=GET_DTYPE()))

    def to_cpu(self):
        """Move weights to CPU."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to("cpu"))
