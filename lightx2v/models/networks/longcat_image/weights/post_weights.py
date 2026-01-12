import torch

from lightx2v.common.modules.weight_module import WeightModule

from .pre_weights import _get_required_weight


class LongCatImagePostWeights(WeightModule):
    """Post-processing weights for LongCat Image Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.out_channels = config.get("transformer_in_channels", config.get("in_channels", 64))
        self.patch_size = config.get("patch_size", 1)

    def load_from_state_dict(self, state_dict):
        """Load weights from state dict."""
        # norm_out (AdaLayerNormContinuous)
        self.norm_out_linear_weight = _get_required_weight(state_dict, "norm_out.linear.weight")
        self.norm_out_linear_bias = _get_required_weight(state_dict, "norm_out.linear.bias")

        # proj_out
        self.proj_out_weight = _get_required_weight(state_dict, "proj_out.weight")
        self.proj_out_bias = _get_required_weight(state_dict, "proj_out.bias")

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
