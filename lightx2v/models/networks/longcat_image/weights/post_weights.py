from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class LongCatImagePostWeights(WeightModule):
    """Post-processing weights for LongCat Image Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner_dim = config["num_attention_heads"] * config["attention_head_dim"]
        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.out_channels = config.get("transformer_in_channels", config.get("in_channels", 64))
        self.patch_size = config.get("patch_size", 1)
        self.mm_type = config.get("dit_quant_scheme", "Default")

        # norm_out (AdaLayerNormContinuous)
        self.add_module(
            "norm_out_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "norm_out.linear.weight",
                "norm_out.linear.bias",
            ),
        )

        # proj_out
        self.add_module(
            "proj_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                "proj_out.weight",
                "proj_out.bias",
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
