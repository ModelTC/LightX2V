from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class ZImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.task = config["task"]
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")
        self.add_module(
            "norm_out_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "all_final_layer.2-1.adaLN_modulation.1.weight",
                "all_final_layer.2-1.adaLN_modulation.1.bias",
            ),
        )
        self.add_module("norm_out", LN_WEIGHT_REGISTER["torch"](eps=1e-6))

        self.add_module(
            "proj_out_linear",
            MM_WEIGHT_REGISTER[self.mm_type](
                "all_final_layer.2-1.linear.weight",
                "all_final_layer.2-1.linear.bias",
            ),
        )

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
