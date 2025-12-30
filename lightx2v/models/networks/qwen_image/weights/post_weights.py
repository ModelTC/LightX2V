from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class QwenImagePostWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.task = config["task"]
        self.config = config
        # norm_out
        self.add_module(
            "norm_out_linear",
            MM_WEIGHT_REGISTER["Default"](
                "norm_out.linear.weight",
                "norm_out.linear.bias",
            ),
        )
        self.add_module("norm_out", LN_WEIGHT_REGISTER["Default"](eps=1e-6))

        # proj_out
        self.add_module(
            "proj_out_linear",
            MM_WEIGHT_REGISTER["Default"](
                "proj_out.weight",
                "proj_out.bias",
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
