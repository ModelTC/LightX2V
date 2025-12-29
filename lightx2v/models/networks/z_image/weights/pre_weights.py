from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import (
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class ZImagePreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.add_module(
            "img_in",
            MM_WEIGHT_REGISTER["Default"]("all_x_embedder.2-1.weight", "all_x_embedder.2-1.bias"),
        )
        self.add_module(
            "txt_in",
            MM_WEIGHT_REGISTER["Default"]("cap_embedder.1.weight", "cap_embedder.1.bias"),
        )

        self.add_module("txt_norm", RMS_WEIGHT_REGISTER["Default"]("cap_embedder.0.weight"))
        self.add_module("time_text_embed_timestep_embedder_linear_1", MM_WEIGHT_REGISTER["Default"]("t_embedder.mlp.0.weight", "t_embedder.mlp.0.bias"))
        self.add_module("time_text_embed_timestep_embedder_linear_2", MM_WEIGHT_REGISTER["Default"]("t_embedder.mlp.2.weight", "t_embedder.mlp.2.bias"))
        self.add_module("x_pad_token", TENSOR_REGISTER["Default"]("x_pad_token"))
        self.add_module("cap_pad_token", TENSOR_REGISTER["Default"]("cap_pad_token"))

    def to_cpu(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cpu"):
                module.to_cpu(non_blocking=non_blocking)

    def to_cuda(self, non_blocking=True):
        for module in self._modules.values():
            if module is not None and hasattr(module, "to_cuda"):
                module.to_cuda(non_blocking=non_blocking)
