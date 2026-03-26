from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class Flux2KleinPreWeights(WeightModule):
    """Pre-processing weights for Flux2Klein."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mm_type = config.get("dit_quant_scheme", "Default")

        # x_embedder
        self.add_module(
            "x_embedder",
            MM_WEIGHT_REGISTER[self.mm_type](
                "x_embedder.weight",
            ),
        )

        # context_embedder
        self.add_module(
            "context_embedder",
            MM_WEIGHT_REGISTER[self.mm_type](
                "context_embedder.weight",
            ),
        )

        # timestep_embedder (MLP: linear_1 -> silu -> linear_2)
        self.add_module(
            "timestep_embedder_linear_1",
            MM_WEIGHT_REGISTER[self.mm_type](
                "time_guidance_embed.timestep_embedder.linear_1.weight",
            ),
        )
        self.add_module(
            "timestep_embedder_linear_2",
            MM_WEIGHT_REGISTER[self.mm_type](
                "time_guidance_embed.timestep_embedder.linear_2.weight",
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
