from lightx2v.common.modules.weight_module import WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerAttentionBlock,
    WanTransformerWeights,
)
from lightx2v.utils.registry_factory import (
    CONV3D_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


# "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
# {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6, 14: 7, 16: 8, 18: 9, 20: 10, 22: 11, 24: 12, 26: 13, 28: 14}
class WanVaceTransformerWeights(WanTransformerWeights):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = (1, 2, 2)
        self.vace_blocks = WeightModuleList(
            [WanVaceTransformerAttentionBlock(self.config.vace_layers[i], i, self.task, self.mm_type, self.config, "vace_blocks") for i in range(len(self.config.vace_layers))]
        )
        self.add_module("vace_blocks", self.vace_blocks)

        self.add_module(
            "vace_patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"]("vace_patch_embedding.weight", "vace_patch_embedding.bias", stride=self.patch_size),
        )

    def clear(self):
        super().clear()
        for vace_block in self.vace_blocks:
            for vace_phase in vace_block.compute_phases:
                vace_phase.clear()

    def non_block_weights_to_cuda(self):
        super().non_block_weights_to_cuda()
        self.vace_patch_embedding.to_cuda()

    def non_block_weights_to_cpu(self):
        super().non_block_weights_to_cpu()
        self.vace_patch_embedding.to_cpu()


class WanVaceTransformerAttentionBlock(WanTransformerAttentionBlock):
    def __init__(self, base_block_idx, block_index, task, mm_type, config, block_prefix):
        super().__init__(block_index, task, mm_type, config, block_prefix)
        if base_block_idx == 0:
            self.add_module(
                "before_proj",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.before_proj.weight",
                    f"{block_prefix}.{self.block_index}.before_proj.bias",
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )
        self.add_module(
            "after_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.after_proj.weight",
                f"{block_prefix}.{self.block_index}.after_proj.bias",
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
