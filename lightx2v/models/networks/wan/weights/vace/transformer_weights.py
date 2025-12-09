from lightx2v.common.modules.weight_module import WeightModuleList
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerAttentionBlock,
    WanTransformerWeights,
)
from lightx2v.utils.registry_factory import (
    CONV3D_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
)


class WanVaceTransformerWeights(WanTransformerWeights):
    def __init__(self, config):
        super().__init__(config)
        self.patch_size = (1, 2, 2)
        self.register_offload_buffers(config)
        self.vace_blocks = WeightModuleList(
            [WanVaceTransformerAttentionBlock(self.config["vace_layers"][i], i, self.task, self.mm_type, self.config, False, False, "vace_blocks") for i in range(len(self.config["vace_layers"]))]
        )
        self.add_module("vace_blocks", self.vace_blocks)
        self.add_module(
            "vace_patch_embedding",
            CONV3D_WEIGHT_REGISTER["Default"]("vace_patch_embedding.weight", "vace_patch_embedding.bias", stride=self.patch_size),
        )

    def register_offload_buffers(self, config):
        super().register_offload_buffers(config)
        if config["cpu_offload"]:
            if config["offload_granularity"] == "block":
                self.vace_offload_block_cuda_buffers = WeightModuleList(
                    [
                        WanVaceTransformerAttentionBlock(self.config["vace_layers"][0], 0, self.task, self.mm_type, self.config, True, False, "vace_blocks"),
                        WanVaceTransformerAttentionBlock(self.config["vace_layers"][0], 0, self.task, self.mm_type, self.config, True, False, "vace_blocks"),
                    ]
                )
                self.add_module("vace_offload_block_cuda_buffers", self.vace_offload_block_cuda_buffers)
                self.vace_offload_phase_cuda_buffers = None
            elif config["offload_granularity"] == "phase":
                raise NotImplementedError

    def non_block_weights_to_cuda(self):
        super().non_block_weights_to_cuda()
        self.vace_patch_embedding.to_cuda()

    def non_block_weights_to_cpu(self):
        super().non_block_weights_to_cpu()
        self.vace_patch_embedding.to_cpu()


class WanVaceTransformerAttentionBlock(WanTransformerAttentionBlock):
    def __init__(self, base_block_idx, block_index, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, block_prefix):
        super().__init__(block_index, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, block_prefix)
        if base_block_idx == 0:
            self.compute_phases[0].add_module(
                "before_proj",
                MM_WEIGHT_REGISTER[self.mm_type](
                    f"{block_prefix}.{self.block_index}.before_proj.weight",
                    f"{block_prefix}.{self.block_index}.before_proj.bias",
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )

        self.compute_phases[-1].add_module(
            "after_proj",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{self.block_index}.after_proj.weight",
                f"{block_prefix}.{self.block_index}.after_proj.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
