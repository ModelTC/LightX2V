from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanAudioTransformerWeights(WanTransformerWeights):
    def __init__(self, config):
        super().__init__(config)
        for i in range(self.blocks_num):
            self.blocks[i].compute_phases.append(
                WanAudioAdapterCA(
                    i,
                    f"ca",
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    self.blocks[i].lazy_load,
                    self.blocks[i].lazy_load_file,
                )
            )
        if hasattr(self, "offload_blocks"):
            for i in range(self.offload_blocks_num):
                self.offload_blocks[i].compute_phases.append(
                    WanAudioAdapterCA(
                        i,
                        f"ca",
                        self.task,
                        self.mm_type,
                        self.config,
                        True,
                        self.offload_blocks[i].lazy_load,
                        self.offload_blocks[i].lazy_load_file,
                    )
                )


class WanAudioAdapterCA(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, is_offload_block, lazy_load, lazy_load_file):
        super().__init__()
        self.block_index = block_index
        self.mm_type = mm_type
        self.task = task
        self.config = config
        self.lazy_load = lazy_load
        self.lazy_load_file = lazy_load_file

        self.add_module(
            "to_q",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_index}.to_q.weight",
                f"{block_prefix}.{block_index}.to_q.bias",
                is_offload_block,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "to_kv",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_index}.to_kv.weight",
                f"{block_prefix}.{block_index}.to_kv.bias",
                is_offload_block,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_index}.to_out.weight",
                f"{block_prefix}.{block_index}.to_out.bias",
                is_offload_block,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "norm_kv",
            LN_WEIGHT_REGISTER["Default"](
                f"{block_prefix}.{block_index}.norm_kv.weight",
                f"{block_prefix}.{block_index}.norm_kv.bias",
                is_offload_block,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "norm_q",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module(
            "shift_scale_gate",
            TENSOR_REGISTER["Default"](
                f"{block_prefix}.{block_index}.shift_scale_gate",
                is_offload_block,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
