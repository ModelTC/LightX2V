import os

from safetensors import safe_open

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerWeights,
)
from lightx2v.utils.registry_factory import (
    ATTN_WEIGHT_REGISTER,
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    RMS_WEIGHT_REGISTER,
)


class WanAnimateTransformerWeights(WanTransformerWeights):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path, lora_path)
        self.adapter_blocks_num = self.blocks_num // 5
        for i in range(self.blocks_num):
            if i % 5 == 0:
                self.blocks[i].compute_phases.append(WanAnimateFuserBlock(self.config, i // 5, "face_adapter.fuser_blocks", self.mm_type))
            else:
                self.blocks[i].compute_phases.append(WeightModule())
        self._add_animate_fuserblock_to_offload_buffers()

    def _add_animate_fuserblock_to_offload_buffers(self):
        if hasattr(self, "offload_block_cuda_buffers") and self.offload_block_cuda_buffers is not None:
            for i in range(self.offload_blocks_num):
                self.offload_block_cuda_buffers[i].compute_phases.append(WanAnimateFuserBlock(self.config, 0, "face_adapter.fuser_blocks", self.mm_type, create_cuda_buffer=True))
            if self.lazy_load:
                self.offload_block_cpu_buffers[i].compute_phases.append(WanAnimateFuserBlock(self.config, 0, "face_adapter.fuser_blocks", self.mm_type, create_cpu_buffer=True))
        elif hasattr(self, "offload_phase_cuda_buffers") and self.offload_phase_cuda_buffers is not None:
            self.offload_phase_cuda_buffers.append(WanAnimateFuserBlock(self.config, 0, "face_adapter.fuser_blocks", self.mm_type, create_cuda_buffer=True))
            if self.lazy_load:
                self.offload_phase_cpu_buffers.append(WanAnimateFuserBlock(self.config, 0, "face_adapter.fuser_blocks", self.mm_type, create_cpu_buffer=True))


class WanAnimateFuserBlock(WeightModule):
    def __init__(self, config, block_index, block_prefix, mm_type, create_cuda_buffer=False, create_cpu_buffer=False):
        super().__init__()
        self.config = config
        self.is_post_adapter = True
        lazy_load = config.get("lazy_load", False)
        if lazy_load:
            lazy_load_path = os.path.join(
                config.dit_quantized_ckpt,
                f"{block_prefix[:-1]}_{block_index}.safetensors",
            )
            lazy_load_file = safe_open(lazy_load_path, framework="pt", device="cpu")
        else:
            lazy_load_file = None

        self.add_module(
            "linear1_kv",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.linear1_kv.weight",
                f"{block_prefix}.{block_index}.linear1_kv.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                self.is_post_adapter,
            ),
        )

        self.add_module(
            "linear1_q",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.linear1_q.weight",
                f"{block_prefix}.{block_index}.linear1_q.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                self.is_post_adapter,
            ),
        )
        self.add_module(
            "linear2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.linear2.weight",
                f"{block_prefix}.{block_index}.linear2.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                self.is_post_adapter,
            ),
        )

        self.add_module(
            "q_norm",
            RMS_WEIGHT_REGISTER["sgl-kernel"](
                f"{block_prefix}.{block_index}.q_norm.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                self.is_post_adapter,
            ),
        )

        self.add_module(
            "k_norm",
            RMS_WEIGHT_REGISTER["sgl-kernel"](
                f"{block_prefix}.{block_index}.k_norm.weight",
                create_cuda_buffer,
                create_cpu_buffer,
                lazy_load,
                lazy_load_file,
                self.is_post_adapter,
            ),
        )

        self.add_module(
            "pre_norm_feat",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module(
            "pre_norm_motion",
            LN_WEIGHT_REGISTER["Default"](),
        )

        self.add_module("adapter_attn", ATTN_WEIGHT_REGISTER[config["adapter_attn_type"]]())
