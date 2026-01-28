from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.models.networks.wan.weights.transformer_weights import WanTransformerWeights
from lightx2v.utils.registry_factory import (
    LN_WEIGHT_REGISTER,
    MM_WEIGHT_REGISTER,
    TENSOR_REGISTER,
)


class WanAudioTransformerWeights(WanTransformerWeights):
    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path, lora_path)
        for i in range(self.blocks_num):
            self.blocks[i].compute_phases.append(
                WanAudioAdapterCA(
                    i,
                    f"ca",
                    self.task,
                    self.mm_type,
                    self.config,
                    False,
                    False,
                    self.blocks[i].lazy_load,
                    self.blocks[i].lazy_load_file,
                )
            )

        self._add_audio_adapter_ca_to_offload_buffers()

    def _add_audio_adapter_ca_to_offload_buffers(self):
        if hasattr(self, "offload_block_cuda_buffers") and self.offload_block_cuda_buffers is not None:
            for i in range(self.offload_blocks_num):
                offload_buffer = self.offload_block_cuda_buffers[i]
                adapter_ca = WanAudioAdapterCA(
                    block_index=i,
                    block_prefix=f"ca",
                    task=self.task,
                    mm_type=self.mm_type,
                    config=self.config,
                    create_cuda_buffer=True,
                    create_cpu_buffer=False,
                    lazy_load=offload_buffer.lazy_load,
                    lazy_load_file=offload_buffer.lazy_load_file,
                )
                offload_buffer.compute_phases.append(adapter_ca)
                if self.lazy_load:
                    offload_buffer = self.offload_block_cpu_buffers[i]
                    adapter_ca = WanAudioAdapterCA(
                        block_index=i,
                        block_prefix=f"ca",
                        task=self.task,
                        mm_type=self.mm_type,
                        config=self.config,
                        create_cuda_buffer=False,
                        create_cpu_buffer=True,
                        lazy_load=offload_buffer.lazy_load,
                        lazy_load_file=offload_buffer.lazy_load_file,
                    )
                    offload_buffer.compute_phases.append(adapter_ca)

        elif hasattr(self, "offload_phase_cuda_buffers") and self.offload_phase_cuda_buffers is not None:
            adapter_ca = WanAudioAdapterCA(
                block_index=0,
                block_prefix=f"ca",
                task=self.task,
                mm_type=self.mm_type,
                config=self.config,
                create_cuda_buffer=True,
                create_cpu_buffer=False,
                lazy_load=self.blocks[0].lazy_load,
                lazy_load_file=self.blocks[0].lazy_load_file,
            )
            self.offload_phase_cuda_buffers.append(adapter_ca)
            if self.lazy_load:
                adapter_ca = WanAudioAdapterCA(
                    block_index=0,
                    block_prefix=f"ca",
                    task=self.task,
                    mm_type=self.mm_type,
                    config=self.config,
                    create_cuda_buffer=False,
                    create_cpu_buffer=True,
                    lazy_load=self.blocks[0].lazy_load,
                    lazy_load_file=self.blocks[0].lazy_load_file,
                )
                self.offload_phase_cpu_buffers.append(adapter_ca)


class WanAudioAdapterCA(WeightModule):
    def __init__(self, block_index, block_prefix, task, mm_type, config, create_cuda_buffer, create_cpu_buffer, lazy_load, lazy_load_file):
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
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "to_kv",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_index}.to_kv.weight",
                f"{block_prefix}.{block_index}.to_kv.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "to_out",
            MM_WEIGHT_REGISTER[self.mm_type](
                f"{block_prefix}.{block_index}.to_out.weight",
                f"{block_prefix}.{block_index}.to_out.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "norm_kv",
            LN_WEIGHT_REGISTER["torch"](
                f"{block_prefix}.{block_index}.norm_kv.weight",
                f"{block_prefix}.{block_index}.norm_kv.bias",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )

        self.add_module(
            "norm_q",
            LN_WEIGHT_REGISTER["torch"](),
        )

        self.add_module(
            "shift_scale_gate",
            TENSOR_REGISTER["Default"](
                f"{block_prefix}.{block_index}.shift_scale_gate",
                create_cuda_buffer,
                create_cpu_buffer,
                self.lazy_load,
                self.lazy_load_file,
            ),
        )
