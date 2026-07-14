from lightx2v.models.networks.wan.weights.transformer_weights import (
    WanTransformerAttentionBlock,
    WanTransformerWeights,
)
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class WanLingbotTransformerWeights(WanTransformerWeights):
    """Lingbot-specific transformer weights container."""

    def __init__(self, config, lazy_load_path=None, lora_path=None):
        super().__init__(config, lazy_load_path=lazy_load_path, lora_path=lora_path)
        self._register_lingbot_cam_layers(lora_path=lora_path)

    def _register_lingbot_cam_layers(self, lora_path=None):
        self._register_cam_for_blocks(self.blocks, lora_path=lora_path)
        if hasattr(self, "offload_block_cuda_buffers") and self.offload_block_cuda_buffers is not None:
            self._register_cam_for_blocks(self.offload_block_cuda_buffers, lora_path=lora_path)
        if hasattr(self, "offload_block_cpu_buffers") and self.offload_block_cpu_buffers is not None:
            self._register_cam_for_blocks(self.offload_block_cpu_buffers, lora_path=lora_path)

    def _register_cam_for_blocks(self, blocks, lora_path=None):
        for block in blocks:
            if isinstance(block, WanTransformerAttentionBlock):
                self._register_cam_for_single_block(block, lora_path=lora_path)

    @staticmethod
    def _register_cam_for_single_block(block, lora_path=None):
        block_prefix = "blocks"
        block_index = block.block_index
        mm_type = block.mm_type

        if hasattr(block, "cam_injector_layer1"):
            return

        block.add_module(
            "cam_injector_layer1",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_injector_layer1.weight",
                f"{block_prefix}.{block_index}.cam_injector_layer1.bias",
                block.create_cuda_buffer,
                block.create_cpu_buffer,
                block.lazy_load,
                block.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        block.add_module(
            "cam_injector_layer2",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_injector_layer2.weight",
                f"{block_prefix}.{block_index}.cam_injector_layer2.bias",
                block.create_cuda_buffer,
                block.create_cpu_buffer,
                block.lazy_load,
                block.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        block.add_module(
            "cam_scale_layer",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_scale_layer.weight",
                f"{block_prefix}.{block_index}.cam_scale_layer.bias",
                block.create_cuda_buffer,
                block.create_cpu_buffer,
                block.lazy_load,
                block.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
        block.add_module(
            "cam_shift_layer",
            MM_WEIGHT_REGISTER[mm_type](
                f"{block_prefix}.{block_index}.cam_shift_layer.weight",
                f"{block_prefix}.{block_index}.cam_shift_layer.bias",
                block.create_cuda_buffer,
                block.create_cpu_buffer,
                block.lazy_load,
                block.lazy_load_file,
                lora_prefix=block_prefix,
                lora_path=lora_path,
            ),
        )
