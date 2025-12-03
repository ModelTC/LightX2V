import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE
torch_device_module = getattr(torch, AI_DEVICE)

class QwenImageOffloadTransformerInfer(QwenImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.phases_num = 3
        self.num_blocks = config["num_layers"]
        if self.config.get("cpu_offload", False):
            if "offload_ratio" in self.config:
                self.offload_ratio = self.config["offload_ratio"]
            else:
                self.offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_blocks_offload
                else:
                    assert NotImplementedError
            else:
                assert NotImplementedError

            if offload_granularity != "model":
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            else:
                assert NotImplementedError

    def infer_with_blocks_offload(self, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        for block_idx in range(self.num_blocks):
            self.block_idx = block_idx
            if block_idx == 0:
                self.offload_manager.init_first_buffer(block_weights.blocks)

            if block_idx < self.num_blocks - 1:
                self.offload_manager.prefetch_weights(block_idx + 1, block_weights.blocks)

            with torch_device_module.stream(self.offload_manager.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_block(
                    block_weight=self.offload_manager.cuda_buffers[0], hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb
                )

            self.offload_manager.swap_blocks()

        return encoder_hidden_states, hidden_states
