import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer


class QwenImageOffloadTransformerInfer(QwenImageTransformerInfer):
    def __init__(self, config, blocks):
        super().__init__(config, blocks)
        self.phases_num = 3
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
            elif offload_granularity == "phase":
                assert NotImplementedError
            else:
                assert NotImplementedError

            if offload_granularity != "model":
                self.weights_stream_mgr = WeightAsyncStreamManager(blocks_num=len(self.blocks), offload_ratio=self.offload_ratio, phases_num=self.phases_num)
            else:
                assert NotImplementedError

    def infer_with_blocks_offload(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs):
        for block_idx in range(len(self.blocks)):
            self.block_idx = block_idx
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = self.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < len(self.blocks) - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, self.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_block(
                    block=self.blocks[block_idx],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            self.weights_stream_mgr.swap_weights()

        return encoder_hidden_states, hidden_states
