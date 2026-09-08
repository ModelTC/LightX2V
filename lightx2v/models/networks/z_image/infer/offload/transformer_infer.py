import torch

from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.z_image.infer.transformer_infer import ZImageTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class ZImageOffloadTransformerInfer(ZImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = get_offload_granularity(self.config)
            if offload_granularity == "block":
                self.lazy_load = self.config.get("lazy_load", False)
                self.infer_main_blocks = self.infer_main_blocks_offload
            elif offload_granularity == "phase":
                raise NotImplementedError("offload_granularity=phase not supported")

    def infer_with_blocks_offload(
        self,
        main_blocks,
        unified,
        unified_freqs_cis,
        unified_rope_positions,
        adaln_input,
        image_tokens_len,
    ):
        if self.lazy_load:
            return self.infer_with_lazy_blocks_offload(
                main_blocks,
                unified,
                unified_freqs_cis,
                unified_rope_positions,
                adaln_input,
                image_tokens_len,
            )

        def run_z_image_block(block_idx, block):
            nonlocal unified
            self.block_idx = block_idx
            unified = self.infer_block(
                block_weight=block,
                hidden_states=unified,
                freqs_cis=unified_freqs_cis,
                rope_positions=unified_rope_positions,
                adaln_input=adaln_input,
                image_tokens_len=image_tokens_len,
            )
            return unified

        self.run_blocks_with_offload(main_blocks, run_z_image_block)
        return unified

    def infer_with_lazy_blocks_offload(
        self,
        main_blocks,
        unified,
        unified_freqs_cis,
        unified_rope_positions,
        adaln_input,
        image_tokens_len,
    ):
        manager = self.get_block_offload_manager(main_blocks)
        num_blocks = len(main_blocks)
        for block_idx in range(num_blocks):
            self.block_idx = block_idx

            if self.lazy_load:
                next_prefetch = (block_idx + 1) % num_blocks
                manager.start_prefetch_block(next_prefetch)

            if block_idx == 0:
                manager.init_first_buffer(main_blocks)

            if self.lazy_load:
                manager.swap_cpu_buffers()
            manager.prefetch_weights((block_idx + 1) % num_blocks, main_blocks)

            with torch_device_module.stream(manager.compute_stream):
                unified = self.infer_block(
                    block_weight=manager.cuda_buffers[0],
                    hidden_states=unified,
                    freqs_cis=unified_freqs_cis,
                    rope_positions=unified_rope_positions,
                    adaln_input=adaln_input,
                    image_tokens_len=image_tokens_len,
                )

            manager.swap_blocks()

        return unified

    def infer_main_blocks_offload(
        self,
        main_blocks,
        hidden_states,
        encoder_hidden_states,
        unified_freqs_cis,
        unified_rope_positions,
        adaln_input,
        x_len,
        cap_len,
    ):
        unified = torch.cat([hidden_states, encoder_hidden_states], dim=0)
        unified = self.infer_with_blocks_offload(
            main_blocks=main_blocks,
            unified=unified,
            unified_freqs_cis=unified_freqs_cis,
            unified_rope_positions=unified_rope_positions,
            adaln_input=adaln_input,
            image_tokens_len=x_len,
        )
        return unified
