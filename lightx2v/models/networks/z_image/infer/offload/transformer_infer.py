import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.z_image.infer.transformer_infer import ZImageTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class ZImageOffloadTransformerInfer(ZImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                self.lazy_load = self.config.get("lazy_load", False)
                self.infer_main_blocks = self.infer_main_blocks_offload
                if self.lazy_load:
                    self.offload_manager.init_lazy_load(num_workers=self.config.get("num_disk_workers", 4))
            elif offload_granularity == "phase":
                raise NotImplementedError("offload_granularity=phase not supported")

    def infer_with_blocks_offload(
        self,
        main_blocks,
        unified,
        unified_freqs_cis,
        adaln_input,
    ):
        num_blocks = len(main_blocks)
        for block_idx in range(num_blocks):
            self.block_idx = block_idx

            if self.lazy_load:
                next_prefetch = (block_idx + 1) % num_blocks
                self.offload_manager.start_prefetch_block(next_prefetch)

            if block_idx == 0:
                self.offload_manager.init_first_buffer(main_blocks)

            if self.lazy_load:
                self.offload_manager.swap_cpu_buffers()
            self.offload_manager.prefetch_weights((block_idx + 1) % num_blocks, main_blocks)

            with torch_device_module.stream(self.offload_manager.compute_stream):
                unified = self.infer_block(
                    block_weight=self.offload_manager.cuda_buffers[0],
                    hidden_states=unified,
                    freqs_cis=unified_freqs_cis,
                    adaln_input=adaln_input,
                )

            self.offload_manager.swap_blocks()

        return unified

    def infer_main_blocks_offload(
        self,
        main_blocks,
        hidden_states,
        encoder_hidden_states,
        x_freqs_cis,
        cap_freqs_cis,
        adaln_input,
        x_len,
        cap_len,
    ):
        unified = torch.cat([hidden_states, encoder_hidden_states], dim=0)
        unified_freqs_cis = torch.cat([x_freqs_cis[:x_len], cap_freqs_cis[:cap_len]], dim=0)
        unified = self.infer_with_blocks_offload(
            main_blocks=main_blocks,
            unified=unified,
            unified_freqs_cis=unified_freqs_cis,
            adaln_input=adaln_input,
        )
        return unified
