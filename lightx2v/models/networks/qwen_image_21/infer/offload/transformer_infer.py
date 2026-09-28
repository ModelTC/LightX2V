from contextlib import suppress

import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.qwen_image_21.infer.transformer_infer import QwenImage21TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class QwenImage21OffloadTransformerInfer(QwenImage21TransformerInfer):
    """Run condition prefill and denoising through double-buffered block prefetch."""

    def __init__(self, config):
        super().__init__(config)
        offload_granularity = config.get("offload_granularity", "model")
        if offload_granularity == "block":
            self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
        elif offload_granularity != "model":
            raise NotImplementedError(f"Qwen-Image-2.1 does not support offload_granularity={offload_granularity!r}")

    def get_compile_block_key(self, block_idx, block):
        if hasattr(self, "offload_manager"):
            return id(block)
        return super().get_compile_block_key(block_idx, block)

    def _run_blocks_offload(self, blocks, hidden_states, run_block):
        num_blocks = len(blocks)
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        try:
            for block_index in range(num_blocks):
                if self.offload_manager.need_init_first_buffer:
                    self.offload_manager.init_first_buffer(blocks)

                next_block_index = (block_index + 1) % num_blocks
                self.offload_manager.prefetch_weights(next_block_index, blocks)
                block = self.offload_manager.cuda_buffers[0]
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    hidden_states = run_block(block_index, block, hidden_states)
                self.offload_manager.swap_blocks()
        except Exception:
            # A partial loop leaves buffer 0 at an arbitrary block. Drain both
            # streams and reload block 0 before the next request.
            with suppress(Exception):
                torch_device_module.synchronize()
            self.offload_manager.need_init_first_buffer = True
            raise

        return hidden_states

    def _prefill_blocks(self, blocks, x, modulation, state, cache):
        if not hasattr(self, "offload_manager"):
            return super()._prefill_blocks(blocks, x, modulation, state, cache)
        return self._run_blocks_offload(
            blocks,
            x,
            lambda index, block, hidden: self.prefill_block(index, block, hidden, modulation, state, cache),
        )

    def _infer_blocks(self, blocks, x, modulation, state, cache):
        if not hasattr(self, "offload_manager"):
            return super()._infer_blocks(blocks, x, modulation, state, cache)
        return self._run_blocks_offload(
            blocks,
            x,
            lambda index, block, hidden: self.run_block(
                index,
                block,
                hidden,
                modulation,
                state.rotary,
                state.rotary_positions,
                cache.k_cache(index),
                cache.v_cache(index),
            ),
        )
