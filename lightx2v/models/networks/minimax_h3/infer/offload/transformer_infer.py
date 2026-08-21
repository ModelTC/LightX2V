import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class MiniMaxH3OffloadTransformerInfer(MiniMaxH3TransformerInfer):
    """Run H3 blocks through the same double-buffered prefetch path as Wan."""

    def __init__(self, config):
        super().__init__(config)
        offload_granularity = config.get("offload_granularity", "model")
        if offload_granularity == "block":
            self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
            self.infer_func = self.infer_with_blocks_offload
        elif offload_granularity != "model":
            raise NotImplementedError(f"MiniMax-H3 does not support offload_granularity={offload_granularity!r}")

    def get_compile_block_key(self, block_idx, block):
        # block offload
        if hasattr(self, "offload_manager"):
            return id(block)
        # model offload
        return super().get_compile_block_key(block_idx, block)

    def _prefetch_weights_without_adaln(self, block_index, blocks):
        with torch_device_module.stream(self.offload_manager.cuda_load_stream):
            if hasattr(self.offload_manager, "cpu_buffers"):
                source_block = self.offload_manager.cpu_buffers[0]
            else:
                source_block = blocks[block_index]
            block_state_dict = source_block.state_dict()
            weights_without_adaln = {}
            for name, tensor in block_state_dict.items():
                if ".adaln_proj." not in name:
                    weights_without_adaln[name] = tensor
            self.offload_manager.cuda_buffers[1].load_state_dict(weights_without_adaln, block_index)

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        num_blocks = len(blocks)
        if self.use_adaln_cache and not self._adaln_cache_hit:
            # The previous forward may have prefetched block 0 without AdaLN.
            # Reload the full block when the current timestep misses.
            self.offload_manager.need_init_first_buffer = True
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        for block_index in range(num_blocks):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            next_block_index = (block_index + 1) % num_blocks
            if self.use_adaln_cache and self._adaln_cache_hit:
                self._prefetch_weights_without_adaln(next_block_index, blocks)
            else:
                self.offload_manager.prefetch_weights(next_block_index, blocks)
            block = self.offload_manager.cuda_buffers[0]
            self.block_idx = block_index
            if AI_DEVICE == "xpu":
                # Match Wan's XPU offload path: overlap the next weight copy on
                # the load stream with current-block compute on the default
                # stream, then let swap_blocks() perform the device-wide sync.
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            self.offload_manager.swap_blocks()

        return hidden_states
