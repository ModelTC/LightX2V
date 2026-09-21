from lightx2v.common.offload.mps_manager import MpsSharedWeightAsyncStreamManager
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3MpsOffloadTransformerInfer(MiniMaxH3TransformerInfer):
    """Prefetch disk weights into two shared MPS buffers on the default stream."""

    def __init__(self, config):
        super().__init__(config)
        self.offload_manager = MpsSharedWeightAsyncStreamManager(offload_granularity=config.get("offload_granularity", "model"))

    def get_compile_block_key(self, block_idx, block):
        return id(block)

    def infer(self, block_weights, pre_infer_out):
        block_weights._ensure_streaming_block()
        if not self.offload_manager.cuda_buffers:
            self.offload_manager.init_cuda_buffer(block_weights.offload_block_cuda_buffers)
        if self.use_adaln_cache:
            self._prepare_adaln_cache(pre_infer_out)
        hidden_states = pre_infer_out.hidden_states
        num_blocks = len(block_weights)
        try:
            for block_index in range(num_blocks):
                if self.offload_manager.need_init_first_buffer:
                    self.offload_manager.init_first_buffer(block_weights)

                block = self.offload_manager.cuda_buffers[0]
                block_weights.prepare_streaming_block(block, block_index)
                self.offload_manager.prefetch_weights((block_index + 1) % num_blocks, block_weights)
                self.block_idx = block_index
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
                self.offload_manager.swap_blocks()
            return hidden_states
        except Exception:
            self.offload_manager.close()
            raise
