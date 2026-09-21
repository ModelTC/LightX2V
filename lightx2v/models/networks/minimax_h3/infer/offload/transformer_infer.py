from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


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

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        num_blocks = len(blocks)
        self.offload_manager.prepare_compute()

        for block_index in range(num_blocks):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            block = self.offload_manager.cuda_buffers[0]
            next_block_index = (block_index + 1) % num_blocks
            self.offload_manager.prefetch_weights(next_block_index, blocks)
            self.block_idx = block_index
            with self.offload_manager.compute_context():
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            self.offload_manager.swap_blocks()

        return hidden_states
