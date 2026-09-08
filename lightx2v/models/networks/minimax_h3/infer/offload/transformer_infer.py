from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3OffloadTransformerInfer(MiniMaxH3TransformerInfer):
    """Run H3 blocks through the same double-buffered prefetch path as Wan."""

    def __init__(self, config):
        super().__init__(config)
        offload_granularity = get_offload_granularity(config)
        if offload_granularity == "block":
            self.infer_func = self.infer_with_blocks_offload
        elif offload_granularity != "model":
            raise NotImplementedError(f"MiniMax-H3 does not support offload_granularity={offload_granularity!r}")

    def get_compile_block_key(self, block_idx, block):
        if self.has_block_offload_manager():
            return id(block)
        return super().get_compile_block_key(block_idx, block)

    @staticmethod
    def _weights_without_adaln(_block_index, state_dict):
        return {name: tensor for name, tensor in state_dict.items() if ".adaln_proj." not in name}

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        def run_h3_block(block_index, block):
            nonlocal hidden_states
            self.block_idx = block_index
            hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            return hidden_states

        state_dict_transform = self._weights_without_adaln if self.use_adaln_cache and self._adaln_cache_hit else None
        self.run_blocks_with_offload(
            blocks,
            run_h3_block,
            state_dict_transform=state_dict_transform,
        )
        return hidden_states
