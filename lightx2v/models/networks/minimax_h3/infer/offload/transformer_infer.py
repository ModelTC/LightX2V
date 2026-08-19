import torch
from loguru import logger

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
            self.synchronous_block_offload = bool(config.get("synchronous_block_offload", AI_DEVICE == "xpu"))
            if self.synchronous_block_offload:
                logger.info("MiniMax-H3 uses synchronized block offload on {}", AI_DEVICE)
                self.infer_func = self.infer_with_blocks_offload_synchronously
            else:
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
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        for block_index in range(num_blocks):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            self.offload_manager.prefetch_weights((block_index + 1) % num_blocks, blocks)
            block = self.offload_manager.cuda_buffers[0]
            self.block_idx = block_index
            with torch_device_module.stream(self.offload_manager.compute_stream):
                hidden_states = self.run_block(block_index, block, hidden_states, pre_infer_out)
            self.offload_manager.swap_blocks()

        return hidden_states

    def infer_with_blocks_offload_synchronously(self, blocks, hidden_states, pre_infer_out):
        """Serialize XPU weight copies and GEMMs to avoid queue lifetime races."""
        manager = self.offload_manager
        compute_buffer = manager.cuda_buffers[0]
        for block_index, source_block in enumerate(blocks):
            compute_buffer.load_state_dict(source_block.state_dict(), block_index)
            torch_device_module.synchronize()
            self.block_idx = block_index
            hidden_states = self.run_block(block_index, compute_buffer, hidden_states, pre_infer_out)
            torch_device_module.synchronize()
        manager.need_init_first_buffer = False
        return hidden_states
