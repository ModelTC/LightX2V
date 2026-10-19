import torch

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.offload.resident import ResidentBlocks, uniform_resident_indices, validate_residency_config
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def validate_minimax_h3_residency_config(config):
    if "offload_resident_blocks" not in config:
        return
    if config.get("offload_granularity", "model") != "block":
        raise ValueError("Static MiniMax-H3 residency requires offload_granularity='block'")
    validate_residency_config(config)
    uniform_resident_indices(config["num_layers"], config["offload_resident_blocks"])
    if config.get("model_cls") != "minimax_h3" or config.get("model_variant") != "fl2av":
        raise ValueError("Static MiniMax-H3 residency currently requires the fl2av model variant")
    if not config.get("use_adaln_cache", False) or GET_DTYPE() != torch.bfloat16:
        raise ValueError("Static MiniMax-H3 residency requires BF16 and use_adaln_cache=true")
    for option in ("dit_quantized", "dit_disk_streaming", "use_fused_qkv", "enable_cfg"):
        if config.get(option, False):
            raise ValueError(f"Static MiniMax-H3 residency does not support {option}")
    if config.get("dit_quant_scheme", "Default") != "Default":
        raise ValueError("Static MiniMax-H3 residency requires unquantized block weights")
    if config.get("cpu_offload_block_contiguous", False) and config.get("dit_release_block_offload_buffers", False):
        raise ValueError("MiniMax-H3 contiguous offload does not support dit_release_block_offload_buffers")


class MiniMaxH3OffloadTransformerInfer(MiniMaxH3TransformerInfer):
    """Run H3 blocks through the same double-buffered prefetch path as Wan."""

    def __init__(self, config):
        validate_minimax_h3_residency_config(config)
        super().__init__(config)
        self.resident_blocks = ResidentBlocks(config["offload_resident_blocks"]) if "offload_resident_blocks" in config else None
        offload_granularity = config.get("offload_granularity", "model")
        if offload_granularity == "block":
            self.offload_manager = self.create_offload_manager()
            self.infer_func = self.infer_with_blocks_offload
        elif offload_granularity != "model":
            raise NotImplementedError(f"MiniMax-H3 does not support offload_granularity={offload_granularity!r}")

    def create_offload_manager(self):
        manager_class = EventSlotWeightAsyncStreamManager if self.resident_blocks is not None else WeightAsyncStreamManager
        return manager_class(offload_granularity="block")

    def release_resident_blocks(self):
        if self.resident_blocks is not None:
            self.resident_blocks.release()
            if self.offload_manager is not None:
                self.offload_manager.reset_slots()

    def get_compile_block_key(self, block_idx, block):
        # block offload
        if hasattr(self, "offload_manager"):
            return id(block)
        # model offload
        return super().get_compile_block_key(block_idx, block)

    def infer_with_blocks_offload(self, blocks, hidden_states, pre_infer_out):
        if self.resident_blocks is not None:
            self.resident_blocks.prepare(blocks)

            def run_block(index, block, state):
                self.block_idx = index
                return self.run_block(index, block, state, pre_infer_out)

            # These inputs originate on the caller stream but are consumed on compute_stream.
            for tensor in (hidden_states, pre_infer_out.temb, pre_infer_out.timestep_indices, pre_infer_out.adaln_indices, *pre_infer_out.rotary_emb):
                if tensor is not None:
                    tensor.record_stream(self.offload_manager.compute_stream)
            try:
                hidden_states = self.offload_manager.run_blocks(blocks, self.resident_blocks.indices, hidden_states, run_block)
            except BaseException:
                self.release_resident_blocks()
                raise
            hidden_states.record_stream(torch_device_module.current_stream())
            return hidden_states

        num_blocks = len(blocks)
        current_stream = torch_device_module.current_stream()
        self.offload_manager.compute_stream.wait_stream(current_stream)

        for block_index in range(num_blocks):
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)

            next_block_index = (block_index + 1) % num_blocks
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
