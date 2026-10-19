from contextlib import suppress

import torch

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.common.offload.resident import ResidentBlocks, uniform_resident_indices, validate_residency_config
from lightx2v.models.networks.qwen_image_21.infer.transformer_infer import QwenImage21TransformerInfer
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def validate_qwen_image_21_residency_config(config, lora_path=None):
    if "offload_resident_blocks" not in config:
        return
    if config.get("offload_granularity", "model") != "block":
        raise ValueError("Static Qwen-Image-2.1 residency requires offload_granularity='block'")
    validate_residency_config(config)
    if lora_path:
        raise ValueError("Static Qwen-Image-2.1 residency does not support LoRA")
    uniform_resident_indices(config["num_layers"], config["offload_resident_blocks"])
    if not config["causal_condition"]:
        raise ValueError("Static Qwen-Image-2.1 residency requires causal condition KV caching")
    if GET_DTYPE() != torch.bfloat16 or config.get("dit_quant_scheme", "Default") != "Default":
        raise ValueError("Static Qwen-Image-2.1 residency requires unquantized BF16 block weights")
    for option in ("dit_quantized", "dit_disk_streaming", "text_encoder_disk_streaming", "text_encoder_tensor_parallel", "lora_path", "enable_cfg"):
        if config.get(option, False):
            raise ValueError(f"Static Qwen-Image-2.1 residency does not support {option}")


class QwenImage21OffloadTransformerInfer(QwenImage21TransformerInfer):
    """Run condition prefill and denoising through double-buffered block prefetch."""

    def __init__(self, config):
        super().__init__(config)
        use_residency = "offload_resident_blocks" in config
        self.resident_blocks = ResidentBlocks(config["offload_resident_blocks"]) if use_residency else None
        offload_granularity = config.get("offload_granularity", "model")
        if offload_granularity == "block":
            manager_class = EventSlotWeightAsyncStreamManager if use_residency else WeightAsyncStreamManager
            self.offload_manager = manager_class(offload_granularity="block")
        elif offload_granularity != "model":
            raise NotImplementedError(f"Qwen-Image-2.1 does not support offload_granularity={offload_granularity!r}")

    def release_resident_blocks(self):
        if self.resident_blocks is not None:
            self.resident_blocks.release()
            self.offload_manager.reset_slots()

    def get_compile_block_key(self, block_idx, block):
        if hasattr(self, "offload_manager"):
            return id(block)
        return super().get_compile_block_key(block_idx, block)

    def _run_blocks_offload(self, blocks, hidden_states, run_block, inputs):
        if self.resident_blocks is not None:
            # Called first by prefill, then reused by every denoising step.
            self.resident_blocks.prepare(blocks)
            for tensor in (hidden_states, *inputs):
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
            (*modulation, state.rotary, state.rotary_positions),
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
            (*modulation, state.rotary, state.rotary_positions),
        )
