import torch

from lightx2v.common.offload.event_manager import EventSlotWeightAsyncStreamManager
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.qwen_image.infer.transformer_infer import (
    QwenImageTransformerInfer,
)
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class QwenImageOffloadTransformerInfer(QwenImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_blocks = config["num_layers"]
        self.phases_num = 4
        if self.config.get("cpu_offload", False):
            self.offload_ratio = self.config.get("offload_ratio", 1)
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if self.config.get("use_event_offload", False):
                    self.infer_func = self.infer_with_event_offload
                    self.offload_manager = EventSlotWeightAsyncStreamManager(offload_granularity=offload_granularity)
                else:
                    if self.config.get("offload_resident_blocks", 0) not in (None, 0):
                        raise ValueError("Qwen-Image resident block offload requires use_event_offload=true")
                    self.infer_func = self.infer_with_blocks_offload
                    self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            elif offload_granularity == "phase":
                self.infer_func = self.infer_with_phases_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                self.compiled_phases = {}

            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load:
                if isinstance(self.offload_manager, EventSlotWeightAsyncStreamManager):
                    raise NotImplementedError("Qwen-Image event block offload does not support lazy_load")
                self.offload_manager.init_lazy_load(num_workers=self.config.get("num_disk_workers", 4))

    def get_compile_block_key(self, _block_idx, block):
        return id(block)

    def infer_with_phases_offload(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        image_rotary_positions,
        modulate_index,
    ):
        for block_idx in range(len(blocks)):
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(blocks)
                self.offload_manager.start_prefetch_block(next_prefetch)

            for phase_idx in range(self.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    self.offload_manager.init_first_buffer(blocks)

                next_block_idx = (block_idx + 1) % len(blocks) if phase_idx == self.phases_num - 1 else block_idx
                next_phase_idx = (phase_idx + 1) % self.phases_num
                if self.lazy_load and phase_idx == self.phases_num - 1:
                    self.offload_manager.swap_cpu_buffers()

                self.offload_manager.prefetch_phase(next_block_idx, next_phase_idx, blocks)
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    if phase_idx == 0:
                        img_query, img_key, img_value, img_gate1, img_mod2 = self.run_phase(
                            phase_idx,
                            self.offload_manager.cuda_buffers[phase_idx],
                            hidden_states,
                            temb_img_silu,
                            image_rotary_emb[0],
                            image_rotary_positions[0],
                            modulate_index,
                        )
                    elif phase_idx == 1:
                        txt_query, txt_key, txt_value, seq_txt, txt_gate1, txt_mod2 = self.run_phase(
                            phase_idx,
                            self.offload_manager.cuda_buffers[phase_idx],
                            encoder_hidden_states,
                            temb_txt_silu,
                            image_rotary_emb[1],
                            image_rotary_positions[1],
                        )
                    elif phase_idx == 2:
                        hidden_states, encoder_hidden_states = self.run_phase(
                            phase_idx,
                            self.offload_manager.cuda_buffers[phase_idx],
                            seq_txt,
                            img_query,
                            img_key,
                            img_value,
                            txt_query,
                            txt_key,
                            txt_value,
                            img_gate1,
                            txt_gate1,
                            hidden_states,
                            encoder_hidden_states,
                        )
                    elif phase_idx == 3:
                        encoder_hidden_states, hidden_states = self.run_phase(
                            phase_idx,
                            self.offload_manager.cuda_buffers[phase_idx],
                            hidden_states,
                            encoder_hidden_states,
                            img_mod2,
                            txt_mod2,
                            modulate_index,
                        )
                self.offload_manager.swap_phases()

        return hidden_states

    def infer_phase(self, phase_idx, phase, *args):
        phase_func = (
            self.infer_img_qkv,
            self.infer_txt_qkv,
            self.infer_cross_attn,
            self.infer_ffn,
        )[phase_idx]
        return phase_func(phase, *args)

    def get_compiled_phase(self, phase_idx, phase):
        cached = self.compiled_phases.get(phase_idx)
        if cached is not None and cached[0] is phase:
            return cached[1]

        def phase_runner(*args):
            return self.infer_phase(phase_idx, phase, *args)

        compiled = torch.compile(phase_runner, dynamic=None)
        self.compiled_phases[phase_idx] = (phase, compiled)
        return compiled

    def run_phase(self, phase_idx, phase, *args):
        if self.use_compile:
            return self.get_compiled_phase(phase_idx, phase)(*args)
        return self.infer_phase(phase_idx, phase, *args)

    def infer_with_blocks_offload(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        image_rotary_positions,
        modulate_index,
    ):
        for block_idx in range(self.num_blocks):
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % self.num_blocks
                self.offload_manager.start_prefetch_block(next_prefetch)

            if block_idx == 0:
                self.offload_manager.init_first_buffer(blocks)

            if self.lazy_load:
                self.offload_manager.swap_cpu_buffers()
            self.offload_manager.prefetch_weights((block_idx + 1) % self.num_blocks, blocks)

            with torch_device_module.stream(self.offload_manager.compute_stream):
                encoder_hidden_states, hidden_states = self.run_block(
                    block_idx,
                    self.offload_manager.cuda_buffers[0],
                    hidden_states,
                    encoder_hidden_states,
                    temb_img_silu,
                    temb_txt_silu,
                    image_rotary_emb,
                    image_rotary_positions,
                    modulate_index,
                )

            self.offload_manager.swap_blocks()

        return hidden_states

    def infer_with_event_offload(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        image_rotary_positions,
        modulate_index,
    ):
        resident_indices = set(getattr(self.block_weights, "resident_block_indices", ()))
        offloaded_indices = [idx for idx in range(self.num_blocks) if idx not in resident_indices]

        device_module = self.offload_manager.device_module
        current_stream = device_module.current_stream()
        compute_stream = self.offload_manager.compute_stream
        compute_stream.wait_stream(current_stream)

        scheduled_slots = {}
        next_offloaded = 0

        def prefetch_next(slot_idx):
            nonlocal next_offloaded
            if next_offloaded >= len(offloaded_indices):
                return
            block_idx = offloaded_indices[next_offloaded]
            self.offload_manager.prefetch_to_slot(slot_idx, block_idx, blocks)
            scheduled_slots[block_idx] = slot_idx
            next_offloaded += 1

        if offloaded_indices:
            for slot_idx in range(min(self.offload_manager.slot_count, len(offloaded_indices))):
                prefetch_next(slot_idx)

        for block_idx, resident_block in enumerate(blocks):
            if block_idx in resident_indices:
                block = resident_block
                slot_idx = None
            else:
                slot_idx = scheduled_slots.pop(block_idx)
                block = self.offload_manager.wait_ready(slot_idx)

            with device_module.stream(compute_stream):
                encoder_hidden_states, hidden_states = self.run_block(
                    block_idx,
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb_img_silu,
                    temb_txt_silu,
                    image_rotary_emb,
                    image_rotary_positions,
                    modulate_index,
                )

            if slot_idx is not None:
                self.offload_manager.record_free(slot_idx)
                prefetch_next(slot_idx)

        with device_module.stream(compute_stream):
            final_done = compute_stream.record_event()
        current_stream.wait_event(final_done)
        hidden_states.record_stream(current_stream)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        self.block_weights = block_weights
        return super().infer(block_weights, pre_infer_out)
