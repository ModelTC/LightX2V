import torch
import torch.nn.functional as F

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.flux2.infer.transformer_infer import Flux2TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class Flux2OffloadTransformerInfer(Flux2TransformerInfer):
    """Flux2 transformer inference with block-level CPU offload."""

    def __init__(self, config):
        super().__init__(config)
        self.use_npu_event_offload = AI_DEVICE == "npu" and self.config.get("offload_use_npu_events", False)
        resident_blocks_requested = any(
            self.config.get(key, 0) not in (None, 0)
            for key in (
                "offload_resident_double_blocks",
                "offload_resident_single_blocks",
            )
        )
        if resident_blocks_requested and not self.use_npu_event_offload:
            raise ValueError("Flux2 resident block offload currently requires AI_DEVICE='npu' and offload_use_npu_events=true")
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.offload_manager_double = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                if self.use_npu_event_offload:
                    self.offload_manager_single = WeightAsyncStreamManager(
                        offload_granularity=offload_granularity,
                        load_stream=self.offload_manager_double.cuda_load_stream,
                        compute_stream=self.offload_manager_double.compute_stream,
                    )
                    self.infer_func = self.infer_with_npu_event_offload
                else:
                    self.offload_manager_single = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                    self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "model":
                self.infer_func = super().infer
            else:
                raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")
        else:
            self.infer_func = super().infer

    @staticmethod
    def _resident_indices(block_weights, block_kind):
        attr_name = f"resident_{block_kind}_block_indices"
        return set(getattr(block_weights, attr_name, ()))

    def _run_npu_block_stage(self, manager, blocks, resident_indices, run_block):
        """Run one block family with fixed event-protected staging slots.

        Resident blocks use their original weight objects. Non-resident blocks
        are prefetched into two fixed device slots. Once a staged block has
        been enqueued on the compute stream, its slot is immediately recycled
        through a device-side ``free`` event; no host synchronization occurs.
        """
        offloaded_indices = [idx for idx in range(len(blocks)) if idx not in resident_indices]
        if not offloaded_indices:
            for block_idx, block in enumerate(blocks):
                run_block(block_idx, block)
            return

        scheduled_slots = {}
        next_offloaded = 0

        def prefetch_next(slot_idx):
            nonlocal next_offloaded
            if next_offloaded >= len(offloaded_indices):
                return
            block_idx = offloaded_indices[next_offloaded]
            manager.prefetch_to_slot(slot_idx, block_idx, blocks)
            scheduled_slots[block_idx] = slot_idx
            next_offloaded += 1

        for slot_idx in range(min(2, len(offloaded_indices))):
            prefetch_next(slot_idx)

        for block_idx, block in enumerate(blocks):
            if block_idx in resident_indices:
                run_block(block_idx, block)
                continue

            slot_idx = scheduled_slots.pop(block_idx)
            staged_block = manager.wait_ready(slot_idx)
            run_block(block_idx, staged_block)
            manager.record_free(slot_idx)
            prefetch_next(slot_idx)

    def infer_with_npu_event_offload(self, block_weights, pre_infer_out):
        """NPU block offload using device events instead of host barriers."""
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        timestep = pre_infer_out.timestep
        image_rotary_emb = pre_infer_out.image_rotary_emb
        image_rotary_positions = pre_infer_out.image_rotary_positions

        num_txt_tokens = encoder_hidden_states.shape[0]
        timestep_act = F.silu(timestep)
        double_stream_mod_img = block_weights.double_stream_modulation_img_linear.apply(timestep_act)
        double_stream_mod_txt = block_weights.double_stream_modulation_txt_linear.apply(timestep_act)
        single_stream_mod = block_weights.single_stream_modulation_linear.apply(timestep_act)

        current_stream = torch_device_module.current_stream()
        compute_stream = self.offload_manager_double.compute_stream
        compute_stream.wait_stream(current_stream)

        resident_double = self._resident_indices(block_weights, "double")
        resident_single = self._resident_indices(block_weights, "single")

        def run_double(block_idx, block):
            nonlocal encoder_hidden_states, hidden_states
            self.block_idx = block_idx
            with torch_device_module.stream(compute_stream):
                encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    image_rotary_emb,
                    image_rotary_positions,
                )

        self._run_npu_block_stage(
            self.offload_manager_double,
            block_weights.double_blocks,
            resident_double,
            run_double,
        )

        with torch_device_module.stream(compute_stream):
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        def run_single(block_idx, block):
            nonlocal hidden_states
            self.block_idx = block_idx
            with torch_device_module.stream(compute_stream):
                hidden_states = self.infer_single_stream_block(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    image_rotary_emb,
                    image_rotary_positions,
                    num_txt_tokens=num_txt_tokens,
                )

        self._run_npu_block_stage(
            self.offload_manager_single,
            block_weights.single_blocks,
            resident_single,
            run_single,
        )

        with torch_device_module.stream(compute_stream):
            hidden_states = hidden_states[num_txt_tokens:, ...]
            final_done = compute_stream.record_event()

        current_stream.wait_event(final_done)
        hidden_states.record_stream(current_stream)
        return hidden_states

    def infer_with_blocks_offload(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        timestep = pre_infer_out.timestep
        image_rotary_emb = pre_infer_out.image_rotary_emb
        image_rotary_positions = pre_infer_out.image_rotary_positions

        num_txt_tokens = encoder_hidden_states.shape[0]
        timestep_act = F.silu(timestep)
        double_stream_mod_img = block_weights.double_stream_modulation_img_linear.apply(timestep_act)
        double_stream_mod_txt = block_weights.double_stream_modulation_txt_linear.apply(timestep_act)
        single_stream_mod = block_weights.single_stream_modulation_linear.apply(timestep_act)

        current_stream = torch_device_module.current_stream()
        self.offload_manager_double.compute_stream.wait_stream(current_stream)
        for block_idx in range(len(block_weights.double_blocks)):
            self.block_idx = block_idx

            if self.offload_manager_double.need_init_first_buffer:
                self.offload_manager_double.init_first_buffer(block_weights.double_blocks)

            self.offload_manager_double.prefetch_weights((block_idx + 1) % len(block_weights.double_blocks), block_weights.double_blocks)

            with torch_device_module.stream(self.offload_manager_double.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                    self.offload_manager_double.cuda_buffers[0],
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    image_rotary_emb,
                    image_rotary_positions,
                )

            self.offload_manager_double.swap_blocks()

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        self.offload_manager_single.compute_stream.wait_stream(self.offload_manager_double.compute_stream)
        for block_idx in range(len(block_weights.single_blocks)):
            self.block_idx = block_idx

            if self.offload_manager_single.need_init_first_buffer:
                self.offload_manager_single.init_first_buffer(block_weights.single_blocks)

            self.offload_manager_single.prefetch_weights((block_idx + 1) % len(block_weights.single_blocks), block_weights.single_blocks)

            with torch_device_module.stream(self.offload_manager_single.compute_stream):
                hidden_states = self.infer_single_stream_block(
                    self.offload_manager_single.cuda_buffers[0],
                    hidden_states,
                    None,
                    single_stream_mod,
                    image_rotary_emb,
                    image_rotary_positions,
                    num_txt_tokens=num_txt_tokens,
                )

            self.offload_manager_single.swap_blocks()

        hidden_states = hidden_states[num_txt_tokens:, ...]
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        return self.infer_func(block_weights, pre_infer_out)
