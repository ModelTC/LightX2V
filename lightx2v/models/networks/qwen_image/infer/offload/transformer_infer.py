import torch

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
                self.infer_func = self.infer_with_blocks_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            elif offload_granularity == "phase":
                self.infer_func = self.infer_with_phases_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                self.compiled_phases = {}

            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load:
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
