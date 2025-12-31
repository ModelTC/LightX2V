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
            if "offload_ratio" in self.config:
                self.offload_ratio = self.config["offload_ratio"]
            else:
                self.offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            elif offload_granularity == "phase":
                self.infer_func = self.infer_with_phases_offload
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)

            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load:
                self.offload_manager.init_lazy_load(num_workers=self.config.get("num_disk_workers", 1))

    def infer_with_phases_offload(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        modulate_index,
    ):
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            if self.lazy_load:
                next_prefetch = (block_idx + 1) % len(blocks)
                self.offload_manager.start_prefetch_block(next_prefetch)

            for phase_idx in range(self.phases_num):
                # if self.offload_manager.need_init_first_buffer:
                if block_idx == 0 and phase_idx == 0:
                    self.offload_manager.init_first_buffer(blocks)

                next_block_idx = (block_idx + 1) % len(blocks) if phase_idx == self.phases_num - 1 else block_idx
                next_phase_idx = (phase_idx + 1) % self.phases_num
                if self.lazy_load:
                    if phase_idx == self.phases_num - 1:
                        self.offload_manager.swap_cpu_buffers()

                self.offload_manager.prefetch_phase(next_block_idx, next_phase_idx, blocks)
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    if phase_idx == 0:
                        img_query, img_key, img_value, img_gate1, img_mod2 = self.infer_img_qkv(
                            img_attn_phase=self.offload_manager.cuda_buffers[phase_idx],
                            hidden_states=hidden_states,
                            temb_img_silu=temb_img_silu,
                            img_freqs=image_rotary_emb[0],
                            modulate_index=modulate_index,
                        )
                    elif phase_idx == 1:
                        txt_query, txt_key, txt_value, seq_txt, txt_gate1, txt_mod2 = self.infer_txt_qkv(
                            txt_attn_phase=self.offload_manager.cuda_buffers[phase_idx],
                            encoder_hidden_states=encoder_hidden_states,
                            temb_txt_silu=temb_txt_silu,
                            txt_freqs=image_rotary_emb[1],
                        )
                    elif phase_idx == 2:
                        hidden_states, encoder_hidden_states = self.infer_cross_attn(
                            cross_attn_phase=self.offload_manager.cuda_buffers[phase_idx],
                            seq_txt=seq_txt,
                            img_query=img_query,
                            img_key=img_key,
                            img_value=img_value,
                            txt_query=txt_query,
                            txt_key=txt_key,
                            txt_value=txt_value,
                            img_gate1=img_gate1,
                            txt_gate1=txt_gate1,
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                        )

                    elif phase_idx == 3:
                        encoder_hidden_states, hidden_states = self.infer_ffn(
                            ffn_phase=self.offload_manager.cuda_buffers[phase_idx],
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            img_mod2=img_mod2,
                            txt_mod2=txt_mod2,
                            modulate_index=modulate_index,
                        )
                self.offload_manager.swap_phases()

        return hidden_states

    def infer_with_blocks_offload(
        self,
        blocks,
        hidden_states,
        encoder_hidden_states,
        temb_img_silu,
        temb_txt_silu,
        image_rotary_emb,
        modulate_index,
    ):
        for block_idx in range(self.num_blocks):
            self.block_idx = block_idx

            if self.lazy_load:
                next_prefetch = (block_idx + 1) % self.num_blocks
                self.offload_manager.start_prefetch_block(next_prefetch)

            if block_idx == 0:
                self.offload_manager.init_first_buffer(blocks)

            if self.lazy_load:
                self.offload_manager.swap_cpu_buffers()
            self.offload_manager.prefetch_weights((block_idx + 1) % self.num_blocks, blocks)

            with torch_device_module.stream(self.offload_manager.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_block(
                    block=self.offload_manager.cuda_buffers[0],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_img_silu=temb_img_silu,
                    temb_txt_silu=temb_txt_silu,
                    image_rotary_emb=image_rotary_emb,
                    modulate_index=modulate_index,
                )

            self.offload_manager.swap_blocks()

        return hidden_states
