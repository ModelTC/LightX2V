import torch
import torch.nn.functional as F

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.qwen_image.infer.transformer_infer import QwenImageTransformerInfer, apply_attn


class QwenImageOffloadTransformerInfer(QwenImageTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.phases_num = 3
        self.num_blocks = config["num_layers"]
        if self.config.get("cpu_offload", False):
            if "offload_ratio" in self.config:
                self.offload_ratio = self.config["offload_ratio"]
            else:
                self.offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_blocks_offload
                else:
                    assert NotImplementedError
            elif offload_granularity == "phase":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self.infer_with_phases_offload
                    self.clean_cuda_cache = True
                else:
                    assert NotImplementedError
                self.phase_params = {
                    "img_mod2": None,
                    "txt_mod2": None,
                    "img_modulated": None,
                    "img_gate1": None,
                    "txt_modulated": None,
                    "txt_gate1": None,
                    "attn_output": None,
                }
            else:
                assert NotImplementedError

            if offload_granularity != "model":
                self.weights_stream_mgr = WeightAsyncStreamManager(blocks_num=self.num_blocks, offload_ratio=self.offload_ratio, phases_num=self.phases_num)
            else:
                assert NotImplementedError

    def infer_with_blocks_offload(self, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        for block_idx in range(self.num_blocks):
            self.block_idx = block_idx
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = block_weights.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < self.num_blocks - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, block_weights.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_block(
                    block_weight=block_weights.blocks[block_idx], hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb
                )

            self.weights_stream_mgr.swap_weights()
        return encoder_hidden_states, hidden_states

    def infer_with_phases_offload(self, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        for block_idx in range(self.num_blocks):
            self.block_idx = block_idx
            encoder_hidden_states, hidden_states = self.infer_phases(block_idx, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        if self.clean_cuda_cache:
            self.clear_offload_params(temb, image_rotary_emb)

        return encoder_hidden_states, hidden_states

    def infer_phases(self, block_idx, block_weights, hidden_states, encoder_hidden_states, temb, image_rotary_emb, lazy=False):
        for phase_idx in range(self.phases_num):
            if block_idx == 0 and phase_idx == 0:
                if lazy:
                    assert NotImplementedError
                else:
                    phase = block_weights.blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_phase(
                    phase_idx,
                    block_weights.blocks[block_idx],
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

            is_last_phase = block_idx == self.num_blocks - 1 and phase_idx == self.phases_num - 1
            if not is_last_phase:
                next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                next_phase_idx = (phase_idx + 1) % self.phases_num
                self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, block_weights.blocks)

            self.weights_stream_mgr.swap_phases()
        return encoder_hidden_states, hidden_states

    def infer_phase(self, cur_phase_idx, block_weight, hidden_states, encoder_hidden_states, temb, image_rotary_emb):
        cur_phase = block_weight.compute_phases[cur_phase_idx]

        if cur_phase_idx == 0:
            # Get modulation parameters for both streams
            img_mod_params = cur_phase.img_mod.apply(F.silu(temb))
            txt_mod_params = cur_phase.txt_mod.apply(F.silu(temb))

            # Split modulation parameters for norm1 and norm2
            img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
            txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

            # Process image stream - norm1 + modulation
            img_normed = cur_phase.img_norm1.apply(hidden_states)
            img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

            # Process text stream - norm1 + modulation
            txt_normed = cur_phase.txt_norm1.apply(encoder_hidden_states)
            txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

            self.phase_params["img_mod2"] = img_mod2
            self.phase_params["txt_mod2"] = txt_mod2
            self.phase_params["img_modulated"] = img_modulated
            self.phase_params["img_gate1"] = img_gate1
            self.phase_params["txt_modulated"] = txt_modulated
            self.phase_params["txt_gate1"] = txt_gate1

        elif cur_phase_idx == 1:
            attn_output = apply_attn(
                block_weight=cur_phase,
                hidden_states=self.phase_params["img_modulated"],  # Image stream (will be processed as "sample")
                encoder_hidden_states=self.phase_params["txt_modulated"],  # Text stream (will be processed as "context")
                image_rotary_emb=image_rotary_emb,
            )
            self.phase_params["attn_output"] = attn_output

        elif cur_phase_idx == 2:
            # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
            img_attn_output, txt_attn_output = self.phase_params["attn_output"]

            # Apply attention gates and add residual (like in Megatron)
            hidden_states = hidden_states + self.phase_params["img_gate1"] * img_attn_output
            encoder_hidden_states = encoder_hidden_states + self.phase_params["txt_gate1"] * txt_attn_output

            # Process image stream - norm2 + MLP
            img_normed2 = cur_phase.img_norm2.apply(hidden_states)
            img_modulated2, img_gate2 = self._modulate(img_normed2, self.phase_params["img_mod2"])
            img_mlp_output = F.silu(cur_phase.img_mlp.mlp_0.apply(img_modulated2.squeeze(0)))
            img_mlp_output = cur_phase.img_mlp.mlp_2.apply(img_mlp_output)
            hidden_states = hidden_states + img_gate2 * img_mlp_output

            # Process text stream - norm2 + MLP
            txt_normed2 = cur_phase.txt_norm2.apply(encoder_hidden_states)
            txt_modulated2, txt_gate2 = self._modulate(txt_normed2, self.phase_params["txt_mod2"])
            txt_mlp_output = F.silu(cur_phase.txt_mlp.mlp_0.apply(txt_modulated2.squeeze(0)))
            txt_mlp_output = cur_phase.txt_mlp.mlp_2.apply(txt_mlp_output)
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

            # Clip to prevent overflow for fp16
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def clear_offload_params(self):
        del (
            self.phase_params["img_mod2"],
            self.phase_params["txt_mod2"],
            self.phase_params["img_modulated"],
            self.phase_params["img_gate1"],
            self.phase_params["txt_modulated"],
            self.phase_params["txt_gate1"],
            self.phase_params["attn_output"],
        )
