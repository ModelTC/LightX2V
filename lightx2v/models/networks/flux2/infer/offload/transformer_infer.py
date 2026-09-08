import torch
import torch.nn.functional as F

from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.flux2.infer.transformer_infer import Flux2TransformerInfer


class Flux2OffloadTransformerInfer(Flux2TransformerInfer):
    """Flux2 transformer inference with block-level CPU offload."""

    def __init__(self, config):
        super().__init__(config)
        if not config.get("cpu_offload", False):
            self.infer_func = super().infer
            return

        offload_granularity = get_offload_granularity(config)
        if offload_granularity == "model":
            self.infer_func = super().infer
            return
        if offload_granularity != "block":
            raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")

        self.infer_func = self.infer_with_blocks_offload

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

        def run_double(block_idx, block):
            nonlocal encoder_hidden_states, hidden_states
            self.block_idx = block_idx
            encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                image_rotary_emb,
                image_rotary_positions,
            )
            return encoder_hidden_states, hidden_states

        self.run_blocks_with_offload(
            block_weights.double_blocks,
            run_double,
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        def run_single(block_idx, block):
            nonlocal hidden_states
            self.block_idx = block_idx
            hidden_states = self.infer_single_stream_block(
                block,
                hidden_states,
                None,
                single_stream_mod,
                image_rotary_emb,
                image_rotary_positions,
                num_txt_tokens=num_txt_tokens,
            )
            return hidden_states

        self.run_blocks_with_offload(
            block_weights.single_blocks,
            run_single,
        )
        return hidden_states[num_txt_tokens:, ...]

    def infer(self, block_weights, pre_infer_out):
        return self.infer_func(block_weights, pre_infer_out)
