import torch

from lightx2v.common.offload.config import get_offload_granularity
from lightx2v.models.networks.longcat_image.infer.transformer_infer import LongCatImageTransformerInfer


class LongCatImageOffloadTransformerInfer(LongCatImageTransformerInfer):
    """Offload transformer inference for LongCat Image model.

    Supports block-level offload with double-buffer async prefetch for both
    double-stream blocks and single-stream blocks.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = get_offload_granularity(self.config)
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload

    def infer_with_blocks_offload(self, blocks, pre_infer_out):
        """Run transformer inference with block-level offload.

        Two-phase approach: first process all double blocks, then all single blocks,
        each with their own offload manager and cuda buffers.
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        temb = pre_infer_out.temb
        image_rotary_emb = pre_infer_out.image_rotary_emb
        image_rotary_positions = pre_infer_out.image_rotary_positions

        # For I2I task: concatenate output latents with input image latents
        output_seq_len = None
        if pre_infer_out.input_image_latents is not None:
            output_seq_len = pre_infer_out.output_seq_len
            hidden_states = torch.cat([hidden_states, pre_infer_out.input_image_latents], dim=0)

        def run_double_block(block_idx, block):
            nonlocal encoder_hidden_states, hidden_states
            self.block_idx = block_idx
            encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                image_rotary_positions,
            )
            return encoder_hidden_states, hidden_states

        self.run_blocks_with_offload(
            blocks.double_blocks,
            run_double_block,
        )

        def run_single_block(block_idx, block):
            nonlocal encoder_hidden_states, hidden_states
            self.block_idx = block_idx
            encoder_hidden_states, hidden_states = self.infer_single_stream_block(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                image_rotary_positions,
            )
            return encoder_hidden_states, hidden_states

        self.run_blocks_with_offload(
            blocks.single_blocks,
            run_single_block,
        )

        # For I2I task: only return output image latents
        if output_seq_len is not None:
            hidden_states = hidden_states[:output_seq_len]

        return hidden_states
