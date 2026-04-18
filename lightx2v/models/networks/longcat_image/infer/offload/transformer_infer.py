import torch

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.longcat_image.infer.transformer_infer import LongCatImageTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class LongCatImageOffloadTransformerInfer(LongCatImageTransformerInfer):
    """Offload transformer inference for LongCat Image model.

    Supports block-level offload with double-buffer async prefetch for both
    double-stream blocks and single-stream blocks.
    """

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            if offload_granularity != "model":
                self.offload_manager_double = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                self.offload_manager_single = WeightAsyncStreamManager(offload_granularity=offload_granularity)

    def infer_with_blocks_offload(self, blocks, pre_infer_out):
        """Run transformer inference with block-level offload.

        Two-phase approach: first process all double blocks, then all single blocks,
        each with their own offload manager and cuda buffers.
        """
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        temb = pre_infer_out.temb
        image_rotary_emb = pre_infer_out.image_rotary_emb

        # For I2I task: concatenate output latents with input image latents
        output_seq_len = None
        if pre_infer_out.input_image_latents is not None:
            output_seq_len = pre_infer_out.output_seq_len
            hidden_states = torch.cat([hidden_states, pre_infer_out.input_image_latents], dim=0)

        # Stage 1: double blocks offload
        # wait for default stream
        current_stream = torch_device_module.current_stream()
        self.offload_manager_double.compute_stream.wait_stream(current_stream)
        for block_idx in range(len(blocks.double_blocks)):
            self.block_idx = block_idx

            if self.offload_manager_double.need_init_first_buffer:
                self.offload_manager_double.init_first_buffer(blocks.double_blocks)

            self.offload_manager_double.prefetch_weights((block_idx + 1) % len(blocks.double_blocks), blocks.double_blocks)

            with torch_device_module.stream(self.offload_manager_double.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_double_stream_block(
                    self.offload_manager_double.cuda_buffers[0],
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            self.offload_manager_double.swap_blocks()

        # Stage 2: single blocks offload
        # wait for double stream
        self.offload_manager_single.compute_stream.wait_stream(self.offload_manager_double.compute_stream)
        for block_idx in range(len(blocks.single_blocks)):
            self.block_idx = block_idx

            if self.offload_manager_single.need_init_first_buffer:
                self.offload_manager_single.init_first_buffer(blocks.single_blocks)

            self.offload_manager_single.prefetch_weights((block_idx + 1) % len(blocks.single_blocks), blocks.single_blocks)

            with torch_device_module.stream(self.offload_manager_single.compute_stream):
                encoder_hidden_states, hidden_states = self.infer_single_stream_block(
                    self.offload_manager_single.cuda_buffers[0],
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                )

            self.offload_manager_single.swap_blocks()

        # For I2I task: only return output image latents
        if output_seq_len is not None:
            hidden_states = hidden_states[:output_seq_len]

        return hidden_states
