import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.flux2.infer.transformer_infer import Flux2TransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class Flux2OffloadTransformerInfer(Flux2TransformerInfer):
    """Flux2 transformer inference with block-level CPU offload."""

    def __init__(self, config):
        super().__init__(config)
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
                self.offload_manager_double = WeightAsyncStreamManager(offload_granularity=offload_granularity)
                self.offload_manager_single = WeightAsyncStreamManager(offload_granularity=offload_granularity)
            elif offload_granularity == "model":
                self.infer_func = super().infer
            else:
                raise ValueError(f"Unsupported offload_granularity: {offload_granularity}")
        else:
            self.infer_func = super().infer

    def infer_with_blocks_offload(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        timestep = pre_infer_out.timestep
        image_rotary_emb = pre_infer_out.image_rotary_emb

        num_txt_tokens = encoder_hidden_states.shape[0]

        if self.seq_p_group is not None and image_rotary_emb is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)

            if isinstance(image_rotary_emb, tuple):
                freqs_cos, freqs_sin = image_rotary_emb

                txt_cos = freqs_cos[:num_txt_tokens]
                img_cos = freqs_cos[num_txt_tokens:]
                txt_sin = freqs_sin[:num_txt_tokens]
                img_sin = freqs_sin[num_txt_tokens:]

                seqlen = img_cos.shape[0]
                padding_size = (world_size - (seqlen % world_size)) % world_size
                if padding_size > 0:
                    img_cos = F.pad(img_cos, (0, 0, 0, padding_size))
                    img_sin = F.pad(img_sin, (0, 0, 0, padding_size))
                img_cos = torch.chunk(img_cos, world_size, dim=0)[cur_rank]
                img_sin = torch.chunk(img_sin, world_size, dim=0)[cur_rank]

                freqs_cos = torch.cat([txt_cos, img_cos], dim=0)
                freqs_sin = torch.cat([txt_sin, img_sin], dim=0)
                image_rotary_emb = (freqs_cos, freqs_sin)
            else:
                txt_emb = image_rotary_emb[:num_txt_tokens]
                img_emb = image_rotary_emb[num_txt_tokens:]

                seqlen = img_emb.shape[0]
                padding_size = (world_size - (seqlen % world_size)) % world_size
                if padding_size > 0:
                    img_emb = F.pad(img_emb, (0, 0, 0, padding_size))
                img_emb = torch.chunk(img_emb, world_size, dim=0)[cur_rank]

                image_rotary_emb = torch.cat([txt_emb, img_emb], dim=0)

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
                    num_txt_tokens=num_txt_tokens,
                )

            self.offload_manager_single.swap_blocks()

        hidden_states = hidden_states[num_txt_tokens:, ...]
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        return self.infer_func(block_weights, pre_infer_out)
