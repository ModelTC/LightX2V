import torch

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .triton_ops import (
    fuse_scale_shift_kernel,
)
from .utils import apply_rotary_emb_qwen, apply_wan_rope_with_flashinfer


def calculate_q_k_len(q, k_lens):
    q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
    return cu_seqlens_q, cu_seqlens_k


class ZImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.zero_cond_t = config.get("zero_cond_t", False)
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.seq_p_fp8_comm = False
        if self.config.get("modulate_type", "triton") == "triton":
            self.modulate_func = fuse_scale_shift_kernel
        else:
            self.modulate_func = lambda x, scale, shift: x * (1 + scale) + shift
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            self.apply_rope_func = apply_wan_rope_with_flashinfer
        else:
            self.apply_rope_func = apply_rotary_emb_qwen

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def apply_attn(self, block_weight, hidden_states, freqs_cis):
        is_3d = hidden_states.dim() == 3
        if is_3d:
            B, T, D = hidden_states.shape
            hidden_states_2d = hidden_states.reshape(-1, D)
            freqs_cis_2d = freqs_cis.reshape(-1, freqs_cis.shape[-1])
        else:
            hidden_states_2d = hidden_states
            freqs_cis_2d = freqs_cis

        query = block_weight.attention.to_q.apply(hidden_states_2d)
        key = block_weight.attention.to_k.apply(hidden_states_2d)
        value = block_weight.attention.to_v.apply(hidden_states_2d)

        query = query.unflatten(-1, (block_weight.attention.heads, -1))
        key = key.unflatten(-1, (block_weight.attention.heads, -1))
        value = value.unflatten(-1, (block_weight.attention.heads, -1))

        if block_weight.attention.norm_q is not None:
            query = block_weight.attention.norm_q.apply(query)
        if block_weight.attention.norm_k is not None:
            key = block_weight.attention.norm_k.apply(key)

        query, key = self.apply_rope_func(query, key, freqs_cis_2d)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        total_seq_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32, device="cpu").to(query.device, non_blocking=True)

        if self.config["seq_parallel"]:
            hidden_states_out = block_weight.attention.calculate_parallel.apply(
                q=query,
                k=key,
                v=value,
                slice_qkv_len=total_seq_len,
                cu_seqlens_qkv=cu_seqlens,
                attention_module=block_weight.attention.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                model_cls=self.config["model_cls"],
                img_first=False,
            )
        else:
            hidden_states_out = block_weight.attention.calculate.apply(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=total_seq_len,
                max_seqlen_kv=total_seq_len,
                model_cls="z_image",
            )

        output = block_weight.attention.to_out[0].apply(hidden_states_out)
        if len(block_weight.attention.to_out) > 1:
            output = block_weight.attention.to_out[1].apply(output)

        if is_3d:
            output = output.reshape(B, T, -1)

        return output

    def infer_block(
        self,
        block_weight,
        hidden_states,
        freqs_cis,
        adaln_input=None,
    ):
        if block_weight.modulation:
            assert adaln_input is not None
            mod_params = block_weight.adaLN_modulation.apply(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod_params.unsqueeze(1).chunk(4, dim=2)

            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            norm1_out = block_weight.attention_norm1.apply(hidden_states)
            scaled_norm1 = norm1_out * scale_msa

            # Attention block
            attn_out = self.apply_attn(
                block_weight=block_weight,
                hidden_states=scaled_norm1,
                freqs_cis=freqs_cis,
            )
            norm2_attn = block_weight.attention_norm2.apply(attn_out)

            hidden_states = hidden_states + gate_msa * norm2_attn

            ffn_norm1_out = block_weight.ffn_norm1.apply(hidden_states)
            scaled_ffn_norm1 = ffn_norm1_out * scale_mlp

            ffn_out = block_weight.feed_forward.forward(scaled_ffn_norm1)
            norm2_ffn = block_weight.ffn_norm2.apply(ffn_out)

            hidden_states = hidden_states + gate_mlp * norm2_ffn
        else:
            norm1_out = block_weight.attention_norm1.apply(hidden_states)

            # Attention block
            attn_out = self.apply_attn(
                block_weight=block_weight,
                hidden_states=norm1_out,
                freqs_cis=freqs_cis,
            )

            norm2_attn = block_weight.attention_norm2.apply(attn_out)
            hidden_states = hidden_states + norm2_attn

            # FFN block
            ffn_norm1_out = block_weight.ffn_norm1.apply(hidden_states)
            ffn_out = block_weight.feed_forward.forward(ffn_norm1_out)
            norm2_ffn = block_weight.ffn_norm2.apply(ffn_out)
            hidden_states = hidden_states + norm2_ffn

        # Clip to prevent overflow for fp16
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states

    def infer_calculating(
        self,
        block_weights,
        hidden_states,
        encoder_hidden_states,
        x_freqs_cis,
        cap_freqs_cis,
        adaln_input,
        x_item_seqlens,
        cap_item_seqlens,
    ):
        from torch.nn.utils.rnn import pad_sequence

        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # ==================== Stage 1: Noise Refiner (Image Stream) ====================
        # Process image stream with modulation
        if block_weights.noise_refiner is not None and len(block_weights.noise_refiner) > 0:
            # Build attention mask for image tokens
            x_max_seqlen = max(x_item_seqlens)
            x_attn_mask = torch.zeros((batch_size, x_max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(x_item_seqlens):
                x_attn_mask[i, :seq_len] = True

            # Process through noise_refiner layers (with modulation)
            # Use 3D [B, T, D] format to match official implementation
            for idx in range(len(block_weights.noise_refiner)):
                hidden_states = self.infer_block(
                    block_weight=block_weights.noise_refiner[idx],
                    hidden_states=hidden_states,
                    freqs_cis=x_freqs_cis,
                    adaln_input=adaln_input,
                )

        # ==================== Stage 2: Context Refiner (Text Stream) ====================
        # Process text stream without modulation
        if block_weights.context_refiner is not None and len(block_weights.context_refiner) > 0:
            # Build attention mask for text tokens
            cap_max_seqlen = max(cap_item_seqlens)
            cap_attn_mask = torch.zeros((batch_size, cap_max_seqlen), dtype=torch.bool, device=device)
            for i, seq_len in enumerate(cap_item_seqlens):
                cap_attn_mask[i, :seq_len] = True

            # Process through context_refiner layers (without modulation)
            # Use 3D [B, L, D] format to match official implementation
            for idx in range(len(block_weights.context_refiner)):
                encoder_hidden_states = self.infer_block(
                    block_weight=block_weights.context_refiner[idx],
                    hidden_states=encoder_hidden_states,  # [B, L, D]
                    freqs_cis=cap_freqs_cis,  # [B, L, D_rope]
                    adaln_input=None,  # No modulation for context_refiner
                )

        # ==================== Stage 3: Unified Layers (Merged Stream) ====================
        # Merge image and text streams
        unified_list = []
        unified_freqs_cis_list = []
        unified_item_seqlens = []

        for b in range(batch_size):
            x_len = x_item_seqlens[b]
            cap_len = cap_item_seqlens[b]

            # Concatenate image and text tokens: [image_tokens, text_tokens]
            unified_item = torch.cat(
                [
                    hidden_states[b, :x_len],
                    encoder_hidden_states[b, :cap_len],
                ],
                dim=0,
            )
            unified_list.append(unified_item)

            # Concatenate freqs_cis: [image_freqs, text_freqs]
            unified_freqs_item = torch.cat(
                [
                    x_freqs_cis[b, :x_len],
                    cap_freqs_cis[b, :cap_len],
                ],
                dim=0,
            )
            unified_freqs_cis_list.append(unified_freqs_item)

            unified_item_seqlens.append(x_len + cap_len)

        # Pad unified sequences
        unified_max_seqlen = max(unified_item_seqlens)
        unified = pad_sequence(unified_list, batch_first=True, padding_value=0.0)  # [B, max_seqlen, D]
        unified_freqs_cis = pad_sequence(unified_freqs_cis_list, batch_first=True, padding_value=0.0)  # [B, max_seqlen, D_rope]

        # Build attention mask for unified stream
        unified_attn_mask = torch.zeros((batch_size, unified_max_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = True

        # Process through unified layers (with modulation)
        # Use 3D [B, T_unified, D] format to match official implementation
        if block_weights.blocks is not None and len(block_weights.blocks) > 0:
            for idx in range(len(block_weights.blocks)):
                unified = self.infer_block(
                    block_weight=block_weights.blocks[idx],
                    hidden_states=unified,
                    freqs_cis=unified_freqs_cis,
                    adaln_input=adaln_input,
                )

        return unified

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        adaln_input = pre_infer_out.adaln_input
        x_item_seqlens = pre_infer_out.x_item_seqlens
        cap_item_seqlens = pre_infer_out.cap_item_seqlens

        # Use freqs_cis generated from position ids in pre_infer
        x_freqs_cis = pre_infer_out.x_freqs_cis
        cap_freqs_cis = pre_infer_out.cap_freqs_cis

        hidden_states = self.infer_calculating(
            block_weights=block_weights,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            x_freqs_cis=x_freqs_cis,
            cap_freqs_cis=cap_freqs_cis,
            adaln_input=adaln_input,
            x_item_seqlens=x_item_seqlens,
            cap_item_seqlens=cap_item_seqlens,
        )
        return hidden_states
