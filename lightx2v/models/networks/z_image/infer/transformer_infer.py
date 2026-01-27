import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer

from .utils import apply_rotary_emb_qwen, apply_wan_rope_with_flashinfer


class ZImageTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.infer_conditional = True
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.attn_type = config.get("attn_type", "flash_attn3")
        self.zero_cond_t = config.get("zero_cond_t", False)
        self.n_heads = config.get("n_heads", config.get("num_attention_heads", 24))
        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None
        self.seq_p_fp8_comm = False
        if self.config.get("rope_type", "flashinfer") == "flashinfer":
            self.apply_rope_func = apply_wan_rope_with_flashinfer
        else:
            self.apply_rope_func = apply_rotary_emb_qwen

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer_mod(self, mod_phase, hidden_states, adaln_input):
        if mod_phase is None:
            return None, None, None, None

        mod_params = mod_phase.adaLN_modulation.apply(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod_params.chunk(4, dim=-1)
        gate_msa.tanh_()
        gate_mlp.tanh_()

        scale_msa.add_(1.0)
        scale_mlp.add_(1.0)

        return scale_msa, gate_msa, scale_mlp, gate_mlp

    def infer_attn(self, attn_phase, hidden_states, freqs_cis, scale_msa=None):
        norm1_out = attn_phase.attention_norm1.apply(hidden_states)
        if scale_msa is not None:
            scaled_norm1 = norm1_out * scale_msa
        else:
            scaled_norm1 = norm1_out

        query = attn_phase.to_q.apply(scaled_norm1)
        key = attn_phase.to_k.apply(scaled_norm1)
        value = attn_phase.to_v.apply(scaled_norm1)
        head_dim = query.shape[-1] // self.n_heads
        query = query.unflatten(-1, (self.n_heads, head_dim))
        key = key.unflatten(-1, (self.n_heads, head_dim))
        value = value.unflatten(-1, (self.n_heads, head_dim))

        if attn_phase.norm_q is not None:
            query = attn_phase.norm_q.apply(query)
        if attn_phase.norm_k is not None:
            key = attn_phase.norm_k.apply(key)

        query, key = self.apply_rope_func(query, key, freqs_cis)

        total_seq_len = query.shape[0]
        cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32, device="cpu").to(query.device, non_blocking=True)

        if self.config["seq_parallel"]:
            hidden_states_out = attn_phase.calculate_parallel.apply(
                q=query,
                k=key,
                v=value,
                slice_qkv_len=total_seq_len,
                cu_seqlens_qkv=cu_seqlens,
                attention_module=attn_phase.calculate,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                img_first=False,
            )
        else:
            # todo
            hidden_states_out = attn_phase.calculate.apply(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=total_seq_len,
                max_seqlen_kv=total_seq_len,
            )

        output = attn_phase.to_out[0].apply(hidden_states_out)
        if len(attn_phase.to_out) > 1:
            output = attn_phase.to_out[1].apply(output)

        attn_out = attn_phase.attention_norm2.apply(output)

        return attn_out

    def infer_ffn(self, ffn_phase, hidden_states, scale_mlp=None, gate_mlp=None):
        ffn_norm1_out = ffn_phase.ffn_norm1.apply(hidden_states)
        if scale_mlp is not None:
            ffn_norm1_out.mul_(scale_mlp)
        w1_out = ffn_phase.w1.apply(ffn_norm1_out)
        w3_out = ffn_phase.w3.apply(ffn_norm1_out)
        silu_gated = F.silu(w1_out) * w3_out
        ffn_out = ffn_phase.w2.apply(silu_gated)
        norm2_ffn = ffn_phase.ffn_norm2.apply(ffn_out)

        return norm2_ffn, gate_mlp

    def infer_block(
        self,
        block_weight,
        hidden_states,
        freqs_cis,
        adaln_input=None,
    ):
        mod_phase = block_weight.compute_phases[0] if block_weight.has_modulation else None
        attn_phase = block_weight.compute_phases[1]
        ffn_phase = block_weight.compute_phases[2]

        scale_msa, gate_msa, scale_mlp, gate_mlp = self.infer_mod(mod_phase, attn_phase, hidden_states, adaln_input)
        attn_out = self.infer_attn(attn_phase, hidden_states, freqs_cis, scale_msa)

        if gate_msa is not None:
            hidden_states.add_(gate_msa * attn_out)
        else:
            hidden_states.add_(attn_out)
        norm2_ffn, gate_mlp = self.infer_ffn(ffn_phase, hidden_states, scale_mlp, gate_mlp)

        if gate_mlp is not None:
            hidden_states.add_(gate_mlp * norm2_ffn)
        else:
            hidden_states.add_(norm2_ffn)

        if hidden_states.dtype == torch.float16:
            hidden_states.clip_(-65504, 65504)

        return hidden_states

    def infer_noise_refiner(
        self,
        noise_refiner_blocks,
        hidden_states,
        x_freqs_cis,
        adaln_input,
        x_len,
    ):
        x_hidden = hidden_states[:x_len]
        x_freqs = x_freqs_cis[:x_len]
        for block_weight in noise_refiner_blocks:
            x_hidden = self.infer_block(
                block_weight=block_weight,
                hidden_states=x_hidden,
                freqs_cis=x_freqs,
                adaln_input=adaln_input,
            )

        return x_hidden

    def infer_context_refiner(
        self,
        context_refiner_blocks,
        encoder_hidden_states,
        cap_freqs_cis,
        cap_len,
    ):
        cap_hidden = encoder_hidden_states[:cap_len]
        cap_freqs = cap_freqs_cis[:cap_len]
        for block_weight in context_refiner_blocks:
            cap_hidden = self.infer_block(
                block_weight=block_weight,
                hidden_states=cap_hidden,
                freqs_cis=cap_freqs,
                adaln_input=None,
            )

        return cap_hidden

    def infer_main_blocks(
        self,
        main_blocks,
        hidden_states,
        encoder_hidden_states,
        x_freqs_cis,
        cap_freqs_cis,
        adaln_input,
        x_len,
        cap_len,
    ):
        unified = torch.cat([hidden_states, encoder_hidden_states], dim=0)
        unified_freqs_cis = torch.cat([x_freqs_cis[:x_len], cap_freqs_cis[:cap_len]], dim=0)
        for block_weight in main_blocks:
            unified = self.infer_block(
                block_weight=block_weight,
                hidden_states=unified,
                freqs_cis=unified_freqs_cis,
                adaln_input=adaln_input,
            )

        return unified

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
        x_len = x_item_seqlens[0]
        cap_len = cap_item_seqlens[0]

        # Stage 1: Noise Refiner (Image Stream)
        hidden_states = self.infer_noise_refiner(
            noise_refiner_blocks=block_weights.noise_refiner,
            hidden_states=hidden_states,
            x_freqs_cis=x_freqs_cis,
            adaln_input=adaln_input,
            x_len=x_len,
        )

        # Stage 2: Context Refiner (Text Stream)
        encoder_hidden_states = self.infer_context_refiner(
            context_refiner_blocks=block_weights.context_refiner,
            encoder_hidden_states=encoder_hidden_states,
            cap_freqs_cis=cap_freqs_cis,
            cap_len=cap_len,
        )

        # Stage 3: Main Blocks (Unified Stream)
        unified = self.infer_main_blocks(
            main_blocks=block_weights.blocks,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            x_freqs_cis=x_freqs_cis,
            cap_freqs_cis=cap_freqs_cis,
            adaln_input=adaln_input,
            x_len=x_len,
            cap_len=cap_len,
        )

        return unified

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        encoder_hidden_states = pre_infer_out.encoder_hidden_states
        adaln_input = pre_infer_out.adaln_input
        x_item_seqlens = pre_infer_out.x_item_seqlens
        cap_item_seqlens = pre_infer_out.cap_item_seqlens
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
