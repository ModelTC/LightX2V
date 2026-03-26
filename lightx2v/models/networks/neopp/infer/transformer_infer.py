import torch
import torch.nn.functional as F
from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.profiler import *


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.squeeze(0).unsqueeze(1)  # [seq, 1, head_dim] — broadcast over num_heads
    sin = sin.squeeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NeoppTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        llm_config = config["llm_config"]
        self.num_layers = llm_config["num_hidden_layers"]
        self.hidden_size = llm_config["hidden_size"]
        self.num_heads = llm_config["num_attention_heads"]
        self.num_kv_heads = llm_config["num_key_value_heads"]
        self.head_dim = llm_config["head_dim"]
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.num_experts_per_tok = llm_config["num_experts_per_tok"]
        self.norm_topk_prob = llm_config.get("norm_topk_prob", True)

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, inputs):
        if self.scheduler.infer_condition:
            past_key_values = inputs["past_key_values_cond"]  # torch.Size([48, 2, past_seq, num_kv_heads, head_dim])
            cos_sin = inputs["cos_sin_cond"]
        else:
            past_key_values = inputs["past_key_values_uncond"]
            cos_sin = inputs["cos_sin_uncond"]

        hidden_states = pre_infer_out.image_embeds  # torch.Size([1, 1032, 2048])

        seq_len_q = hidden_states.shape[1]
        seq_len_k = past_key_values.shape[2] + seq_len_q
        self._cu_seqlens_q = torch.tensor([0, seq_len_q], dtype=torch.int32, device=hidden_states.device)
        self._cu_seqlens_k = torch.tensor([0, seq_len_k], dtype=torch.int32, device=hidden_states.device)
        self._max_seqlen_q = seq_len_q
        self._max_seqlen_k = seq_len_k

        for layer_idx, block_weight in enumerate(weights.blocks):
            hidden_states = self._decoder_layer(block_weight, layer_idx, hidden_states, cos_sin, past_key_values)

        hidden_states = weights.norm_mot_gen.apply(hidden_states.squeeze(0)).unsqueeze(0)
        hidden_states = self._fm_head(weights.fm_head, hidden_states.squeeze(0)).unsqueeze(0)
        return hidden_states

    def _decoder_layer(self, block_weight, layer_idx, hidden_states, cos_sin, past_key_values):
        residual = hidden_states
        hidden_states = block_weight.input_layernorm_mot_gen.apply(hidden_states.squeeze(0)).unsqueeze(0)

        hidden_states = self._self_attn(block_weight.self_attn, layer_idx, hidden_states, cos_sin, past_key_values)
        hidden_states = residual + hidden_states

        residual = hidden_states
        gen_hidden = block_weight.post_attention_layernorm_mot_gen.apply(hidden_states.squeeze(0))
        gen_hidden = self._sparse_moe(block_weight.mlp_mot_gen, gen_hidden)
        hidden_states = residual + gen_hidden.unsqueeze(0)

        return hidden_states

    # @ProfilingContext4DebugL1("Self Attn")
    def _self_attn(self, attn_w, layer_idx, hidden_states, cos_sin, past_key_values):
        cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = cos_sin

        query_states = attn_w.q_proj_mot_gen.apply(hidden_states.squeeze(0))
        query_states = query_states.view(-1, self.num_heads, self.head_dim)  # [seq, num_heads, head_dim]
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)

        query_states_t = attn_w.q_norm_mot_gen.apply(query_states_t)  # [seq, num_heads, head_dim//2]
        query_states_hw = attn_w.q_norm_hw_mot_gen.apply(query_states_hw)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = attn_w.k_proj_mot_gen.apply(hidden_states.squeeze(0))
        key_states = key_states.view(-1, self.num_kv_heads, self.head_dim)  # [seq, num_kv_heads, head_dim]
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)

        key_states_t = attn_w.k_norm_mot_gen.apply(key_states_t)
        key_states_hw = attn_w.k_norm_hw_mot_gen.apply(key_states_hw)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = attn_w.v_proj_mot_gen.apply(hidden_states.squeeze(0))
        value_states = value_states.view(-1, self.num_kv_heads, self.head_dim)  # [seq, num_kv_heads, head_dim]

        query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)
        query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)
        query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        past_k = past_key_values[layer_idx, 0]  # [past_seq, num_kv_heads, head_dim]
        past_v = past_key_values[layer_idx, 1]

        key_states = torch.cat([past_k, key_states], dim=0)  # [total_seq, num_kv_heads, head_dim]
        value_states = torch.cat([past_v, value_states], dim=0)

        attn_output = self._compute_attn(attn_w, query_states, key_states, value_states)

        attn_output = attn_w.o_proj_mot_gen.apply(attn_output.squeeze(0)).unsqueeze(0)
        return attn_output

    # @ProfilingContext4DebugL1("Compute Attn")
    def _compute_attn(self, attn_w, query_states, key_states, value_states):
        attn_output = attn_w.cross_attn.apply(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=self._cu_seqlens_q,
            cu_seqlens_kv=self._cu_seqlens_k,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_kv=self._max_seqlen_k,
        )
        return attn_output.unsqueeze(0)

    # @ProfilingContext4DebugL1("Sparse MoE")
    def _sparse_moe(self, moe_w, hidden_states):
        router_logits = moe_w.gate.apply(hidden_states)
        if self.norm_topk_prob:
            _, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1, sorted=False)
            routing_weights = F.softmax(router_logits.float().gather(1, selected_experts), dim=-1)
        else:
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)

        output = flashinfer_cutlass_fused_moe(
            hidden_states.contiguous(),
            selected_experts.to(torch.int32),
            routing_weights,
            moe_w._fi_fc1_weight,
            moe_w._fi_fc2_weight,
            hidden_states.dtype,
            quant_scales=None,
        )[0]

        return output

    # @ProfilingContext4DebugL1("FM Head")
    def _fm_head(self, fm_head_w, hidden_states):
        hidden_states = fm_head_w.fm_head_0.apply(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = fm_head_w.fm_head_2.apply(hidden_states)
        return hidden_states
