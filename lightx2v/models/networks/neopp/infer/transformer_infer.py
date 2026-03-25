import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.profiler import *


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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

        self.inv_freq_t = self._build_inv_freq(self.head_dim // 2, llm_config["rope_theta"])
        self.inv_freq_hw = self._build_inv_freq(self.head_dim // 4, llm_config["rope_theta_hw"])

    def _build_inv_freq(self, half_head_dim, theta):
        full_dim = half_head_dim * 2
        inv_freq_full = 1.0 / (theta ** (torch.arange(0, full_dim, 2, dtype=torch.float32) / full_dim))
        return inv_freq_full[::2]

    def _compute_rope(self, x, position_ids, inv_freq):
        inv_freq = inv_freq.to(x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, inputs):
        if self.scheduler.infer_condition:
            indexes = inputs["indexes_image_condition"]
            past_key_values = inputs["past_key_values_cond"]
        else:
            indexes = inputs["indexes_image_uncondition"]
            past_key_values = inputs["past_key_values_uncond"]

        hidden_states = pre_infer_out.image_embeds

        seq_len_q = hidden_states.shape[1]
        seq_len_k = past_key_values.shape[3] + seq_len_q
        self._cu_seqlens_q = torch.tensor([0, seq_len_q], dtype=torch.int32, device=hidden_states.device)
        self._cu_seqlens_k = torch.tensor([0, seq_len_k], dtype=torch.int32, device=hidden_states.device)
        self._max_seqlen_q = seq_len_q
        self._max_seqlen_k = seq_len_k

        for layer_idx, block_weight in enumerate(weights.blocks):
            hidden_states = self._decoder_layer(block_weight, layer_idx, hidden_states, indexes, past_key_values)

        hidden_states = weights.norm_mot_gen.apply(hidden_states.squeeze(0)).unsqueeze(0)
        hidden_states = self._fm_head(weights.fm_head, hidden_states.squeeze(0)).unsqueeze(0)
        return hidden_states

    def _decoder_layer(self, block_weight, layer_idx, hidden_states, indexes, past_key_values):
        residual = hidden_states
        hidden_states = block_weight.input_layernorm_mot_gen.apply(hidden_states.squeeze(0)).unsqueeze(0)

        hidden_states = self._self_attn(block_weight.self_attn, layer_idx, hidden_states, indexes, past_key_values)
        hidden_states = residual + hidden_states

        residual = hidden_states
        gen_hidden = block_weight.post_attention_layernorm_mot_gen.apply(hidden_states.squeeze(0))
        gen_hidden = self._sparse_moe(block_weight.mlp_mot_gen, gen_hidden)
        hidden_states = residual + gen_hidden.unsqueeze(0)

        return hidden_states

    def _self_attn(self, attn_w, layer_idx, hidden_states, indexes, past_key_values):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = attn_w.q_proj_mot_gen.apply(hidden_states.squeeze(0))
        query_states = query_states.view(hidden_shape)
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)

        query_states_t = attn_w.q_norm_mot_gen.apply(query_states_t.squeeze(0)).unsqueeze(0).transpose(1, 2)
        query_states_hw = attn_w.q_norm_hw_mot_gen.apply(query_states_hw.squeeze(0)).unsqueeze(0).transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = attn_w.k_proj_mot_gen.apply(hidden_states.squeeze(0))
        key_states = key_states.view(hidden_shape)
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)

        key_states_t = attn_w.k_norm_mot_gen.apply(key_states_t.squeeze(0)).unsqueeze(0).transpose(1, 2)
        key_states_hw = attn_w.k_norm_hw_mot_gen.apply(key_states_hw.squeeze(0)).unsqueeze(0).transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = attn_w.v_proj_mot_gen.apply(hidden_states.squeeze(0))
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self._compute_rope(hidden_states, indexes[0].unsqueeze(0), self.inv_freq_t)
        query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

        cos_h, sin_h = self._compute_rope(hidden_states, indexes[1].unsqueeze(0), self.inv_freq_hw)
        query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

        cos_w, sin_w = self._compute_rope(hidden_states, indexes[2].unsqueeze(0), self.inv_freq_hw)
        query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)

        past_k = past_key_values[layer_idx, 0].unsqueeze(0)
        past_v = past_key_values[layer_idx, 1].unsqueeze(0)

        key_states = torch.cat([past_k, key_states], dim=2)
        value_states = torch.cat([past_v, value_states], dim=2)

        attn_output = self._compute_attn(attn_w, query_states, key_states, value_states, input_shape)

        attn_output = attn_w.o_proj_mot_gen.apply(attn_output.squeeze(0)).unsqueeze(0)
        return attn_output

    def _compute_attn(self, attn_w, query_states, key_states, value_states, input_shape):
        q = query_states.squeeze(0).transpose(0, 1)
        k = key_states.squeeze(0).transpose(0, 1)
        v = value_states.squeeze(0).transpose(0, 1)

        attn_output = attn_w.cross_attn.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._cu_seqlens_q,
            cu_seqlens_kv=self._cu_seqlens_k,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_kv=self._max_seqlen_k,
        )
        return attn_output.unsqueeze(0)

    def _sparse_moe(self, moe_w, hidden_states):
        input_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)
        elif hidden_states.dim() == 2:
            flat = hidden_states
            hidden_dim = hidden_states.shape[-1]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {hidden_states.dim()}D")

        router_logits = moe_w.gate.apply(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        num_experts = moe_w.num_experts
        final_hidden_states = torch.zeros_like(flat)

        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            ei = expert_idx.item()
            expert_w = moe_w.experts[ei]
            idx, top_x = torch.where(expert_mask[ei])

            current_state = flat[None, top_x].reshape(-1, hidden_dim)
            gate_out = expert_w.gate_proj.apply(current_state)
            up_out = expert_w.up_proj.apply(current_state)
            current_hidden = expert_w.down_proj.apply(F.silu(gate_out) * up_out)
            current_hidden = current_hidden * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden.to(flat.dtype))

        return final_hidden_states.view(input_shape)

    def _fm_head(self, fm_head_w, hidden_states):
        hidden_states = fm_head_w.fm_head_0.apply(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = fm_head_w.fm_head_2.apply(hidden_states)
        return hidden_states
