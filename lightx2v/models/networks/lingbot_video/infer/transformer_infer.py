import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


class LingBotVideoTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.compute_dtype = GET_DTYPE()
        self.hidden_size = int(config.get("hidden_size", 2048))
        self.num_heads = int(config.get("num_attention_heads", 16))
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = int(config.get("num_experts", 128))
        self.top_k = int(config.get("num_experts_per_tok", 8))
        self.score_func = config.get("score_func", "sigmoid")
        self.norm_topk_prob = bool(config.get("norm_topk_prob", True))
        self.n_group = config.get("n_group", 4)
        self.topk_group = config.get("topk_group", 2)
        self.route_scale = float(config.get("routed_scaling_factor", 2.5))
        self.init_compile(config)

    def _attention(self, weights, hidden_states, rotary_emb):
        q = weights.attn.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.attn.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.attn.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))

        q, k = weights.attn.rope.apply(
            weights.attn.norm_q.apply(q),
            weights.attn.norm_k.apply(k),
            rotary_emb,
        )
        seq_len = q.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=q.device)
        hidden_states = weights.attn.calculate.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
        )
        return weights.attn.to_out.apply(hidden_states.to(dtype=self.compute_dtype))

    def _dense_mlp(self, weights, hidden_states):
        gate = weights.gate_proj.apply(hidden_states)
        up = weights.up_proj.apply(hidden_states)
        return weights.down_proj.apply(F.silu(gate) * up)

    def _group_limited_topk(self, scores_for_choice):
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // int(self.n_group)
        grouped = scores_for_choice.view(seq_len, int(self.n_group), experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=int(self.topk_group), dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(seq_len, int(self.n_group), experts_per_group).reshape(seq_len, -1)
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def _route(self, weights, tokens):
        logits = weights.router.weight.apply(tokens.float())
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = logits.sigmoid()
        scores_for_choice = scores + weights.router.e_score_correction_bias.tensor.unsqueeze(0)
        if self.n_group is not None and int(self.n_group) > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype)

    def _moe(self, weights, hidden_states):
        top_indices, top_scores = self._route(weights, hidden_states)
        output = weights.fused_moe.apply(hidden_states, top_indices, top_scores)
        if getattr(weights, "shared_experts", None) is not None:
            output = output + self._dense_mlp(weights.shared_experts, hidden_states)
        return output

    def _ffn(self, weights, hidden_states):
        if getattr(weights.ffn, "use_moe", False):
            return self._moe(weights.ffn, hidden_states)
        return self._dense_mlp(weights.ffn.dense, hidden_states)

    def infer_block(self, weights, hidden_states, temb6, rotary_emb):
        mod = temb6 + weights.scale_shift_table.tensor.squeeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        attn_in = (weights.norm1.apply(hidden_states) * scale_msa + shift_msa).to(self.compute_dtype)
        attn_out = self._attention(weights, attn_in, rotary_emb)
        hidden_states = hidden_states + (gate_msa * weights.norm_post_attn.apply(attn_out)).to(hidden_states.dtype)

        ffn_in = (weights.norm2.apply(hidden_states) * scale_mlp + shift_mlp).to(self.compute_dtype)
        ffn_out = self._ffn(weights, ffn_in)
        hidden_states = hidden_states + (gate_mlp * weights.norm_post_ffn.apply(ffn_out)).to(hidden_states.dtype)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        for block_idx, block in enumerate(block_weights.blocks):
            hidden_states = self.run_block(block_idx, block, hidden_states, pre_infer_out.temb6, pre_infer_out.rotary_emb)
        return hidden_states
