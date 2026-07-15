import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.lingbot_video.infer.utils import apply_rotary_emb
from lightx2v.utils.envs import GET_DTYPE


def _round_up_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


class LingBotVideoTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
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

    def _attention(self, weights, hidden_states, rotary_emb):
        q = weights.attn.to_q.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        k = weights.attn.to_k.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        v = weights.attn.to_v.apply(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))

        q = apply_rotary_emb(weights.attn.norm_q.apply(q), rotary_emb)
        k = apply_rotary_emb(weights.attn.norm_k.apply(k), rotary_emb)
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
        return weights.attn.to_out.apply(hidden_states.to(dtype=GET_DTYPE()))

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

    @staticmethod
    def _reorder_tokens(tokens, top_scores, top_indices, num_experts):
        num_tokens = tokens.shape[0]
        top_k = top_indices.shape[1]
        flat_scores = top_scores.reshape(-1)
        flat_indices = top_indices.reshape(-1)
        active_positions = torch.where(flat_scores != 0)[0]
        active_experts = flat_indices[active_positions]

        counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
        counts.scatter_add_(0, active_experts, torch.ones_like(active_experts, dtype=torch.int64))

        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        sorted_scores = flat_scores[sorted_positions]
        original_token_idx = sorted_positions // top_k
        permuted_tokens = tokens[original_token_idx]
        return permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k

    @staticmethod
    def _pad_grouped_tokens(tokens, counts, align=8):
        num_tokens = tokens.shape[0]
        num_experts = int(counts.shape[0])
        max_len = _round_up_to_multiple(num_tokens + num_experts * align, align)
        counts_i64 = counts.to(torch.int64)
        total_per_expert = torch.clamp_min(counts_i64, align)
        aligned_counts_i64 = (total_per_expert + align - 1) // align * align
        write_offsets = torch.cumsum(aligned_counts_i64, dim=0) - aligned_counts_i64
        end_offsets = torch.cumsum(aligned_counts_i64, dim=0)
        start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

        slots = torch.arange(max_len, dtype=torch.int64, device=tokens.device)
        expert_idx = torch.bucketize(slots, end_offsets, right=True)
        valid_expert = expert_idx < num_experts
        safe_expert_idx = expert_idx.clamp(max=num_experts - 1)
        local_idx = slots - write_offsets[safe_expert_idx]
        source_idx = start_indices[safe_expert_idx] + local_idx
        valid = valid_expert & (local_idx < counts_i64[safe_expert_idx])
        fill = torch.full_like(source_idx, num_tokens)
        permuted_indices = torch.where(valid, source_idx, fill)

        tokens_with_pad = torch.vstack((tokens, tokens.new_zeros((tokens.shape[-1],))))
        input_shape = tokens_with_pad.shape
        return input_shape, tokens_with_pad[permuted_indices], permuted_indices, aligned_counts_i64.to(torch.int32)

    @staticmethod
    def _unpad_grouped_tokens(output, input_shape, permuted_indices):
        unpermuted = output.new_empty(input_shape)
        unpermuted[permuted_indices, :] = output
        return unpermuted[:-1]

    def _run_experts_for_loop(self, weights, tokens, counts):
        count_list = counts.tolist()
        splits = torch.split(tokens, count_list, dim=0)
        outputs = []
        for expert_idx, expert_tokens in enumerate(splits):
            if expert_tokens.numel() == 0:
                continue
            h = F.silu(expert_tokens @ weights.experts.w1.tensor[expert_idx].transpose(-2, -1))
            h = h * (expert_tokens @ weights.experts.w3.tensor[expert_idx].transpose(-2, -1))
            h = h @ weights.experts.w2.tensor[expert_idx].transpose(-2, -1)
            outputs.append(h)
        if not outputs:
            return tokens.new_zeros(tokens.shape)
        return torch.cat(outputs, dim=0)

    def _run_grouped_experts(self, weights, tokens, counts):
        if not hasattr(torch, "_grouped_mm"):
            return self._run_experts_for_loop(weights, tokens, counts)
        input_shape, padded_tokens, permuted_indices, aligned_counts = self._pad_grouped_tokens(tokens, counts)
        offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
        h = F.silu(
            torch._grouped_mm(
                padded_tokens.bfloat16(),
                weights.experts.w1.tensor.bfloat16().transpose(-2, -1),
                offs=offsets,
            )
        )
        h = h * torch._grouped_mm(
            padded_tokens.bfloat16(),
            weights.experts.w3.tensor.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        out = torch._grouped_mm(
            h,
            weights.experts.w2.tensor.bfloat16().transpose(-2, -1),
            offs=offsets,
        ).type_as(padded_tokens)
        return self._unpad_grouped_tokens(out, input_shape, permuted_indices)

    @staticmethod
    def _restore_tokens(expert_output, sorted_positions, sorted_scores, num_tokens, top_k):
        dim = expert_output.shape[-1]
        unsorted = torch.zeros((num_tokens * top_k, dim), dtype=expert_output.dtype, device=expert_output.device)
        unsorted[sorted_positions] = expert_output
        unsorted = unsorted.reshape(num_tokens, top_k, dim)

        scores_unsorted = torch.zeros(num_tokens * top_k, dtype=sorted_scores.dtype, device=sorted_scores.device)
        scores_unsorted[sorted_positions] = sorted_scores
        scores_unsorted = scores_unsorted.reshape(num_tokens, top_k, 1)
        return (unsorted.float() * scores_unsorted).sum(dim=1).to(expert_output.dtype)

    def _moe(self, weights, hidden_states):
        top_indices, top_scores = self._route(weights, hidden_states)
        permuted_tokens, counts, sorted_positions, sorted_scores, num_tokens, top_k = self._reorder_tokens(
            hidden_states,
            top_scores,
            top_indices,
            self.num_experts,
        )
        expert_output = self._run_grouped_experts(weights, permuted_tokens, counts)
        output = self._restore_tokens(expert_output, sorted_positions, sorted_scores, num_tokens, top_k)
        if getattr(weights, "shared_experts", None) is not None:
            output = output + self._dense_mlp(weights.shared_experts, hidden_states)
        return output

    def _ffn(self, weights, hidden_states):
        if getattr(weights.ffn, "use_moe", False):
            return self._moe(weights.ffn, hidden_states)
        return self._dense_mlp(weights.ffn.dense, hidden_states)

    def _block(self, weights, hidden_states, temb6, rotary_emb):
        mod = temb6 + weights.scale_shift_table.tensor.squeeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        bulk_dtype = GET_DTYPE()
        attn_in = (weights.norm1.apply(hidden_states) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self._attention(weights, attn_in, rotary_emb)
        hidden_states = hidden_states + (gate_msa * weights.norm_post_attn.apply(attn_out)).to(hidden_states.dtype)

        ffn_in = (weights.norm2.apply(hidden_states) * scale_mlp + shift_mlp).to(bulk_dtype)
        ffn_out = self._ffn(weights, ffn_in)
        hidden_states = hidden_states + (gate_mlp * weights.norm_post_ffn.apply(ffn_out)).to(hidden_states.dtype)
        return hidden_states

    def infer(self, block_weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        for block in block_weights.blocks:
            hidden_states = self._block(block, hidden_states, pre_infer_out.temb6, pre_infer_out.rotary_emb)
        return hidden_states
