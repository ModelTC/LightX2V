import torch
import torch.nn.functional as F


@torch.no_grad()
def infer_moe_ffn(ffn_weights, hidden_states):
    out = ffn_weights.fc1.apply(hidden_states)
    out = F.gelu(out)
    return ffn_weights.fc2.apply(out)


@torch.no_grad()
def infer_moe_block(moe_weights, hidden_states):
    identity = hidden_states
    bsz, seq_len, hidden_dim = hidden_states.shape
    moe_top_k = moe_weights.moe_top_k

    flat = hidden_states.reshape(-1, hidden_dim)
    logits = moe_weights.gate.apply(flat)
    scores = logits.softmax(dim=-1)
    topk_weight, topk_idx = torch.topk(scores, k=moe_top_k, dim=-1, sorted=False)

    flat_topk_idx = topk_idx.reshape(-1)
    flat_topk_weight = topk_weight.reshape(-1, 1)

    expert_cache = torch.zeros_like(flat)
    idxs = flat_topk_idx.argsort()
    tokens_per_expert = flat_topk_idx.bincount(minlength=moe_weights.num_experts).cpu().numpy().cumsum(0)
    token_idxs = idxs // moe_top_k

    for expert_idx, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]
        if start_idx == end_idx:
            continue
        exp_token_idx = token_idxs[start_idx:end_idx]
        expert_tokens = flat[exp_token_idx]
        expert_out = infer_moe_ffn(moe_weights.experts[expert_idx], expert_tokens)
        expert_out.mul_(flat_topk_weight[idxs[start_idx:end_idx]])
        expert_cache = expert_cache.to(expert_out.dtype)
        expert_cache.scatter_reduce_(
            0,
            exp_token_idx.view(-1, 1).repeat(1, hidden_dim),
            expert_out,
            reduce="sum",
        )

    routed = expert_cache.view(bsz, seq_len, hidden_dim)
    shared = infer_moe_ffn(moe_weights.shared_experts, flat).view(bsz, seq_len, hidden_dim)
    return routed + shared
