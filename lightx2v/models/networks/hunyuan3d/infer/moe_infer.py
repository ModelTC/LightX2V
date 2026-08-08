import torch
import torch.nn.functional as F


@torch.no_grad()
def infer_moe_ffn(ffn_weights, hidden_states):
    out = ffn_weights.fc1.apply(hidden_states)
    out = F.gelu(out)
    return ffn_weights.fc2.apply(out)


def _infer_routed_expert(moe_weights, expert_idx, hidden_states):
    return infer_moe_ffn(moe_weights.experts[expert_idx], hidden_states)


@torch.no_grad()
def infer_moe_block(moe_weights, hidden_states):
    bsz, seq_len, hidden_dim = hidden_states.shape
    moe_top_k = moe_weights.moe_top_k

    flat = hidden_states.reshape(-1, hidden_dim)
    logits = moe_weights.gate.apply(flat)

    if moe_weights.moe_backend == "npu":
        moe_weights._ensure_npu_weights()
        routed = moe_weights.routed.apply(
            flat,
            router_logits=logits,
            top_k=moe_top_k,
            num_experts=moe_weights.num_experts,
            fc1_weight=moe_weights._npu_fc1_weight,
            fc2_weight=moe_weights._npu_fc2_weight,
            fc1_bias=moe_weights._npu_fc1_bias,
            fc2_bias=moe_weights._npu_fc2_bias,
            activation="gelu",
        )
    else:
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=moe_top_k, dim=-1, sorted=False)
        if moe_weights.moe_backend == "flashinfer":
            if not hasattr(moe_weights, "_fi_fc1_weight"):
                moe_weights._build_flashinfer_weights()
            routed = moe_weights.routed.apply(
                flat,
                topk_idx,
                topk_weight,
                fc1_weight=moe_weights._fi_fc1_weight,
                fc2_weight=moe_weights._fi_fc2_weight,
                fc1_bias=moe_weights._fi_fc1_bias,
                fc2_bias=moe_weights._fi_fc2_bias,
                tune_max_num_tokens=moe_weights.moe_flashinfer_tune_max_num_tokens,
                activation="gelu",
            )
        else:
            routed = moe_weights.routed.apply(
                flat,
                topk_idx,
                topk_weight,
                num_experts=moe_weights.num_experts,
                expert_context=moe_weights,
                expert_fn=_infer_routed_expert,
                route_index_dtype=torch.int32 if flat.device.type == "mlu" else None,
                combine_mode="slot_sum",
            )

    routed = routed.view(bsz, seq_len, hidden_dim)
    shared = infer_moe_ffn(moe_weights.shared_experts, flat).view(bsz, seq_len, hidden_dim)
    return routed + shared
