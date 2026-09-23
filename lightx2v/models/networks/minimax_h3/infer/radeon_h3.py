import os
import sys

import torch
from torch.profiler import record_function
from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTP, unwrap_tp_weight
from radeon_coresw_ops.h3 import H3TP2Workspace, modulate


def enabled(name):
    return os.environ.get("RADEON_CORESW_" + name, "1") != "0"


def apply_modulation(inputs, scale, shift, indices):
    if enabled("LOSSLESS"):
        return modulate(inputs, scale, shift, indices)
    return inputs * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def project_attention(owner, module, inputs, residual, gate, indices, workspace):
    if enabled("OUT_PROJECTION"):
        weight = local_weight(module, owner.tp_group, "row", (3584, 5376))
        if enabled("LOSSLESS"):
            return workspace.attention_output(inputs, weight, residual, gate, indices)
        from radeon_coresw_ops._out_projection import project_out
        output = torch.empty_like(residual)
        project_out(inputs, weight, output)
        torch.distributed.all_reduce(output, group=owner.tp_group)
    else:
        output = module.apply(inputs)
    return residual + gate.index_select(0, indices) * output


def feed_forward(owner, weights, inputs, residual, gate, indices, workspace):
    custom_up, custom_down = enabled("FFN_UP"), enabled("FFN_DOWN")
    if custom_up:
        up_weight = local_weight(weights.in_proj, owner.tp_group, "col", (5376, 14336))
    if custom_down:
        down_weight = local_weight(weights.out_proj, owner.tp_group, "row", (7168, 5376))
    if custom_up and custom_down and enabled("LOSSLESS"):
        return workspace.feed_forward(inputs, up_weight.t(), down_weight, residual, gate, indices)
    if custom_up:
        from radeon_coresw_ops._ffn_up import fused_up
        activation = fused_up(inputs, up_weight.t())
    else:
        value, activation_gate = weights.in_proj.apply(inputs).chunk(2, dim=-1)
        activation = value * torch.nn.functional.silu(activation_gate)
    if custom_down:
        from radeon_coresw_ops._ffn_down import down_out
        output = torch.empty_like(residual)
        down_out(activation, down_weight, output)
        torch.distributed.all_reduce(output, group=owner.tp_group)
    else:
        output = weights.out_proj.apply(activation)
    return residual + gate.index_select(0, indices) * output


def report_status(owner):
    enabled = bool(owner.config.get("radeon_coresw_h3", False))
    if getattr(owner, "_radeon_reported_status", None) == enabled:
        return
    rank = os.environ.get("RANK", "0")
    status = "ENABLED" if enabled else "DISABLED"
    detail = ("awaiting first fused block submission" if enabled else
              "block fusion OFF; library FFN/projection + RCCL remain active; "
              "set radeon_coresw_h3=true to enable; attention dispatch is independent")
    print(f"========== [RADEON OP][rank={rank}][{status}] {detail} ==========",
          file=sys.stderr, flush=True)
    owner._radeon_reported_status = enabled


def local_weight(module, group, split, shape):
    if not isinstance(module, MMWeightTP) or module.tp_size != 2 or module.tp_group is not group or module.split_dim != split:
        raise ValueError("Radeon H3 requires matching TP2 weight partitions")
    inner = unwrap_tp_weight(module)
    if type(inner) is not MMWeight or inner.has_lora_branch or inner.has_diff:
        raise ValueError("Radeon H3 supports unquantized BF16 weights without LoRA/diff only")
    if getattr(inner, "bias", None) is not None or getattr(module, "_row_split_bias", None) is not None:
        raise ValueError("Radeon H3 fused projections require no bias")
    if split == "row" and not module.reduce_output:
        raise ValueError("Radeon H3 row projections require SUM")
    if split == "col" and module.lora_column_chunks != 2:
        raise ValueError("Radeon H3 requires paired local SwiGLU halves")
    weight = inner._get_actual_weight()
    if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16 or weight.stride() != (1, shape[0]):
        raise ValueError(f"Unexpected Radeon H3 weight layout: {weight.shape}, {weight.stride()}, {weight.dtype}")
    return weight


def infer_block(owner, weights, hidden_states, pre_infer_out, modulation):
    if owner.tp_size != 2 or owner.seq_p_group is not None or pre_infer_out.sequence_parallel_state is not None:
        raise ValueError("Radeon H3 block fusion requires TP2 without sequence parallelism")
    if hidden_states.dtype != torch.bfloat16 or owner.hidden_size != 5376 or owner.num_heads != 28 or owner.head_dim != 128:
        raise ValueError("Radeon H3 block fusion requires BF16 H3 TP2 geometry")
    workspace = None
    if enabled("LOSSLESS") and (enabled("OUT_PROJECTION") or (enabled("FFN_UP") and enabled("FFN_DOWN"))):
        workspace = getattr(owner, "_radeon_workspace", None)
        if workspace is None:
            workspace = H3TP2Workspace(hidden_states.shape[0], hidden_states.device, owner.tp_group)
            owner._radeon_workspace = workspace
        if workspace.rows != hidden_states.shape[0] or workspace.device != hidden_states.device:
            raise ValueError("Radeon H3 workspace shape/device changed; recreate the infer instance")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    indices = pre_infer_out.adaln_indices
    with record_function("radeon_h3.norm_modulation.attn"):
        normed = apply_modulation(weights.norm1.apply(hidden_states), scale_msa, shift_msa, indices)
    with record_function("radeon_h3.attention"):
        attention = norm_rope_attention(owner, weights.attn, normed, pre_infer_out)
        if attention is None:
            attention = owner._attention(weights.attn, normed, pre_infer_out, project_output=False)
    with record_function("radeon_h3.out_projection_reduce_residual"):
        hidden_states = project_attention(owner, weights.attn.to_out, attention, hidden_states, gate_msa, indices, workspace)
    del attention, normed
    with record_function("radeon_h3.norm_modulation.ffn"):
        normed = apply_modulation(weights.norm2.apply(hidden_states), scale_mlp, shift_mlp, indices)
    with record_function("radeon_h3.ffn_reduce_residual"):
        output = feed_forward(owner, weights.ff, normed, hidden_states, gate_mlp, indices, workspace)
    if not getattr(owner, "_radeon_active_reported", False):
        rank = os.environ.get("RANK", "0")
        print(f"========== [RADEON OP][rank={rank}][ACTIVE] TP2 fused block submitted; "
              f"device={hidden_states.device} shape={tuple(hidden_states.shape)}; "
              f"switches={dict((name, enabled(name)) for name in ('LOSSLESS', 'ATTENTION', 'OUT_PROJECTION', 'FFN_UP', 'FFN_DOWN'))}; "
              "asynchronous submission only, confirm completion in GPU trace ==========",
              file=sys.stderr, flush=True)
        owner._radeon_active_reported = True
    return output


def _qk_rope_inputs(owner, weights, query, key, value, rotary_emb):
    from lightx2v.common.ops.norm.rms_norm_weight import RMSWeightNative
    from lightx2v.common.ops.rope.torch_rope import TorchRealRope

    def fallback(reason):
        reported = getattr(owner, "_radeon_qk_fallbacks", set())
        if reason not in reported:
            print(f"[RADEON OP][rank={os.environ.get('RANK', '0')}][FALLBACK] QK norm/RoPE: {reason}",
                  file=sys.stderr, flush=True)
            reported.add(reason)
            owner._radeon_qk_fallbacks = reported
        return None

    if owner.tp_size != 2 or owner.seq_p_group is not None or owner.num_heads != 28 or owner.head_dim != 128:
        return fallback("requires TP2 local H28 D128 without SP")
    if type(weights.norm_q) is not RMSWeightNative or type(weights.norm_k) is not RMSWeightNative:
        return fallback("requires torch_native RMSNorm")
    if weights.norm_q.eps != 1e-5 or weights.norm_k.eps != 1e-5:
        return fallback("requires QK norm eps=1e-5")
    if type(weights.rope) is not TorchRealRope or weights.rope.layout != "split_half" or weights.rope.compute_dtype != torch.float32:
        return fallback("requires FP32 split-half TorchRealRope")
    if query.ndim not in (2, 3) or query.shape[0] < 1:
        return fallback("unsupported QKV shape")
    rows = query.shape[0]
    if any(tuple(tensor.shape) not in ((rows, 3584), (rows, 28, 128)) for tensor in (query, key, value)):
        return fallback("requires local QKV [S,3584] or [S,28,128]")
    if query.numel() >= 2**31:
        return fallback("int32 index limit")
    if not isinstance(rotary_emb, tuple) or len(rotary_emb) != 2:
        return fallback("requires full cosine/sine tuple")
    cosine, sine = rotary_emb
    if any(not isinstance(tensor, torch.Tensor) or tensor.shape != (rows, 96) or tensor.dtype != torch.float32
           for tensor in (cosine, sine)):
        return fallback("requires full FP32 cosine/sine [S,96]")
    query_weight = weights.norm_q._get_actual_weight()
    key_weight = weights.norm_k._get_actual_weight()
    if any(not isinstance(tensor, torch.Tensor) or tensor.shape != (128,) or tensor.dtype != torch.bfloat16
           for tensor in (query_weight, key_weight)):
        return fallback("requires BF16 norm weights [128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        return fallback("requires BF16 QKV")
    tensors = (query, key, value, cosine, sine, query_weight, key_weight)
    if any(not tensor.is_cuda or tensor.device != query.device or not tensor.is_contiguous() or tensor.requires_grad for tensor in tensors):
        return fallback("requires contiguous same-device inference tensors")
    if not torch.cuda.get_device_properties(query.device).gcnArchName.startswith("gfx1201"):
        return fallback("requires gfx1201")
    query, key, value = (tensor.view(rows, 28, 128) for tensor in (query, key, value))
    return query, key, value, query_weight, key_weight, cosine, sine


def prepare_qk_rope(owner, weights, query, key, value, rotary_emb):
    if not enabled("LOSSLESS") or not owner.config.get("radeon_coresw_qk_norm_rope", owner.config.get("radeon_coresw_h3", False)):
        return None
    from radeon_coresw_ops import qk_norm, rope

    inputs = _qk_rope_inputs(owner, weights, query, key, value, rotary_emb)
    if inputs is None:
        return None
    query, key, value, query_weight, key_weight, cosine, sine = inputs
    with record_function("radeon_h3.qk_norm_single_wave"):
        query = qk_norm(query, query_weight)
        key = qk_norm(key, key_weight)
    with record_function("radeon_h3.rope"):
        query, key = rope(query, key, cosine, sine)
    route = "QK_NORM_SINGLE_WAVE + ROPE"
    if getattr(owner, "_radeon_qk_active", None) != route:
        print(f"[RADEON OP][rank={os.environ.get('RANK', '0')}][ACTIVE] {route}; "
              "original BF16 norm rounding; asynchronous submission only",
              file=sys.stderr, flush=True)
        owner._radeon_qk_active = route
    return query, key, value


def norm_rope_attention(owner, weights, hidden_states, pre_infer_out):
    """Projected QKV -> fused QK norm/RoPE/Sage quant -> attention core; None selects the unfused path."""
    if not enabled("LOSSLESS") or not owner.config.get("radeon_coresw_qk_norm_rope", owner.config.get("radeon_coresw_h3", False)):
        return None
    if not enabled("ATTENTION"):
        return None
    rank = os.environ.get("RANK", "0")
    if (pre_infer_out.sequence_parallel_state is not None or type(weights.calculate).__name__ != "RadeonCoreswAttnWeight"
            or (owner.use_fused_qkv and weights.has_fused_qkv)):
        if not getattr(owner, "_radeon_fused_qk_fallback", False):
            print(f"[RADEON OP][rank={rank}][FALLBACK] fused QK norm/RoPE attention: "
                  "requires radeon_coresw_attn, no SP and separate Q/K/V projections", file=sys.stderr, flush=True)
            owner._radeon_fused_qk_fallback = True
        return None
    query = weights.to_q.apply(hidden_states)
    key = weights.to_k.apply(hidden_states)
    value = weights.to_v.apply(hidden_states)
    inputs = _qk_rope_inputs(owner, weights, query, key, value, pre_infer_out.rotary_emb)
    if inputs is None:
        return None
    query, key, value, query_weight, key_weight, cosine, sine = inputs
    with record_function("radeon_h3.norm_rope_attention"):
        output = torch.ops.radeon_coresw_ops.norm_rope_attention(query, key, value, query_weight, key_weight, cosine, sine)
    if not getattr(owner, "_radeon_fused_qk_active", False):
        print(f"[RADEON OP][rank={rank}][ACTIVE] QK_NORM + ROPE + SAGE_QUANT fused HIP prepare -> attention core; "
              "bitwise equal to separate kernels; asynchronous submission only", file=sys.stderr, flush=True)
        owner._radeon_fused_qk_active = True
    return output.view(query.shape[0], 28 * 128)
