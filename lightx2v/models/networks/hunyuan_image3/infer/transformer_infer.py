import torch
import torch.nn.functional as F

from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_mlp, apply_rotary_pos_emb, first_weight_device, repeat_kv, to_device

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except Exception:
    try:
        import flashinfer

        flashinfer_cutlass_fused_moe = flashinfer.fused_moe.cutlass_fused_moe
    except Exception:
        flashinfer_cutlass_fused_moe = None


class HunyuanImage3TransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads") or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.get("attention_head_dim", self.hidden_size // self.num_heads)
        self.hidden_act = config.get("hidden_act", "silu")

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        hidden_states = pre_infer_out.hidden_states
        for block_idx, block in enumerate(weights.blocks):
            hidden_states = self.infer_block(block_idx, block, hidden_states, pre_infer_out)
        return hidden_states

    def infer_block(self, block_idx, block, hidden_states, pre_infer_out):
        attention_phase = block.compute_phases[0]
        mlp_phase = block.compute_phases[1]
        device = first_weight_device(attention_phase)
        hidden_states = to_device(hidden_states, device)
        attention_mask = to_device(pre_infer_out.attention_mask, device)
        position_ids = to_device(pre_infer_out.position_ids, device)
        custom_pos_emb = to_device(pre_infer_out.custom_pos_emb, device)

        residual = hidden_states
        normed = attention_phase.input_layernorm.apply(hidden_states)
        attn_out = self.infer_attention(
            block_idx,
            attention_phase,
            normed,
            attention_mask,
            position_ids,
            custom_pos_emb,
            pre_infer_out.past_key_values if pre_infer_out.use_cache else None,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = mlp_phase.post_attention_layernorm.apply(hidden_states)
        mlp_out = self.infer_mlp(mlp_phase, normed)
        return residual + mlp_out

    def infer_attention(self, block_idx, phase, hidden_states, attention_mask, position_ids, custom_pos_emb, past_key_values=None):
        batch, q_len, _ = hidden_states.shape
        qkv_states = apply_linear(phase.qkv_proj, hidden_states.reshape(-1, hidden_states.shape[-1]))
        qkv_states = qkv_states.reshape(
            batch,
            q_len,
            self.num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)
        query_states = query_states.reshape(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if custom_pos_emb is not None:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if getattr(phase, "query_layernorm", None) is not None:
            query_states = phase.query_layernorm.apply(query_states)
            key_states = phase.key_layernorm.apply(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
        if past_key_values is not None:
            if position_ids is None:
                raise ValueError("HunyuanImage3 KV cache requires position_ids.")
            key_states, value_states = past_key_values.update(key_states, value_states, block_idx, position_ids)
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, q_len, -1)
        attn_output = apply_linear(phase.o_proj, attn_output.reshape(-1, attn_output.shape[-1]))
        return attn_output.reshape(batch, q_len, -1)

    def infer_mlp(self, phase, hidden_states):
        if not phase.is_moe:
            return apply_mlp(phase.gate_and_up_proj, phase.down_proj, hidden_states, self.hidden_act)

        moe = phase.moe
        moe_impl = getattr(moe, "moe_impl", self.config.get("moe_impl", "eager"))
        if moe_impl == "flashinfer":
            return self._infer_mlp_flashinfer(moe, hidden_states)
        if moe_impl != "eager":
            raise ValueError(f"Unsupported HunyuanImage3 moe_impl={moe_impl!r}. Expected 'eager' or 'flashinfer'.")

        return self._infer_mlp_eager(moe, hidden_states)

    def _moe_easy_topk(self, moe, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = apply_linear(moe.gate, flat)
        topk_weight, topk_idx = torch.topk(torch.softmax(logits, dim=-1), moe.moe_topk, dim=-1)
        topk_weight = topk_weight / torch.clamp(topk_weight.sum(dim=-1, keepdim=True), min=1e-8)
        return flat, topk_weight, topk_idx

    def _infer_mlp_eager(self, moe, hidden_states):
        flat, topk_weight, topk_idx = self._moe_easy_topk(moe, hidden_states)
        repeated = flat.repeat_interleave(moe.moe_topk, dim=0)
        expert_outputs = torch.zeros_like(repeated)
        flat_topk_idx = topk_idx.reshape(-1)
        for expert_idx, expert in enumerate(moe.experts):
            mask = flat_topk_idx == expert_idx
            if not torch.any(mask):
                continue
            expert_out = apply_mlp(expert.gate_and_up_proj, expert.down_proj, repeated[mask], self.hidden_act)
            expert_outputs[mask] = expert_out.to(expert_outputs.dtype)
        combined = (expert_outputs.reshape(flat.shape[0], moe.moe_topk, -1) * topk_weight.to(expert_outputs.dtype).unsqueeze(-1)).sum(dim=1)
        output = combined.reshape_as(hidden_states)
        if getattr(moe, "shared_mlp", None) is not None:
            shared_out = apply_mlp(moe.shared_mlp.gate_and_up_proj, moe.shared_mlp.down_proj, hidden_states, self.hidden_act)
            output = output + shared_out.to(output.dtype)
        return output

    def _infer_mlp_flashinfer(self, moe, hidden_states):
        if flashinfer_cutlass_fused_moe is None:
            raise ImportError("HunyuanImage3 moe_impl='flashinfer' requires flashinfer.fused_moe.cutlass_fused_moe.")
        if self.hidden_act != "silu":
            raise NotImplementedError("HunyuanImage3 moe_impl='flashinfer' currently supports only silu/SwiGLU experts.")
        if not hasattr(moe, "ensure_flashinfer_weights"):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' requires HunyuanImage3MoEWeights.")

        if hidden_states.device.type == "cuda" and hidden_states.device.index is not None:
            torch.cuda.set_device(hidden_states.device.index)

        original_dtype = hidden_states.dtype
        compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        flat, topk_weight, topk_idx = self._moe_easy_topk(moe, hidden_states)
        fused_input = flat.to(dtype=compute_dtype).contiguous()
        moe_weight, moe_weight_2 = moe.ensure_flashinfer_weights(fused_input.device, compute_dtype)
        combined_output = torch.zeros_like(fused_input)
        flashinfer_cutlass_fused_moe(
            fused_input,
            topk_idx.to(torch.int32).contiguous(),
            topk_weight.to(torch.float32).contiguous(),
            moe_weight,
            moe_weight_2,
            compute_dtype,
            output=combined_output,
            quant_scales=None,
        )
        output = combined_output.reshape_as(hidden_states).to(original_dtype)
        if getattr(moe, "shared_mlp", None) is not None:
            shared_out = apply_mlp(moe.shared_mlp.gate_and_up_proj, moe.shared_mlp.down_proj, hidden_states, self.hidden_act)
            output = output + shared_out.to(output.dtype)
        return output
