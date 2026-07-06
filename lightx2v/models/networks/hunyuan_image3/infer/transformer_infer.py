import torch
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.ops.attn.flash_attn import FlashAttn2Weight, flash_attn_func_v2
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_mlp, apply_rotary_pos_emb, first_weight_device, repeat_kv, to_device

try:
    import flashinfer
except ImportError:
    flashinfer = None


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
        self.attn_impl = config.get("attn_impl", config.get("attn_implementation", "sdpa"))
        self.moe_impl = config.get("moe_impl", "eager")
        self._warned_flash_attention_fallback = False
        self._warned_flash_attention_mask_fallback = False
        self.flash_attn2 = FlashAttn2Weight() if self.attn_impl == "flash_attention_2" and flash_attn_func_v2 is not None else None
        self._warned_flashinfer_fallback = False

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
        allow_segmented_mask = past_key_values is None or self._is_full_prefill_position_ids(position_ids, q_len, key_states.shape[-2])
        flash_attention_segments = self._flash_attention_2_segments(
            attention_mask,
            q_len,
            key_states.shape[-2],
            query_states,
            allow_segmented_mask=allow_segmented_mask,
        )
        if flash_attention_segments is None:
            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0)
        else:
            attn_output = self._apply_flash_attention_2_segments(query_states, key_states, value_states, flash_attention_segments)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, q_len, -1)
        attn_output = apply_linear(phase.o_proj, attn_output.reshape(-1, attn_output.shape[-1]))
        return attn_output.reshape(batch, q_len, -1)

    def _causal_attention_mask(self, q_len, kv_len, device):
        return torch.ones((q_len, kv_len), dtype=torch.bool, device=device).tril(diagonal=kv_len - q_len)

    def _is_full_prefill_position_ids(self, position_ids, q_len, kv_len):
        if position_ids is None or q_len != kv_len:
            return False
        expected = torch.arange(q_len, dtype=position_ids.dtype, device=position_ids.device).reshape(1, q_len)
        return bool(torch.equal(position_ids, expected.expand_as(position_ids)))

    def _is_causal_attention_mask(self, attention_mask, q_len, kv_len):
        if attention_mask is None:
            return False
        if attention_mask.dtype != torch.bool:
            return False
        expected = self._causal_attention_mask(q_len, kv_len, attention_mask.device)
        return bool(torch.equal(attention_mask.squeeze(1), expected.expand(attention_mask.shape[0], q_len, kv_len)))

    def _attention_mask_to_flash_attention_2_segments(self, attention_mask, q_len, kv_len, allow_segmented_mask=True):
        if attention_mask is None:
            return [(0, q_len, kv_len, False)]
        if attention_mask.dtype != torch.bool:
            return None
        if attention_mask.ndim == 4:
            if attention_mask.shape[1] != 1:
                return None
            mask = attention_mask[:, 0]
        elif attention_mask.ndim == 3:
            mask = attention_mask
        else:
            return None
        if mask.shape[-2:] != (q_len, kv_len):
            return None
        if not torch.equal(mask, mask[:1].expand_as(mask)):
            batch_segments = []
            for sample_mask in mask:
                sample_segments = self._single_attention_mask_to_flash_attention_2_segments(
                    sample_mask,
                    q_len,
                    kv_len,
                    allow_segmented_mask=allow_segmented_mask,
                )
                if sample_segments is None:
                    return None
                batch_segments.append(sample_segments)
            return batch_segments
        return self._single_attention_mask_to_flash_attention_2_segments(mask[0], q_len, kv_len, allow_segmented_mask=allow_segmented_mask)

    def _single_attention_mask_to_flash_attention_2_segments(self, sample_mask, q_len, kv_len, allow_segmented_mask=True):
        causal_mask = self._causal_attention_mask(q_len, kv_len, sample_mask.device)
        if torch.equal(sample_mask, causal_mask):
            return [(0, q_len, kv_len, True)]
        cache_segments = self._kv_cache_denoise_mask_to_flash_attention_2_segments(sample_mask, q_len, kv_len)
        if cache_segments is not None:
            return cache_segments
        if not allow_segmented_mask or q_len != kv_len:
            return None
        if torch.any(causal_mask & ~sample_mask):
            return None

        extra_mask = sample_mask & ~causal_mask
        reconstructed = causal_mask.clone()
        full_blocks = []
        row_idx = 0
        while row_idx < q_len:
            extra_cols = torch.nonzero(extra_mask[row_idx], as_tuple=False).flatten()
            if extra_cols.numel() == 0:
                row_idx += 1
                continue
            block_start = row_idx
            block_end = int(extra_cols.max().item()) + 1
            if block_end <= block_start + 1:
                return None
            reconstructed[block_start:block_end, block_start:block_end] = True
            full_blocks.append((block_start, block_end))
            row_idx = block_end

        if not full_blocks or not torch.equal(reconstructed, sample_mask):
            return None

        segments = []
        prev_end = 0
        for block_start, block_end in full_blocks:
            if prev_end < block_start:
                segments.append((prev_end, block_start, block_start, True))
            segments.append((block_start, block_end, block_end, False))
            prev_end = block_end
        if prev_end < q_len:
            segments.append((prev_end, q_len, kv_len, True))
        return segments

    def _kv_cache_denoise_mask_to_flash_attention_2_segments(self, sample_mask, q_len, kv_len):
        if q_len >= kv_len:
            return None
        prefix_lengths = self._row_prefix_lengths(sample_mask)
        if prefix_lengths is None:
            return None
        suffix_start = q_len - 1
        image_kv_end = prefix_lengths[-1]
        while suffix_start > 0 and prefix_lengths[suffix_start - 1] == image_kv_end:
            suffix_start -= 1
        if suffix_start == q_len:
            return None

        reconstructed = torch.zeros_like(sample_mask)
        segments = []
        special_end = suffix_start
        if special_end > 0 and prefix_lengths[:special_end] == list(range(1, special_end + 1)):
            prefix = self._causal_attention_mask(special_end, special_end, sample_mask.device)
            if not torch.equal(sample_mask[:special_end, :special_end], prefix):
                return None
            if torch.any(sample_mask[:special_end, special_end:]):
                return None
            reconstructed[:special_end, :special_end] = prefix
            segments.append((0, special_end, special_end, True))
        else:
            for row_idx in range(special_end):
                prefix_len = prefix_lengths[row_idx]
                if prefix_len <= 0:
                    return None
                reconstructed[row_idx, :prefix_len] = True
                segments.append((row_idx, row_idx + 1, prefix_len, False))

        if image_kv_end <= 0:
            return None
        reconstructed[suffix_start:, :image_kv_end] = True
        if not torch.equal(reconstructed, sample_mask):
            return None
        segments.append((suffix_start, q_len, image_kv_end, False))
        return segments

    def _row_prefix_lengths(self, sample_mask):
        prefix_lengths = []
        for row in sample_mask:
            false_positions = torch.nonzero(~row, as_tuple=False).flatten()
            prefix_len = int(false_positions[0].item()) if false_positions.numel() else int(row.shape[0])
            if torch.any(row[prefix_len:]):
                return None
            prefix_lengths.append(prefix_len)
        return prefix_lengths

    def _flash_attention_2_segments(self, attention_mask, q_len, kv_len, query_states, allow_segmented_mask=True):
        if self.attn_impl != "flash_attention_2":
            return None
        if self.flash_attn2 is None or flash_attn_func_v2 is None:
            if not self._warned_flash_attention_fallback:
                logger.warning("HunyuanImage3 attn_impl='flash_attention_2' requested but flash-attn2 is not available; falling back to PyTorch SDPA.")
                self._warned_flash_attention_fallback = True
            return None
        segments = self._attention_mask_to_flash_attention_2_segments(attention_mask, q_len, kv_len, allow_segmented_mask=allow_segmented_mask)
        if segments is None:
            if attention_mask is not None and not self._warned_flash_attention_mask_fallback:
                logger.warning("HunyuanImage3 attn_impl='flash_attention_2' does not support the current attention_mask layout; falling back to PyTorch SDPA for masked attention.")
                self._warned_flash_attention_mask_fallback = True
            return None
        if query_states.device.type != "cuda":
            if not self._warned_flash_attention_fallback:
                logger.warning("HunyuanImage3 attn_impl='flash_attention_2' requires CUDA tensors; falling back to PyTorch SDPA.")
                self._warned_flash_attention_fallback = True
            return None
        if query_states.dtype not in (torch.float16, torch.bfloat16):
            if not self._warned_flash_attention_fallback:
                logger.warning("HunyuanImage3 attn_impl='flash_attention_2' requires fp16/bf16 tensors; falling back to PyTorch SDPA.")
                self._warned_flash_attention_fallback = True
            return None
        return segments

    def _build_cu_seqlens(self, batch, seq_len, device):
        return torch.arange(0, (batch + 1) * seq_len, seq_len, dtype=torch.int32, device=device)

    def _apply_flash_attention_2(self, query_states, key_states, value_states, causal=False):
        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[-2]
        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()
        attn_output = self.flash_attn2.apply(
            q,
            k,
            v,
            cu_seqlens_q=self._build_cu_seqlens(batch, q_len, q.device),
            cu_seqlens_kv=self._build_cu_seqlens(batch, kv_len, k.device),
            max_seqlen_q=q_len,
            max_seqlen_kv=kv_len,
            causal=causal,
        )
        return attn_output.reshape(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _apply_flash_attention_2_segments(self, query_states, key_states, value_states, segments):
        if segments and isinstance(segments[0], list):
            return torch.cat(
                [
                    self._apply_flash_attention_2_segments(
                        query_states[batch_idx : batch_idx + 1],
                        key_states[batch_idx : batch_idx + 1],
                        value_states[batch_idx : batch_idx + 1],
                        batch_segments,
                    )
                    for batch_idx, batch_segments in enumerate(segments)
                ],
                dim=0,
            )
        outputs = []
        for q_start, q_end, kv_end, causal in segments:
            outputs.append(
                self._apply_flash_attention_2(
                    query_states[:, :, q_start:q_end, :],
                    key_states[:, :, :kv_end, :],
                    value_states[:, :, :kv_end, :],
                    causal=causal,
                )
            )
        return torch.cat(outputs, dim=2)

    def infer_mlp(self, phase, hidden_states):
        if not phase.is_moe:
            return apply_mlp(phase.gate_and_up_proj, phase.down_proj, hidden_states, self.hidden_act)

        moe = phase.moe
        if self.moe_impl == "flashinfer":
            fused_out = self._infer_mlp_flashinfer(moe, hidden_states)
            if fused_out is not None:
                return fused_out

        output = torch.zeros_like(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = apply_linear(moe.gate, flat)
        topk_weight, topk_idx = torch.topk(torch.softmax(logits, dim=-1), moe.moe_topk, dim=-1)
        topk_weight = topk_weight / torch.clamp(topk_weight.sum(dim=-1, keepdim=True), min=1e-8)
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

    def _linear_actual_weight(self, linear):
        if getattr(linear, "has_lora_branch", False):
            return None
        if getattr(linear, "bias", None) is not None or getattr(linear, "pin_bias", None) is not None:
            return None
        if hasattr(linear, "_get_actual_weight"):
            return linear._get_actual_weight()
        weight = getattr(linear, "weight", None)
        if weight is not None:
            return weight
        return getattr(linear, "pin_weight", None)

    def _get_flashinfer_moe_weights(self, moe, device, dtype):
        cache_key = (str(device), str(dtype))
        cache = getattr(moe, "_flashinfer_weight_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        fc1_weights = []
        fc2_weights = []
        for expert in moe.experts:
            gate_up = self._linear_actual_weight(expert.gate_and_up_proj)
            down = self._linear_actual_weight(expert.down_proj)
            if gate_up is None or down is None:
                return None
            fc1_weights.append(gate_up.t().to(device=device, dtype=dtype))
            fc2_weights.append(down.t().to(device=device, dtype=dtype))
        packed = (torch.stack(fc1_weights, dim=0).contiguous(), torch.stack(fc2_weights, dim=0).contiguous())
        cache[cache_key] = packed
        moe._flashinfer_weight_cache = cache
        return packed

    def _infer_mlp_flashinfer(self, moe, hidden_states):
        if (
            flashinfer is None
            or not hasattr(flashinfer, "fused_moe")
            or hidden_states.device.type != "cuda"
            or self.hidden_act != "silu"
            or getattr(moe, "shared_mlp", None) is not None
        ):
            if not self._warned_flashinfer_fallback:
                logger.warning("HunyuanImage3 moe_impl='flashinfer' requested but requirements are not met; falling back to eager MoE.")
                self._warned_flashinfer_fallback = True
            return None

        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = apply_linear(moe.gate, flat)
        topk_weight, topk_idx = torch.topk(torch.softmax(logits, dim=-1), moe.moe_topk, dim=-1)
        topk_weight = topk_weight / torch.clamp(topk_weight.sum(dim=-1, keepdim=True), min=1e-8)
        packed_weights = self._get_flashinfer_moe_weights(moe, flat.device, hidden_states.dtype)
        if packed_weights is None:
            if not self._warned_flashinfer_fallback:
                logger.warning("HunyuanImage3 flashinfer MoE does not support the current expert weight layout; falling back to eager MoE.")
                self._warned_flashinfer_fallback = True
            return None

        combined = torch.empty_like(flat)
        flashinfer.fused_moe.cutlass_fused_moe(
            flat.contiguous(),
            topk_idx.to(torch.int32).contiguous(),
            topk_weight.to(torch.float32).contiguous(),
            packed_weights[0],
            packed_weights[1],
            hidden_states.dtype,
            quant_scales=None,
            output=combined,
        )
        return combined.reshape_as(hidden_states)
