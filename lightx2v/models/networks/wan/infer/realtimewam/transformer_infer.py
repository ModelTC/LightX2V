import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.fastwam.transformer_infer import FastWAMTransformerInfer, modulate
from lightx2v.models.networks.wan.weights.realtimewam.transformer_weights import condition_layers


def residual_gate(x, gate, residual):
    return x + gate * residual


class RealtimeWAMTransformerInfer(FastWAMTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.kv_fusion = config.get("kv_fusion", False)
        self.condition_layers = condition_layers(config)
        self.triton_ops = None
        if config.get("triton_ops", False):
            from . import triton_ops

            self.triton_ops = triton_ops
        self.modulate = self.triton_ops.modulate if self.triton_ops else modulate
        self.residual_gate = self.triton_ops.residual_gate if self.triton_ops else residual_gate

    def _reshape_heads(self, x):
        return x.reshape(x.shape[0], -1, self.head_dim)

    def fuse_video_kv(self, weights, index, interval):
        if not self.kv_fusion:
            return interval[-1]
        logits = weights.fusion_logits[self.condition_layers.index(index)].tensor.float()
        if logits.numel() != len(interval):
            raise ValueError(f"KV fusion interval mismatch at layer {index}")
        coefficients = logits.softmax(0).to(interval[0]["k"].dtype)
        fused = {key: interval[0][key] * coefficients[0] for key in ("k", "v")}
        for coefficient, kv in zip(coefficients[1:], interval[1:]):
            for key in fused:
                fused[key] = fused[key] + kv[key] * coefficient
        return fused

    def build_mot_attention_mask(self, video_seq_len, action_seq_len, video_tokens_per_frame, device):
        mask = super().build_mot_attention_mask(video_seq_len, action_seq_len, video_tokens_per_frame, device)
        if self.kv_fusion:
            mask[:video_tokens_per_frame, video_tokens_per_frame:video_seq_len] = False
            mask[video_seq_len:, :video_seq_len] = True
        return mask

    def prefill_video_cache(self, weights, video_pre):
        if not self.kv_fusion:
            return super().prefill_video_cache(weights, video_pre)
        x = video_pre.tokens
        video_mask = self.build_mot_attention_mask(len(x), 0, video_pre.tokens_per_frame, x.device)
        cache, interval = [None] * self.num_layers, []
        for index in range(self.num_layers):
            block = weights.video.blocks[index]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(
                block,
                x,
                video_pre.freqs,
                video_pre.t_mod,
            )
            interval.append({"k": k, "v": v})
            if index in self.condition_layers:
                cache[index] = self.fuse_video_kv(weights, index, interval)
                interval = []
            mixed = block.self_attn.attn.apply(q, k, v, attn_mask=video_mask)
            x = self._post_block(
                block,
                residual_x,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                video_pre.context,
                video_pre.context_mask,
            )
        return cache

    def action_with_video_cache(self, weights, action_pre, video_kv_cache, video_seq_len, attention_mask):
        if not self.kv_fusion:
            return super().action_with_video_cache(weights, action_pre, video_kv_cache, video_seq_len, attention_mask)
        x = action_pre.tokens
        for index in range(self.num_layers):
            block = weights.action.blocks[index]
            q, k_action, v_action, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._build_self_attention_io(
                block,
                x,
                action_pre.freqs,
                action_pre.t_mod,
            )
            conditioned = index in self.condition_layers
            if conditioned:
                kv = video_kv_cache[index]
                k = torch.cat([kv["k"], k_action], dim=0)
                v = torch.cat([kv["v"], v_action], dim=0)
            else:
                k, v = k_action, v_action
            action_mask = attention_mask[video_seq_len:] if conditioned else None
            mixed = block.self_attn.attn.apply(q, k, v, attn_mask=action_mask)
            x = self._post_block(
                block,
                residual_x,
                mixed,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                action_pre.context if conditioned else None,
                action_pre.context_mask,
            )
        return weights.action_head.apply(x)

    def _qk_norm(self, attention, q, k):
        if self.triton_ops and self.config.get("rms_norm_type", "torch") == "torch":
            return self.triton_ops.paired_qk_rms_norm(q, k, attention.norm_q, attention.norm_k)
        return attention.norm_q.apply(q), attention.norm_k.apply(k)

    def _build_self_attention_io(self, block, x, freqs, t_mod):
        self_attn = block.self_attn
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(self_attn, t_mod)
        attn_input = self.modulate(self_attn.norm1.apply(x), shift_msa, scale_msa)

        q, k, v = self_attn.qkv.apply(attn_input)
        q, k = self._qk_norm(self_attn, q, k)
        v = self._reshape_heads(v)
        if self.triton_ops:
            q = self.triton_ops.rope_fp64(q, freqs, self.head_dim)
            k = self.triton_ops.rope_fp64(k, freqs, self.head_dim)
        q, k = self._reshape_heads(q), self._reshape_heads(k)
        if not self.triton_ops:
            q, k = self_attn.rope.apply(q, k, freqs)
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _cross_attn(self, block, x, context, context_mask):
        cross_attn = block.cross_attn
        q = cross_attn.q.apply(cross_attn.norm3.apply(x))
        k, v = cross_attn.kv.apply(context)
        q, k = self._qk_norm(cross_attn, q, k)
        q, k = self._reshape_heads(q), self._reshape_heads(k)
        v = self._reshape_heads(v)
        out = cross_attn.attn.apply(q, k, v, attn_mask=context_mask)
        return cross_attn.o.apply(out)

    def _post_block(self, block, residual_x, mixed_attn_out, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, context_mask):
        x = self.residual_gate(residual_x, gate_msa, block.self_attn.o.apply(mixed_attn_out))
        if context is not None:
            x = x + self._cross_attn(block, x, context, context_mask)
        mlp_input = self.modulate(block.ffn.norm2.apply(x), shift_mlp, scale_mlp)
        x = self.residual_gate(x, gate_mlp, block.ffn.fc2.apply(F.gelu(block.ffn.fc0.apply(mlp_input), approximate="tanh")))
        return x
