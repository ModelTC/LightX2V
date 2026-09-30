import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.fastwam.transformer_infer import FastWAMTransformerInfer, modulate


def residual_gate(x, gate, residual):
    return x + gate * residual


class RealtimeWAMTransformerInfer(FastWAMTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.triton_ops = None
        if config.get("triton_ops", False):
            from . import triton_ops

            self.triton_ops = triton_ops
        self.modulate = self.triton_ops.modulate if self.triton_ops else modulate
        self.residual_gate = self.triton_ops.residual_gate if self.triton_ops else residual_gate

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
