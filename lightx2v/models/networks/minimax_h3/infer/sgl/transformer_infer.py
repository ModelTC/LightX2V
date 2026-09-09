from lightx2v.models.networks.minimax_h3.infer.sgl.tensor_parallel import all_gather_last_dim
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import (
    _silu_mul_with_activation_rounding_inplace,
    indexed_gate_sglang,
    indexed_scale_shift_sglang,
)
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer


class MiniMaxH3SGLTransformerInfer(MiniMaxH3TransformerInfer):
    def _project_qkv(self, weights, hidden_states):
        projected = weights.qkv.apply(hidden_states)
        return weights.qkv.split_qkv(projected)

    def _apply_qk_norm_rope(self, weights, q, k, pre_infer_out):
        if pre_infer_out.prepared_rotary_emb is None:
            pre_infer_out.prepared_rotary_emb = weights.rope.prepare_freqs(
                pre_infer_out.rotary_emb,
                rotary_dim=pre_infer_out.rotary_emb[0].shape[-1],
            )
        pre_infer_out.prepared_rotary_emb = weights.rope.validate_inputs(
            q,
            k,
            pre_infer_out.prepared_rotary_emb,
        )
        q = weights.norm_q.apply(q)
        k = weights.norm_k.apply(k)
        return weights.rope.apply(
            q,
            k,
            pre_infer_out.prepared_rotary_emb,
            rotary_dim=pre_infer_out.rotary_emb[0].shape[-1],
        )

    @staticmethod
    def _ff(weights, hidden_states):
        hidden_states = weights.in_proj.apply(hidden_states)
        hidden_states = _silu_mul_with_activation_rounding_inplace(hidden_states)
        return weights.out_proj.apply(hidden_states)

    @staticmethod
    def _apply_modulation(hidden_states, shift, scale, indices):
        return indexed_scale_shift_sglang(hidden_states, shift, scale, indices)

    @staticmethod
    def _apply_residual(residual, gate, branch, indices):
        return indexed_gate_sglang(residual, gate, branch, indices)

    def _gather_tp_last_dim(self, tensor):
        return all_gather_last_dim(tensor, self.tp_group, self.tp_size)


__all__ = ["MiniMaxH3SGLTransformerInfer"]
