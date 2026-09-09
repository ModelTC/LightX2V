from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.sgl.tensor_parallel import all_gather_last_dim, row_parallel_linear
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import _silu_mul_with_activation_rounding_inplace


class MiniMaxH3SGLPreInfer(MiniMaxH3PreInfer):
    @staticmethod
    def _project_qkv(weights, hidden_states):
        projected = weights.qkv.apply(hidden_states)
        return weights.qkv.split_qkv(projected)

    @staticmethod
    def _ff(weights, hidden_states):
        hidden_states = weights.in_proj.apply(hidden_states)
        hidden_states = _silu_mul_with_activation_rounding_inplace(hidden_states)
        return weights.out_proj.apply(hidden_states)

    def _gather_tp_last_dim(self, tensor):
        return all_gather_last_dim(tensor, self.tp_group, self.tp_size)

    def _apply_time_linear_2(self, module, hidden_states):
        return row_parallel_linear(module, hidden_states, self.tp_group, self.tp_rank, self.tp_size)


__all__ = ["MiniMaxH3SGLPreInfer"]
