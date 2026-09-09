from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.sgl.tensor_parallel import all_gather_last_dim
from lightx2v.models.networks.minimax_h3.infer.sglang_fused import indexed_scale_shift_sglang


class MiniMaxH3SGLPostInfer(MiniMaxH3PostInfer):
    def _gather_tp_last_dim(self, tensor):
        return all_gather_last_dim(tensor, self.tp_group, self.tp_size)

    @staticmethod
    def _apply_modulation(hidden_states, shift, scale, indices):
        return indexed_scale_shift_sglang(hidden_states, shift, scale, indices)


__all__ = ["MiniMaxH3SGLPostInfer"]
