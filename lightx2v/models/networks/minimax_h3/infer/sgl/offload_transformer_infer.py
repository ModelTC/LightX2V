from lightx2v.models.networks.minimax_h3.infer.offload.transformer_infer import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.sgl.transformer_infer import MiniMaxH3SGLTransformerInfer


class MiniMaxH3SGLOffloadTransformerInfer(
    MiniMaxH3OffloadTransformerInfer,
    MiniMaxH3SGLTransformerInfer,
):
    pass


__all__ = ["MiniMaxH3SGLOffloadTransformerInfer"]
