"""Backward-compatible MiniMax-H3 aliases for common TP helpers."""

from lightx2v.common.ops.mm.mm_weight import MMWeightTP, unwrap_tp_weight

MiniMaxH3TensorParallelLinear = MMWeightTP
unwrap_tp_linear = unwrap_tp_weight

__all__ = ["MiniMaxH3TensorParallelLinear", "unwrap_tp_linear"]
