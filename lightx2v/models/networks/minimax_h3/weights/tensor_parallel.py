"""MiniMax-H3 tensor-parallel weight helpers.

The common :class:`MMWeightTP` wrapper owns a concrete MM implementation in
``_mm``.  H3 also uses the wrapper with model/block offload, so its tensor
lifecycle must be delegated to that concrete implementation.
"""

from lightx2v.common.ops.mm.mm_weight import MMWeightTP


class MiniMaxH3TensorParallelLinear(MMWeightTP):
    """Tensor-parallel linear that remains compatible with H3 offload."""

    def set_config(self, config=None):
        config = {} if config is None else config
        self.config = config
        self._mm.set_config(config)

    def state_dict(self, destination=None):
        return self._mm.state_dict(destination)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return self._mm.load_state_dict(destination, block_index, adapter_block_index)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        return self._mm.load_state_dict_from_disk(block_index, adapter_block_index)

    def to_cuda(self, non_blocking=False):
        return self._mm.to_cuda(non_blocking=non_blocking)

    def to_cpu(self, non_blocking=False):
        return self._mm.to_cpu(non_blocking=non_blocking)


def unwrap_tp_linear(module):
    """Return the concrete tensor-owning MM implementation."""
    return module._mm if isinstance(module, MiniMaxH3TensorParallelLinear) else module


__all__ = ["MiniMaxH3TensorParallelLinear", "unwrap_tp_linear"]
