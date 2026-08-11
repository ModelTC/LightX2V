"""MiniMax-H3 tensor-parallel weight helpers.

The common :class:`MMWeightTP` wrapper owns a concrete MM implementation in
``_mm``.  H3 also uses the wrapper with model/block offload, so its tensor
lifecycle must be delegated to that concrete implementation.
"""

import torch

from lightx2v.common.ops.mm.mm_weight import MMWeightTP


class MiniMaxH3TensorParallelLinear(MMWeightTP):
    """Tensor-parallel linear that remains compatible with H3 offload."""

    def _local_lora_weights(self, weight_dict):
        down_name = self._mm.lora_down_name
        if down_name not in weight_dict:
            return {}

        up_name = self._mm.lora_up_name
        if up_name not in weight_dict:
            raise KeyError(f"MiniMax-H3 LoRA is missing the up tensor paired with {down_name}")

        lora_down = weight_dict[down_name]
        lora_up = weight_dict[up_name]
        if self.tp_size > 1:
            if self.split_dim == "row":
                if lora_down.shape[1] % self.tp_size:
                    raise ValueError(f"Cannot row-shard {down_name} shape {tuple(lora_down.shape)} across TP size {self.tp_size}")
                lora_down = torch.chunk(lora_down, self.tp_size, dim=1)[self.tp_rank].contiguous()
            elif ".ff.net.0.proj.weight" in self.weight_name:
                if lora_up.shape[0] % 2:
                    raise ValueError(f"Invalid fused SwiGLU LoRA tensor {up_name} with shape {tuple(lora_up.shape)}")
                value, gate = lora_up.chunk(2, dim=0)
                if value.shape[0] % self.tp_size:
                    raise ValueError(f"Cannot fused-column-shard {up_name} shape {tuple(lora_up.shape)} across TP size {self.tp_size}")
                value = torch.chunk(value, self.tp_size, dim=0)[self.tp_rank]
                gate = torch.chunk(gate, self.tp_size, dim=0)[self.tp_rank]
                lora_up = torch.cat((value, gate), dim=0).contiguous()
            else:
                if lora_up.shape[0] % self.tp_size:
                    raise ValueError(f"Cannot column-shard {up_name} shape {tuple(lora_up.shape)} across TP size {self.tp_size}")
                lora_up = torch.chunk(lora_up, self.tp_size, dim=0)[self.tp_rank].contiguous()

        local_weights = {down_name: lora_down, up_name: lora_up}
        alpha_name = self._mm.lora_alpha_name
        if alpha_name in weight_dict:
            local_weights[alpha_name] = weight_dict[alpha_name]
        return local_weights

    def register_lora(self, weight_dict, strength=1.0):
        self._mm.register_lora(self._local_lora_weights(weight_dict), strength)

    def update_lora(self, weight_dict, strength=1.0):
        self._mm.update_lora(self._local_lora_weights(weight_dict), strength)

    def remove_lora(self):
        self._mm.remove_lora()

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
