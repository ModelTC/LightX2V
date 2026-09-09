import torch

from lightx2v.models.networks.minimax_h3.weights.merged_qkv import (
    _SourceLoRAWeight,
    _operator_device,
    _source_attr_name,
)
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class MiniMaxH3ReorderedMLPWeight:
    """Store the H3 SwiGLU projection in the runtime's gate/value row order."""

    supports_block_offload = True

    def __init__(
        self,
        weight_name,
        bias_name=None,
        mm_type="Default",
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        create_cuda_buffer=False,
        create_cpu_buffer=False,
        lazy_load=False,
        lazy_load_file=None,
        is_post_adapter=False,
        lora_prefix="transformer_blocks",
        lora_path="",
        config=None,
    ):
        if create_cpu_buffer or lazy_load:
            raise NotImplementedError("Packed MiniMax-H3 weights do not support CPU buffers or disk lazy loading")
        if not weight_name.endswith(".weight"):
            raise ValueError(f"Unexpected MiniMax-H3 SwiGLU weight name {weight_name!r}")

        self.source_weight_name = weight_name
        self.source_bias_name = bias_name
        self.weight_name = weight_name.removesuffix(".weight") + ".reordered.weight"
        self.bias_name = self.weight_name.removesuffix(".weight") + ".bias" if bias_name is not None else None
        self.mm_type = mm_type
        self.create_cuda_buffer = create_cuda_buffer
        self._mm = MM_WEIGHT_REGISTER[mm_type](
            weight_name=self.weight_name,
            bias_name=self.bias_name,
            create_cuda_buffer=create_cuda_buffer,
            create_cpu_buffer=False,
            lazy_load=False,
            lazy_load_file=lazy_load_file,
            is_post_adapter=is_post_adapter,
            lora_prefix=lora_prefix,
            lora_path=lora_path,
        )
        if config is not None:
            self._mm.set_config(config)

        self._source_lora = _SourceLoRAWeight(
            weight_name,
            lora_prefix,
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            lora_column_chunks=2,
        )
        self.local_inner_dim = None
        self._source_output_rows = None

    def __getattr__(self, name):
        operator = self.__dict__.get("_mm")
        if operator is None:
            raise AttributeError(name)
        return getattr(operator, name)

    @property
    def has_lora_branch(self):
        return self._source_lora.active

    @property
    def has_diff(self):
        return self._mm.has_diff

    @property
    def registered_source_weight_names(self):
        return (self.source_weight_name,) if self._source_lora.active else ()

    def set_config(self, config=None):
        self._mm.set_config({} if config is None else config)

    def _reorder_output_rows(self, tensor, name):
        if tensor.ndim == 0:
            return tensor.clone()
        if self._source_output_rows is not None and tensor.shape[0] != self._source_output_rows:
            raise ValueError(
                f"MiniMax-H3 SwiGLU {name} must follow the weight output rows "
                f"({self._source_output_rows}), got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] % 2:
            raise ValueError(f"Invalid MiniMax-H3 fused SwiGLU {name} shape {tuple(tensor.shape)}")
        value, gate = tensor.chunk(2, dim=0)
        return torch.cat((gate, value), dim=0).contiguous()

    def _reorder_source_weight(self, weight):
        if weight.ndim != 2 or weight.shape[0] % 2:
            raise ValueError(f"Invalid MiniMax-H3 fused SwiGLU weight shape {tuple(weight.shape)}")
        self._source_output_rows = weight.shape[0]
        self.local_inner_dim = weight.shape[0] // 2
        return self._reorder_output_rows(weight, "weight")

    def load(self, weight_dict):
        if self.source_weight_name not in weight_dict:
            raise KeyError(f"MiniMax-H3 reordered SwiGLU is missing {self.source_weight_name}")
        packed = {self.weight_name: self._reorder_source_weight(weight_dict[self.source_weight_name])}
        consumed = {self.source_weight_name}
        for packed_name, attr_name, _ in self._mm.base_attrs:
            if attr_name == "weight":
                continue
            source_name = _source_attr_name(self.source_weight_name, attr_name, self.source_bias_name)
            if source_name not in weight_dict:
                raise KeyError(f"MiniMax-H3 reordered SwiGLU is missing {source_name}")
            packed[packed_name] = self._reorder_output_rows(weight_dict[source_name], attr_name)
            consumed.add(source_name)
        self._mm.load(packed)
        if not self.create_cuda_buffer:
            for name in consumed:
                weight_dict.pop(name, None)

    def apply(self, input_tensor):
        projected = self._mm.apply(input_tensor)
        if not self._source_lora.active:
            return projected
        value, gate = self._source_lora.apply_lora(input_tensor).chunk(2, dim=-1)
        projected.add_(torch.cat((gate, value), dim=-1))
        return projected

    def _set_source_lora_device(self):
        self._source_lora.set_target_device(_operator_device(self._mm))

    def _pin_source_lora_if_needed(self):
        if (
            self._source_lora.active
            and getattr(self._mm, "pin_weight", None) is not None
            and getattr(self._mm, "weight", None) is None
        ):
            self._source_lora.pin_lora()

    def register_lora(self, weight_dict, strength):
        self._set_source_lora_device()
        self._source_lora.register_lora(weight_dict, strength)
        self._pin_source_lora_if_needed()

    def update_lora(self, weight_dict, strength):
        self._set_source_lora_device()
        self._source_lora.update_lora(weight_dict, strength)
        self._pin_source_lora_if_needed()

    def remove_lora(self):
        self._source_lora.remove_lora()

    def register_diff(self, weight_dict):
        diff_names = (self._source_lora._mm.weight_diff_name, self._source_lora._mm.bias_diff_name)
        present = sorted(set(diff_names).intersection(weight_dict))
        if present:
            raise NotImplementedError(f"Packed MiniMax-H3 SwiGLU does not support diff adapters: {present[:2]}")

    def state_dict(self, destination=None):
        destination = self._mm.state_dict(destination)
        self._source_lora.state_dict(destination)
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        self._mm.load_state_dict(destination, block_index, adapter_block_index)
        self._set_source_lora_device()
        self._source_lora.load_state_dict(destination, block_index, adapter_block_index)
        return destination

    def to_cuda(self, non_blocking=False):
        self._mm.to_cuda(non_blocking)
        self._source_lora.to_cuda(non_blocking)

    def to_cpu(self, non_blocking=False):
        self._mm.to_cpu(non_blocking)
        self._source_lora.to_cpu(non_blocking)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        raise NotImplementedError("Packed MiniMax-H3 SwiGLU does not support disk lazy loading")


__all__ = ["MiniMaxH3ReorderedMLPWeight"]
