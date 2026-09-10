import torch

from lightx2v.common.ops.mm.mm_weight import MMWeight, MMWeightTP
from lightx2v.common.ops.utils import create_pin_tensor, resolve_block_name
from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER


class _LoRAOnlyMMWeight(MMWeight):
    """Use the common LoRA implementation without owning a base matrix."""

    def __init__(self, weight_name, lora_prefix):
        super().__init__(weight_name=weight_name, bias_name=None, lora_prefix=lora_prefix)
        self.base_attrs = []
        self._lora_target_device = torch.device("cpu")

    def set_lora_target_device(self, device):
        self._lora_target_device = torch.device(device)

    def _get_lora_target_device(self):
        return self._lora_target_device


class _SourceLoRAWeight(MMWeightTP):
    """A source-named LoRA branch with the common TP sharding rules."""

    def __init__(
        self,
        weight_name,
        lora_prefix,
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        lora_column_chunks=1,
    ):
        super().__init__(
            weight_name=weight_name,
            bias_name=None,
            mm_type="Default",
            tp_group=tp_group,
            tp_rank=tp_rank,
            tp_size=tp_size,
            split_dim="col",
            reduce_output=False,
            lora_column_chunks=lora_column_chunks,
            lora_prefix=lora_prefix,
        )
        self._mm = _LoRAOnlyMMWeight(weight_name, lora_prefix)
        self._pending_lora_strength = 1.0

    @property
    def active(self):
        return self._mm.has_lora_branch

    def set_target_device(self, device):
        self._mm.set_lora_target_device(device)

    def register_lora(self, weight_dict, lora_strength=1):
        self._pending_lora_strength = float(lora_strength)
        super().register_lora(weight_dict, lora_strength)

    def update_lora(self, weight_dict, lora_strength=1):
        self._pending_lora_strength = float(lora_strength)
        super().update_lora(weight_dict, lora_strength)

    def pin_lora(self):
        for name in ("lora_down", "lora_up", "lora_alpha", "lora_scale"):
            tensor = getattr(self._mm, name, None)
            if isinstance(tensor, torch.Tensor):
                setattr(self._mm, name, create_pin_tensor(tensor.to("cpu")))

    def apply_lora(self, input_tensor):
        return self._mm.apply_lora(input_tensor)

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        down_name = resolve_block_name(self._mm.lora_down_name, block_index)
        if down_name not in destination:
            self._mm.has_lora_branch = False
            return destination

        if not hasattr(self._mm, "lora_down"):
            local_weights = {}
            for attr, name_attr in self._mm.lora_attrs.items():
                if attr in ("weight_diff", "bias_diff"):
                    continue
                source_name = resolve_block_name(getattr(self._mm, name_attr), block_index)
                if source_name in destination:
                    local_weights[getattr(self._mm, name_attr)] = destination[source_name]
            self._mm.register_lora(local_weights, self._pending_lora_strength)
        else:
            self._mm.has_lora_branch = True
            self._mm.load_state_dict(destination, block_index, adapter_block_index)

        if hasattr(self._mm, "lora_alpha"):
            self._mm.lora_scale = self._mm.lora_alpha / self._mm.lora_down.shape[0]
        else:
            self._mm.lora_scale = torch.tensor(1.0, device=self._mm.lora_down.device)
        return destination


def _operator_device(operator):
    for name in ("weight", "pin_weight", "weight_cuda_buffer"):
        tensor = getattr(operator, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


def _source_attr_name(weight_name, attr_name, bias_name=None):
    if attr_name == "weight":
        return weight_name
    if attr_name == "bias":
        if bias_name is None:
            raise ValueError(f"No source bias was provided for {weight_name}")
        return bias_name
    return weight_name.removesuffix(".weight") + f".{attr_name}"


def _validate_tensors(tensors, names, *, output_rows=None):
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise TypeError(f"Packed tensors must use one dtype: {dict(zip(names, (tensor.dtype for tensor in tensors)))}")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(f"Packed tensors must be on one device: {dict(zip(names, (tensor.device for tensor in tensors)))}")
    if len({tuple(tensor.shape[1:]) for tensor in tensors}) != 1:
        raise ValueError(f"Packed tensors must have matching trailing shapes: {dict(zip(names, (tuple(tensor.shape) for tensor in tensors)))}")
    if output_rows is not None and any(tensor.ndim == 0 or tensor.shape[0] != rows for tensor, rows in zip(tensors, output_rows)):
        raise ValueError(f"Packed metadata must follow the weight output rows: {dict(zip(names, (tuple(tensor.shape) for tensor in tensors)))}")


class MiniMaxH3MergedQKVWeight:
    """Pack Q/K/V once, then use the selected common matrix-multiply operator."""

    supports_block_offload = True

    def __init__(
        self,
        weight_names,
        bias_name=None,
        bias_names=None,
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
        self.source_weight_names = tuple(weight_names)
        if len(self.source_weight_names) != 3:
            raise ValueError(f"MiniMax-H3 merged QKV expects three source weights, got {self.source_weight_names}")
        if bias_name is not None:
            raise ValueError("Pass Q/K/V source biases through bias_names")
        self.source_bias_names = None if bias_names is None else tuple(bias_names)
        if self.source_bias_names is not None and len(self.source_bias_names) != 3:
            raise ValueError(f"MiniMax-H3 merged QKV expects three source biases, got {self.source_bias_names}")
        if create_cpu_buffer or lazy_load:
            raise NotImplementedError("Packed MiniMax-H3 weights do not support CPU buffers or disk lazy loading")

        q_weight_name = self.source_weight_names[0]
        if not q_weight_name.endswith(".to_q.weight"):
            raise ValueError(f"Unexpected MiniMax-H3 Q weight name {q_weight_name!r}")
        self.weight_name = q_weight_name.removesuffix(".to_q.weight") + ".qkv_packed.weight"
        self.bias_name = self.weight_name.removesuffix(".weight") + ".bias" if self.source_bias_names is not None else None
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

        self._source_loras = tuple(
            _SourceLoRAWeight(
                source_name,
                lora_prefix,
                tp_group=tp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            for source_name in self.source_weight_names
        )
        self.local_qkv_dim = None
        self._source_output_rows = None

    def __getattr__(self, name):
        operator = self.__dict__.get("_mm")
        if operator is None:
            raise AttributeError(name)
        return getattr(operator, name)

    @property
    def has_lora_branch(self):
        return any(branch.active for branch in self._source_loras)

    @property
    def has_diff(self):
        return self._mm.has_diff

    @property
    def registered_source_weight_names(self):
        return tuple(branch.weight_name for branch in self._source_loras if branch.active)

    def set_config(self, config=None):
        self._mm.set_config({} if config is None else config)

    def _pack_source_weights(self, weight_dict):
        missing = [name for name in self.source_weight_names if name not in weight_dict]
        if missing:
            raise KeyError(f"MiniMax-H3 merged QKV is missing source weights: {missing}")
        weights = tuple(weight_dict[name] for name in self.source_weight_names)
        shapes = [tuple(weight.shape) for weight in weights]
        if any(weight.ndim != 2 for weight in weights):
            raise ValueError(f"MiniMax-H3 Q/K/V weights must be two-dimensional, got {shapes}")
        if len({weight.shape for weight in weights}) != 1:
            raise ValueError(f"MiniMax-H3 Q/K/V weights must have identical shapes, got {shapes}")
        _validate_tensors(weights, self.source_weight_names)
        self.local_qkv_dim = weights[0].shape[0]
        self._source_output_rows = tuple(weight.shape[0] for weight in weights)
        return torch.cat(weights, dim=0).contiguous()

    def _pack_attr(self, weight_dict, attr_name):
        bias_names = self.source_bias_names or (None,) * 3
        names = tuple(
            _source_attr_name(weight_name, attr_name, source_bias_name)
            for weight_name, source_bias_name in zip(self.source_weight_names, bias_names)
        )
        missing = [name for name in names if name not in weight_dict]
        if missing:
            raise KeyError(f"MiniMax-H3 merged QKV is missing {attr_name}: {missing}")
        tensors = tuple(weight_dict[name] for name in names)
        if all(tensor.ndim == 0 for tensor in tensors):
            if any(tensor.item() != tensors[0].item() for tensor in tensors[1:]):
                raise ValueError(f"MiniMax-H3 Q/K/V {attr_name} values must match")
            return tensors[0].clone(), names
        _validate_tensors(tensors, names, output_rows=self._source_output_rows)
        return torch.cat(tensors, dim=0).contiguous(), names

    def load(self, weight_dict):
        packed = {self.weight_name: self._pack_source_weights(weight_dict)}
        consumed = set(self.source_weight_names)
        for packed_name, attr_name, _ in self._mm.base_attrs:
            if attr_name == "weight":
                continue
            value, source_names = self._pack_attr(weight_dict, attr_name)
            packed[packed_name] = value
            consumed.update(source_names)
        self._mm.load(packed)
        if not self.create_cuda_buffer:
            for name in consumed:
                weight_dict.pop(name, None)

    def apply(self, input_tensor):
        projected = self._mm.apply(input_tensor)
        if not self.has_lora_branch:
            return projected
        for index, branch in enumerate(self._source_loras):
            if branch.active:
                start = index * self.local_qkv_dim
                projected[..., start : start + self.local_qkv_dim].add_(branch.apply_lora(input_tensor))
        return projected

    def split_qkv(self, projected):
        if self.local_qkv_dim is None:
            raise RuntimeError("MiniMax-H3 merged QKV has not been loaded")
        expected_width = 3 * self.local_qkv_dim
        if projected.shape[-1] != expected_width:
            raise ValueError(f"MiniMax-H3 merged QKV output width must be {expected_width}, got {projected.shape[-1]}")
        return projected.split(self.local_qkv_dim, dim=-1)

    def _adapter_target_device(self):
        return _operator_device(self._mm)

    def _pin_source_loras_if_needed(self):
        if getattr(self._mm, "pin_weight", None) is not None and getattr(self._mm, "weight", None) is None:
            for branch in self._source_loras:
                if branch.active:
                    branch.pin_lora()

    def _set_source_lora_devices(self):
        target = self._adapter_target_device()
        for branch in self._source_loras:
            branch.set_target_device(target)

    def register_lora(self, weight_dict, strength):
        self._set_source_lora_devices()
        for branch in self._source_loras:
            branch.register_lora(weight_dict, strength)
        self._pin_source_loras_if_needed()

    def update_lora(self, weight_dict, strength):
        self._set_source_lora_devices()
        for branch in self._source_loras:
            branch.update_lora(weight_dict, strength)
        self._pin_source_loras_if_needed()

    def remove_lora(self):
        for branch in self._source_loras:
            branch.remove_lora()

    def register_diff(self, weight_dict):
        diff_names = {
            name
            for branch in self._source_loras
            for name in (branch._mm.weight_diff_name, branch._mm.bias_diff_name)
        }
        present = sorted(diff_names.intersection(weight_dict))
        if present:
            raise NotImplementedError(f"Packed MiniMax-H3 QKV does not support diff adapters: {present[:3]}")

    def state_dict(self, destination=None):
        destination = self._mm.state_dict(destination)
        for branch in self._source_loras:
            branch.state_dict(destination)
        return destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        self._mm.load_state_dict(destination, block_index, adapter_block_index)
        self._set_source_lora_devices()
        for branch in self._source_loras:
            branch.load_state_dict(destination, block_index, adapter_block_index)
        return destination

    def to_cuda(self, non_blocking=False):
        self._mm.to_cuda(non_blocking)
        for branch in self._source_loras:
            branch.to_cuda(non_blocking)

    def to_cpu(self, non_blocking=False):
        self._mm.to_cpu(non_blocking)
        for branch in self._source_loras:
            branch.to_cpu(non_blocking)

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        raise NotImplementedError("Packed MiniMax-H3 QKV does not support disk lazy loading")


__all__ = ["MiniMaxH3MergedQKVWeight"]
