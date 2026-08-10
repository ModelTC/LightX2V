"""Phase-neutral tensor-parallel weights for HunyuanImage3.

The 80B checkpoint is loaded once with a storage TP degree (two for the
four-GPU AR/denoise topology).  Every storage shard contains a fixed number
of contiguous *micro shards*.  Denoising consumes the whole storage shard,
whereas autoregressive inference consumes the local micro shard selected by
``parallel_context.local_micro_shard_id``.

Only views are selected when the phase changes.  The one-time checkpoint to
storage-layout transformation still happens while weights are initialized.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.mm.mm_weight import MMWeightTP

PLAIN_LAYOUT = "plain"
GROUPED_QKV_LAYOUT = "grouped_qkv"
FUSED_GATE_UP_LAYOUT = "fused_gate_up_micro_major"


def _context_value(context: Any, name: str, default=None):
    if context is None:
        return default
    return getattr(context, name, default)


def resolve_storage_tp(config):
    """Return ``(context, group, rank, size)`` for the resident weights.

    The runtime context deliberately owns two topologies.  Loading must use
    ``storage_tp_*`` and must never consult the phase-dependent active TP
    aliases.  Falling back to the original device mesh keeps legacy TP
    configurations working.
    """

    context = config.get("parallel_context")
    if context is not None:
        group = _context_value(context, "storage_tp_group")
        rank = int(_context_value(context, "storage_tp_rank", 0))
        size = int(_context_value(context, "storage_tp_size", 1))
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"Invalid HunyuanImage3 storage TP rank/size: rank={rank}, size={size}.")
        return context, group, rank, size

    if config.get("tensor_parallel", False):
        group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        return None, group, dist.get_rank(group), dist.get_world_size(group)
    return None, None, 0, 1


def resolve_micro_shard_count(config, storage_tp_size=None):
    """Resolve the number of resident micro shards per storage TP rank."""

    context = config.get("parallel_context")
    if storage_tp_size is None:
        _, _, _, storage_tp_size = resolve_storage_tp(config)

    value = _context_value(context, "micro_shard_count")
    if value is not None:
        count = int(value)
    else:
        ar_tp_size = _context_value(context, "ar_tp_size")
        parallel = config.get("parallel") or {}
        if ar_tp_size is None:
            ar_tp_size = parallel.get("ar_tp_size", config.get("ar_tp_size"))
        if ar_tp_size is not None:
            ar_tp_size = int(ar_tp_size)
            if ar_tp_size % storage_tp_size:
                raise ValueError(f"HunyuanImage3 ar_tp_size={ar_tp_size} must be divisible by storage_tp_size={storage_tp_size}.")
            count = ar_tp_size // storage_tp_size
        elif context is None:
            count = 1
        else:
            raise ValueError("HunyuanImage3 parallel_context must expose micro_shard_count (or ar_tp_size) before weights are initialized.")

    if count < 1:
        raise ValueError(f"HunyuanImage3 micro_shard_count must be positive, got {count}.")
    return count


def select_row_storage_shard(tensor, storage_tp_rank, storage_tp_size):
    """Select a checkpoint-layout row-parallel shard (split input dim)."""

    if tensor.ndim == 1:
        return tensor
    if tensor.ndim != 2 or tensor.shape[1] % storage_tp_size:
        raise ValueError(f"Cannot row-shard tensor with shape {tuple(tensor.shape)} across storage TP size {storage_tp_size}.")
    width = tensor.shape[1] // storage_tp_size
    return tensor.narrow(1, storage_tp_rank * width, width).contiguous()


def select_column_storage_shard(tensor, storage_tp_rank, storage_tp_size):
    """Select a checkpoint-layout column-parallel shard (split output dim)."""

    if tensor.shape[0] % storage_tp_size:
        raise ValueError(f"Cannot column-shard tensor with shape {tuple(tensor.shape)} across storage TP size {storage_tp_size}.")
    width = tensor.shape[0] // storage_tp_size
    return tensor.narrow(0, storage_tp_rank * width, width).contiguous()


def select_grouped_qkv_storage_shard(
    tensor,
    storage_tp_rank,
    storage_tp_size,
    micro_shard_count,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    """Shard official HunyuanImage3 QKV by complete KV-head groups.

    The checkpoint output order is
    ``[kv_head, q_heads_per_kv + key + value, head_dim]``.  Treating a raw
    fraction of the fused output as the semantic unit is fragile; this helper
    validates and slices whole KV groups explicitly.
    """

    if num_attention_heads % num_key_value_heads:
        raise ValueError(
            f"HunyuanImage3 Q heads ({num_attention_heads}) must be divisible by KV heads ({num_key_value_heads})."
        )
    total_tp_size = storage_tp_size * micro_shard_count
    if num_key_value_heads % total_tp_size:
        raise ValueError(
            f"HunyuanImage3 KV heads ({num_key_value_heads}) must be divisible by storage_tp_size * micro_shard_count ({total_tp_size})."
        )
    q_per_kv = num_attention_heads // num_key_value_heads
    group_width = (q_per_kv + 2) * head_dim
    expected = num_key_value_heads * group_width
    if tensor.shape[0] != expected:
        raise ValueError(f"Unexpected HunyuanImage3 grouped QKV shape {tuple(tensor.shape)}; expected first dimension {expected}.")

    groups_per_storage_rank = num_key_value_heads // storage_tp_size
    grouped = tensor.reshape(num_key_value_heads, group_width, *tensor.shape[1:])
    local = grouped.narrow(0, storage_tp_rank * groups_per_storage_rank, groups_per_storage_rank)
    return local.reshape(groups_per_storage_rank * group_width, *tensor.shape[1:]).contiguous()


def select_fused_gate_up_storage_shard(tensor, storage_tp_rank, storage_tp_size, micro_shard_count):
    """Build one resident gate/up shard in micro-shard-major order.

    Official order is ``[gate_all, up_all]``.  Resident order is
    ``[gate_micro0, up_micro0, gate_micro1, up_micro1, ...]`` so every AR
    micro shard is one contiguous view.
    """

    if tensor.shape[0] % 2:
        raise ValueError(f"HunyuanImage3 fused gate/up tensor has an odd output dimension: {tuple(tensor.shape)}.")
    gate, up = tensor.chunk(2, dim=0)
    total_tp_size = storage_tp_size * micro_shard_count
    if gate.shape[0] % total_tp_size:
        raise ValueError(f"Cannot shard fused gate/up tensor with shape {tuple(tensor.shape)} across total TP size {total_tp_size}.")

    micro_width = gate.shape[0] // total_tp_size
    first_micro = storage_tp_rank * micro_shard_count
    parts = []
    for local_micro in range(micro_shard_count):
        global_micro = first_micro + local_micro
        start = global_micro * micro_width
        parts.extend((gate.narrow(0, start, micro_width), up.narrow(0, start, micro_width)))
    return torch.cat(parts, dim=0).contiguous()


def restore_gate_up_projection_order(projected, micro_shard_count):
    """Convert micro-major projection output back to ``[gate_all, up_all]``."""

    if micro_shard_count == 1:
        return projected
    if projected.shape[-1] % (2 * micro_shard_count):
        raise ValueError(
            f"Fused gate/up projection width {projected.shape[-1]} is not divisible by 2 * micro_shard_count ({2 * micro_shard_count})."
        )
    micro_width = projected.shape[-1] // (2 * micro_shard_count)
    leading_shape = projected.shape[:-1]
    return (
        projected.reshape(*leading_shape, micro_shard_count, 2, micro_width)
        .transpose(-3, -2)
        .reshape(*leading_shape, 2 * micro_shard_count * micro_width)
    )


class HunyuanImage3HybridTensorParallelLinear(MMWeightTP):
    """Storage-TP linear with phase-dependent, zero-copy weight views."""

    def __init__(
        self,
        *,
        parallel_context,
        micro_shard_count,
        weight_layout=PLAIN_LAYOUT,
        qkv_group_width=None,
        hidden_act="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if kwargs.get("mm_type", "Default") != "Default":
            raise NotImplementedError("HunyuanImage3 hybrid TP currently supports unquantized Default MM weights only.")
        self.parallel_context = parallel_context
        self.micro_shard_count = int(micro_shard_count)
        self.weight_layout = weight_layout
        self.qkv_group_width = qkv_group_width
        self.hidden_act = str(hidden_act).strip().lower()
        self.storage_tp_group = self.tp_group
        self.storage_tp_rank = self.tp_rank
        self.storage_tp_size = self.tp_size
        self.gate_up_layout = FUSED_GATE_UP_LAYOUT if weight_layout == FUSED_GATE_UP_LAYOUT else None

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

    @property
    def active_tp_group(self):
        return _context_value(self.parallel_context, "active_tp_group", _context_value(self.parallel_context, "tp_group", self.storage_tp_group))

    @property
    def active_tp_size(self):
        return int(_context_value(self.parallel_context, "active_tp_size", _context_value(self.parallel_context, "tp_size", self.storage_tp_size)))

    @property
    def uses_micro_shard(self):
        if self.micro_shard_count == 1:
            return False
        active_size = self.active_tp_size
        if active_size == self.storage_tp_size:
            return False
        expected = self.storage_tp_size * self.micro_shard_count
        if active_size != expected:
            raise RuntimeError(
                f"HunyuanImage3 active TP size must be storage_tp_size ({self.storage_tp_size}) or full micro TP size ({expected}); got {active_size}."
            )
        return True

    @property
    def active_micro_shard_id(self):
        micro_id = int(_context_value(self.parallel_context, "local_micro_shard_id", 0)) if self.uses_micro_shard else None
        if micro_id is not None and not 0 <= micro_id < self.micro_shard_count:
            raise RuntimeError(f"Invalid HunyuanImage3 local micro shard id {micro_id} for count {self.micro_shard_count}.")
        return micro_id

    def canonical_tp_rank(self, micro_shard_id=None):
        if micro_shard_id is None:
            micro_shard_id = self.active_micro_shard_id
        if micro_shard_id is None:
            raise RuntimeError("A canonical TP rank is only defined for a selected micro shard.")
        return self.storage_tp_rank * self.micro_shard_count + int(micro_shard_id)

    def _micro_view(self, tensor, split_dim, micro_shard_id):
        if tensor is None:
            return None
        dim = 1 if split_dim == "col" and tensor.ndim == 2 else 0
        extent = tensor.shape[dim]
        if self.weight_layout == GROUPED_QKV_LAYOUT:
            if self.qkv_group_width is None or extent % self.qkv_group_width:
                raise RuntimeError(f"Invalid grouped-QKV tensor shape {tuple(tensor.shape)}.")
            local_groups = extent // self.qkv_group_width
            if local_groups % self.micro_shard_count:
                raise RuntimeError(
                    f"Grouped-QKV local KV groups ({local_groups}) must be divisible by micro_shard_count ({self.micro_shard_count})."
                )
            groups_per_micro = local_groups // self.micro_shard_count
            start = micro_shard_id * groups_per_micro * self.qkv_group_width
            width = groups_per_micro * self.qkv_group_width
        else:
            if extent % self.micro_shard_count:
                raise RuntimeError(
                    f"HunyuanImage3 local {self.split_dim}-parallel extent {extent} must be divisible by micro_shard_count {self.micro_shard_count}."
                )
            width = extent // self.micro_shard_count
            start = micro_shard_id * width
        return tensor.narrow(dim, start, width)

    @property
    def active_weight(self):
        weight = getattr(self._mm, "weight", None)
        if weight is None:
            weight = getattr(self._mm, "pin_weight", None)
        if weight is None:
            raise RuntimeError(f"HunyuanImage3 TP weight {self.weight_name} is not materialized.")
        micro_id = self.active_micro_shard_id
        return weight if micro_id is None else self._micro_view(weight, self.split_dim, micro_id)

    def _active_column_bias(self):
        bias = getattr(self._mm, "bias", None)
        if bias is None:
            bias = getattr(self._mm, "pin_bias", None)
        micro_id = self.active_micro_shard_id
        return bias if micro_id is None else self._micro_view(bias, "col", micro_id)

    def load(self, weight_dict):
        super().load(weight_dict)
        weight = getattr(self._mm, "weight", None)
        if weight is None:
            weight = getattr(self._mm, "pin_weight", None)
        if weight is None:
            return
        split_axis = 1 if self.split_dim == "col" else 0
        if weight.shape[split_axis] % self.micro_shard_count:
            raise ValueError(
                f"HunyuanImage3 resident weight {self.weight_name} shape {tuple(weight.shape)} cannot expose {self.micro_shard_count} micro shards."
            )
        if self.weight_layout == FUSED_GATE_UP_LAYOUT and weight.shape[1] % (2 * self.micro_shard_count):
            raise ValueError(
                f"HunyuanImage3 fused gate/up weight {self.weight_name} has invalid resident width {weight.shape[1]} for {self.micro_shard_count} micro shards."
            )

    def apply(self, input_tensor):
        if getattr(self._mm, "has_lora_branch", False) or getattr(self._mm, "has_diff", False):
            raise NotImplementedError("HunyuanImage3 hybrid TP does not support LoRA or diff weights.")

        weight = self.active_weight
        if self.split_dim == "col":
            bias = self._active_column_bias()
            output = torch.mm(input_tensor, weight) if bias is None else torch.addmm(bias, input_tensor, weight)
            if self.weight_layout == FUSED_GATE_UP_LAYOUT and not self.uses_micro_shard:
                # Only activations are reordered.  The resident weight and all
                # AR views remain untouched across phase switches.
                output = restore_gate_up_projection_order(output, self.micro_shard_count)
            return output

        output = torch.mm(input_tensor, weight)
        if self.reduce_output:
            if self.active_tp_size > 1 and self.active_tp_group is not None:
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.active_tp_group)
            if self._row_split_bias is not None:
                output = output + self._row_split_bias
        return output

    def apply_gate_up_activation(self, input_tensor):
        """Project and apply SwiGLU directly in resident micro-major order.

        The returned intermediate is in canonical local order
        ``[micro0, micro1, ...]``, which matches the resident down-projection
        input rows.  This avoids both a weight transformation and the
        compatibility ``restore_gate_up_projection_order`` activation copy.
        """

        if self.weight_layout != FUSED_GATE_UP_LAYOUT:
            raise RuntimeError(f"apply_gate_up_activation is only valid for fused gate/up weights, got {self.weight_layout!r}.")
        if self.hidden_act != "silu":
            raise NotImplementedError(f"HunyuanImage3 phase-aware fused gate/up currently supports silu/SwiGLU, got {self.hidden_act!r}.")

        weight = self.active_weight
        bias = self._active_column_bias()
        projected = torch.mm(input_tensor, weight) if bias is None else torch.addmm(bias, input_tensor, weight)
        if self.uses_micro_shard:
            gate, up = projected.chunk(2, dim=-1)
            return gate * F.silu(up)

        micro_width = projected.shape[-1] // (2 * self.micro_shard_count)
        parts = projected.reshape(*projected.shape[:-1], self.micro_shard_count, 2, micro_width)
        return (parts[..., 0, :] * F.silu(parts[..., 1, :])).reshape(
            *projected.shape[:-1], self.micro_shard_count * micro_width
        )


__all__ = [
    "PLAIN_LAYOUT",
    "GROUPED_QKV_LAYOUT",
    "FUSED_GATE_UP_LAYOUT",
    "HunyuanImage3HybridTensorParallelLinear",
    "resolve_micro_shard_count",
    "resolve_storage_tp",
    "restore_gate_up_projection_order",
    "select_column_storage_shard",
    "select_fused_gate_up_storage_shard",
    "select_grouped_qkv_storage_shard",
    "select_row_storage_shard",
]
