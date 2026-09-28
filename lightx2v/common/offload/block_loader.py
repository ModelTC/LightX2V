"""Model-independent planning and loading for immutable offload groups."""

from dataclasses import dataclass, replace

import torch

from lightx2v.common.offload.block_layout import BlockBuffer, BlockLayout, BlockLoadContext
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE
from lightx2v_platform.base.offload import get_block_offload_backend
from lightx2v_platform.ops.weight_storage import StorageDescription, TensorMetadata


@dataclass
class OffloadGroup:
    blocks: object
    device_slots: object
    checkpoint_prefixes: tuple[str, ...]

    def __post_init__(self):
        if not self.blocks or len(self.device_slots) != 2 or len(self.blocks) != len(self.checkpoint_prefixes):
            raise ValueError("An offload group requires CPU blocks, their checkpoint prefixes, and two device slots")
        if any(not prefix or not prefix.endswith(".") for prefix in self.checkpoint_prefixes):
            raise ValueError("Block checkpoint prefixes must be nonempty and end with '.'")


def validate_contiguous_config(config, lora_path=None):
    layout = config.get("cpu_offload_layout", "per_tensor")
    if layout not in ("per_tensor", "contiguous"):
        raise ValueError(f"Unsupported cpu_offload_layout: {layout!r}")
    if layout != "contiguous":
        return
    get_block_offload_backend()
    if not config.get("cpu_offload") or config.get("offload_granularity", "block") != "block":
        raise ValueError("contiguous layout requires CPU block offload")
    if bool(config.get("dit_quantized")) != (config.get("dit_quant_scheme", "Default") != "Default"):
        raise ValueError("dit_quantized must agree with dit_quant_scheme")
    incompatible = (
        "shared_cpu_weights",
        "lazy_load",
        "parallel",
        "tensor_parallel",
        "seq_parallel",
        "cfg_parallel",
        "pipefusion_parallel",
        "lora_configs",
        "lora_path",
        "lora_dynamic_apply",
        "use_compile",
        "enable_cuda_graph",
        "dummy_model",
        "do_mm_calib",
        "weight_auto_quant",
        "quant_method",
        "adapter_model_path",
    )
    if any(config.get(key) for key in incompatible) or lora_path or config.get("feature_caching", "NoCaching") != "NoCaching":
        raise ValueError("contiguous layout requires static rank-local weights, single-device eager inference, and NoCaching")
    if GET_SENSITIVE_DTYPE() != GET_DTYPE():
        raise ValueError("contiguous layout requires the default sensitive-layer dtype")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("contiguous layout currently supports a single rank")


def _describe(leaf, metadata, path):
    if getattr(leaf, "is_post_adapter", False):
        raise ValueError(f"contiguous layout does not support adapter blocks: {path}")
    describe = getattr(leaf, "describe_storage", None)
    if describe is not None:
        return describe(metadata)
    # Stateless computation operators expose an empty state_dict before loading.
    state_dict = getattr(leaf, "state_dict", None)
    if state_dict is not None:
        try:
            if not state_dict():
                return StorageDescription()
        except AttributeError:
            pass
    raise ValueError(f"Missing storage contract: {path} ({type(leaf).__name__})")


class BlockLoadPlan:
    def __init__(self, block, metadata, source_names, prefix=None, device_buffer=False):
        self.operators = dict(block.named_weight_leaves())
        self.auxiliary = {}
        entries = []
        for path, leaf in self.operators.items():
            description = _describe(leaf, metadata, path)
            if description.tensors and bool(leaf.create_cuda_buffer) != device_buffer:
                raise ValueError(f"Block storage role disagrees with operator buffers: {path}")
            for spec in description.tensors:
                entries.append(((path, spec.attr), spec.name, spec.shape, spec.dtype, spec.transpose))
            for attr, name in description.auxiliary:
                self.auxiliary[path, attr] = name
        self.layout = BlockLayout.build(entries)
        names = {spec.name for spec in self.layout.tensors}
        if prefix is not None:
            scoped = {name for name in source_names if name.startswith(prefix)}
            if names != scoped:
                raise ValueError(f"Unmapped or out-of-scope block weights for {prefix}: {sorted(names ^ scoped)}")
        self.device = torch.device(AI_DEVICE if device_buffer else "cpu")
        self.device_buffer = device_buffer

    def load(self, block, sources):
        backend = get_block_offload_backend()
        buffer = BlockBuffer.allocate(self.layout, self.device, backend=backend)
        context = BlockLoadContext(sources, buffer)
        for leaf in self.operators.values():
            if hasattr(leaf, "bind_storage"):
                leaf.bind_storage(context)
            elif hasattr(leaf, "load"):
                leaf.load(context)
        context.finish()
        for spec in self.layout.tensors:
            path, attr = spec.key
            leaf = self.operators[path]
            actual = getattr(leaf, f"{attr}_cuda_buffer" if self.device_buffer else f"pin_{attr}")
            expected = buffer.view(spec, operator=True)
            if (actual.data_ptr(), actual.dtype, actual.shape, actual.stride(), actual.device) != (expected.data_ptr(), expected.dtype, expected.shape, expected.stride(), expected.device):
                raise ValueError(f"Contiguous weight escaped its planned view: {spec.name}")
            if self.device.type == "cpu" and not actual.is_pinned():
                raise ValueError(f"Contiguous weight is not pinned: {spec.name}")
            if self.device_buffer:
                setattr(leaf, attr, actual)
        names = {spec.name for spec in self.layout.tensors} | set(self.auxiliary.values())
        extra = {name for name, tensor in block.state_dict().items() if tensor is not None} - names
        if extra:
            raise ValueError(f"Undeclared contiguous block state: {sorted(extra)}")
        block.block_buffer = buffer
        block.block_auxiliary = {(path, attr): getattr(self.operators[path], attr) for path, attr in self.auxiliary}


def prepare_contiguous_groups(groups, sources, metadata=None):
    """Plan every member before loaders can consume any checkpoint tensors."""
    get_block_offload_backend().prepare()
    groups = tuple(groups)
    if not groups:
        raise ValueError("The model has not declared any block offload groups")
    if metadata is None:
        metadata = {name: TensorMetadata(tuple(tensor.shape), tensor.dtype) for name, tensor in sources.items()}
    else:
        metadata = {name: replace(metadata[name], loaded_dtype=tensor.dtype) for name, tensor in sources.items()}
    plans, owned = [], set()
    for group in groups:
        group_plans = []
        for block, prefix in zip(group.blocks, group.checkpoint_prefixes):
            plan = BlockLoadPlan(block, metadata, sources.keys(), prefix)
            names = {spec.name for spec in plan.layout.tensors}
            if names & owned:
                raise ValueError(f"Overlapping offload block ownership: {sorted(names & owned)}")
            owned.update(names)
            group_plans.append((block, plan))
        for slot in group.device_slots:
            group_plans.append((slot, BlockLoadPlan(slot, metadata, sources.keys(), device_buffer=True)))
        layout = group_plans[0][1].layout
        if any(plan.layout != layout for _, plan in group_plans):
            raise ValueError("Block layouts differ within an offload group; declare separate groups")
        plans.extend(group_plans)
    for name in owned:
        if sources[name].device.type != "cpu":
            raise ValueError(f"Expected a CPU checkpoint tensor: {name}")
        if tuple(sources[name].shape) != metadata[name].shape:
            raise ValueError(f"Checkpoint shape changed before block loading: {name}")
    for block, plan in plans:
        block._block_load_plan = plan


def validate_group_checkpoints(groups, metadata):
    """Validate original dtypes even when the ordinary loader already cast them."""
    for group in groups:
        for block in group.blocks:
            for _, leaf in block.named_weight_leaves():
                validate = getattr(leaf, "validate_checkpoint", None)
                if validate is not None:
                    validate(metadata)
