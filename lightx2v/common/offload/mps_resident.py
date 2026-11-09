"""Load resident weights directly into CPU-visible MPS allocations."""

import torch
from loguru import logger

from lightx2v.common.modules.weight_module import WeightModule
from lightx2v.common.offload.mps_manager import host_view
from lightx2v_platform.ops.offload.weight_storage import TensorMetadata, describe_weight_storage


def load_resident_groups(checkpoint, groups):
    """Load disjoint weight groups using their operators' storage contracts.

    All selected checkpoint tensors must be declared exactly once. Checkpoint
    and operator dtypes must match so loading can adopt each device allocation
    directly. The caller supplies newly created, resident weight containers.
    """
    metadata = {}
    for name in checkpoint.weight_map:
        dtype, shape, _, _ = checkpoint.tensor_metadata(name)
        metadata[name] = TensorMetadata(shape, dtype)

    plans = []
    declared_names = []
    for label, root in groups:
        operators = root.named_weight_leaves() if isinstance(root, WeightModule) else ((label, root),)
        bindings = []
        for path, operator in operators:
            description = describe_weight_storage(operator, metadata, path)
            for spec in description.tensors:
                source = metadata[spec.name]
                if (source.dtype, source.shape) != (spec.dtype, spec.shape):
                    raise ValueError(f"MPS resident loading requires checkpoint storage to match the operator: {spec.name}")
                bindings.append((operator, spec))
                declared_names.append(spec.name)
        plans.append((label, root, bindings))

    if len(declared_names) != len(set(declared_names)) or set(declared_names) != set(checkpoint.weight_map):
        raise ValueError("Resident checkpoint and model bindings must match exactly")

    total_bytes = 0
    for label, root, bindings in plans:
        tensors = {spec.name: torch.empty(spec.shape, dtype=spec.dtype, device="mps") for _, spec in bindings}
        torch.mps.synchronize()
        aliases = {name: host_view(tensor) for name, tensor in tensors.items()}
        checkpoint.load_tensors_into(aliases)
        torch.mps.synchronize()
        del aliases
        root.load(tensors)
        for operator, spec in bindings:
            actual = getattr(operator, spec.attr)
            source = tensors[spec.name]
            expected = source.t() if spec.transpose else source
            if (
                not isinstance(actual, torch.Tensor)
                or (actual.device, actual.dtype, actual.shape, actual.stride(), actual.data_ptr()) != (expected.device, expected.dtype, expected.shape, expected.stride(), expected.data_ptr())
                or getattr(operator, "pin_" + spec.attr, None) is not None
            ):
                raise RuntimeError("Resident loader requires exact storage adoption: " + spec.name)
        total_bytes += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        logger.info("MPS resident {} loaded; {:.3f} GiB cumulative weights", label, total_bytes / 1024**3)
    return {"weight_bytes": total_bytes, "weight_count": len(declared_names)}
