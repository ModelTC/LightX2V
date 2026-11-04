"""Load resident weight containers directly into CPU-visible MPS storage."""

import torch
from loguru import logger

from lightx2v.common.offload.mps_manager import host_view


def weight_bindings(root):
    seen, stack = set(), [root]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        stack.extend(getattr(obj, "_modules", {}).values())
        stack.extend(getattr(obj, "_parameters", {}).values())
        for name, attr, transpose in getattr(obj, "base_attrs", ()):
            if attr == "weight":
                transpose = getattr(obj, "weight_need_transpose", transpose)
            yield obj, name, attr, transpose


def load_resident_groups(checkpoint, groups):
    """Adopt exact checkpoint dtypes, with no retained host weight copies.

    Groups must partition the selected checkpoint. CPU aliases exist only while
    reading idle allocations; synchronization orders both sides of host writes.
    Container.load must adopt storage or a transpose view without conversion.
    """
    groups = list(groups)
    names = [name for _, root in groups for _, name, _, _ in weight_bindings(root)]
    if len(names) != len(set(names)) or set(names) != set(checkpoint.weight_map):
        raise ValueError("Resident checkpoint and model bindings must match exactly")
    total = 0
    for label, root in groups:
        bindings = list(weight_bindings(root))
        tensors = {}
        for _, name, _, _ in bindings:
            dtype, shape, _, size = checkpoint.tensor_metadata(name)
            tensors[name] = torch.empty(shape, dtype=dtype, device="mps")
            total += size
        torch.mps.synchronize()
        aliases = {name: host_view(tensor) for name, tensor in tensors.items()}
        checkpoint.load_tensors_into(aliases)
        torch.mps.synchronize()
        del aliases
        root.load(tensors)
        for obj, name, attr, transpose in bindings:
            actual, original = getattr(obj, attr, None), tensors[name]
            shape = tuple(reversed(original.shape)) if transpose else tuple(original.shape)
            if (
                not isinstance(actual, torch.Tensor)
                or actual.device.type != "mps"
                or actual.dtype != original.dtype
                or tuple(actual.shape) != shape
                or actual.untyped_storage()._cdata != original.untyped_storage()._cdata
                or getattr(obj, "pin_" + attr, None) is not None
            ):
                raise RuntimeError("Resident loader requires exact storage adoption: " + name)
        logger.info("MPS resident {} loaded; {:.3f} GiB cumulative weights", label, total / 1024**3)
    return {"weight_bytes": total, "weight_count": len(names)}
