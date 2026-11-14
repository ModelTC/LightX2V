"""Load indexed safetensors into FSDP2 parameter shards."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_DEFAULT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "pytorch_model.safetensors.index.json",
)


def _identity_key(checkpoint_key, _):
    return checkpoint_key


def get_sharded_load_settings(config: Mapping[str, Any] | None) -> tuple[bool, dict[str, Any]]:
    fsdp = ((config or {}).get("distributed", {}) or {}).get("fsdp2", {}) or {}
    value = fsdp.get("sharded_load", fsdp.get("serialized_load", False))
    if isinstance(value, Mapping):
        options = dict(value)
        enabled = options.pop("enabled", True)
    else:
        enabled, options = value, {}
    return enabled, options


def make_dtensor_from_template(template: Any, local_tensor: Any) -> Any:
    from torch.distributed.tensor import DTensor

    return DTensor.from_local(
        local_tensor,
        device_mesh=template.device_mesh,
        placements=template.placements,
        run_check=False,
        shape=template.shape,
        stride=template.stride(),
    )


def local_layout_from_template(template: Any):
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    return compute_local_shape_and_global_offset(
        template.shape,
        template.device_mesh,
        template.placements,
    )


def load_fsdp2_safetensors(
    module: Any,
    checkpoint: str | Path,
    *,
    key_resolver: Callable[[str, Mapping[str, Any]], str | None] | None = None,
    index_filename: str | None = None,
    device: Any = None,
    dtype: Any = None,
    initialize_missing: Callable[[str, Any], Any] | None = None,
) -> None:
    checkpoint = Path(checkpoint)
    index_path = checkpoint / (index_filename or _find_index(checkpoint))
    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
    from torch.distributed.tensor import DTensor
    from safetensors import safe_open

    # After FSDP2 wrapping, each DTensor template describes this rank's shard.
    state_templates, _ = get_state_dict(module, (), options=StateDictOptions(strict=False))
    resolve_key = key_resolver or _identity_key
    shard_entries: dict[str, list[tuple[str, str]]] = {}
    loaded_keys = set()
    for checkpoint_key, shard_name in weight_map.items():
        model_key = resolve_key(checkpoint_key, state_templates)
        if model_key is None:
            continue
        shard_entries.setdefault(shard_name, []).append((checkpoint_key, model_key))
        loaded_keys.add(model_key)

    if initialize_missing is not None:
        missing_state = {}
        for model_key, template in state_templates.items():
            if model_key in loaded_keys:
                continue
            value = initialize_missing(model_key, template)
            if value is not None:
                missing_state[model_key] = value
        if missing_state:
            module.load_state_dict(missing_state, strict=False, assign=True)

    # Read only this rank's slice from each checkpoint tensor.
    for shard_name, entries in shard_entries.items():
        local_state = {}
        with safe_open(str(checkpoint / shard_name), framework="pt", device="cpu") as shard:
            for checkpoint_key, model_key in entries:
                template = state_templates[model_key]
                value = shard.get_slice(checkpoint_key)
                if isinstance(template, DTensor):
                    local_shape, offset = local_layout_from_template(template)
                    local_slices = tuple(
                        slice(start, start + size)
                        for start, size in zip(offset, local_shape, strict=True)
                    )
                    value = value[local_slices].contiguous()
                else:
                    if value.get_shape():
                        value = value[:]
                    else:
                        value = value[()]
                value = value.to(dtype=dtype or template.dtype)
                if device is not None:
                    value = value.to(device)
                if isinstance(template, DTensor):
                    value = make_dtensor_from_template(template, value)
                local_state[model_key] = value
        module.load_state_dict(local_state, strict=False, assign=True)


def _find_index(checkpoint: Path) -> str:
    for name in _DEFAULT_INDEX_NAMES:
        if (checkpoint / name).is_file():
            return name
    return next(checkpoint.glob("*.index.json")).name
