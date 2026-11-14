"""Load safetensors directly into FSDP2 parameter shards."""

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


def normalize_sharded_load_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    fsdp = ((config or {}).get("distributed", {}) or {}).get("fsdp2", {}) or {}
    value = fsdp.get("sharded_load", fsdp.get("serialized_load", False))
    if isinstance(value, Mapping):
        options = dict(value)
        enabled = options.pop("enabled", True)
    else:
        enabled, options = value, {}
    return {"enabled": enabled, "options": options}


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
    templates: Mapping[str, Any] | None = None,
    index_filename: str | None = None,
    device: Any = None,
    dtype: Any = None,
    initialize_missing: Callable[[str, Any], Any] | None = None,
) -> dict[str, Any]:
    root, index_path = _resolve_index(checkpoint, index_filename)
    weight_map = _read_index(index_path)
    templates = _state_templates(module, templates)
    resolver = key_resolver or (lambda key, _: key)

    files: dict[str, list[tuple[str, str]]] = {}
    mapped = set()
    for checkpoint_key, filename in weight_map.items():
        model_key = resolver(checkpoint_key, templates)
        if model_key is None or model_key not in templates:
            continue
        files.setdefault(filename, []).append((checkpoint_key, model_key))
        mapped.add(model_key)

    initialized = {}
    if initialize_missing is not None:
        for name, template in templates.items():
            if name not in mapped:
                value = initialize_missing(name, template)
                if value is not None:
                    initialized[name] = value
        _load_state(module, initialized)

    from safetensors import safe_open

    loaded = 0
    loaded_files = []
    for filename, entries in files.items():
        state = {}
        with safe_open(str(root / filename), framework="pt", device="cpu") as shard:
            for checkpoint_key, model_key in entries:
                template = templates[model_key]
                value = shard.get_slice(checkpoint_key)
                if _is_dtensor(template):
                    local_shape, offset = local_layout_from_template(template)
                    slices = tuple(
                        slice(start, start + size)
                        for start, size in zip(offset, local_shape, strict=True)
                    )
                    value = value[slices].contiguous()
                else:
                    value = value[:] if value.get_shape() else value[()]
                value = value.to(dtype=dtype or template.dtype)
                if device is not None:
                    value = value.to(device)
                state[model_key] = make_dtensor_from_template(template, value) if _is_dtensor(template) else value
                loaded += 1
        _load_state(module, state)
        loaded_files.append(filename)

    return {
        "index_path": str(index_path),
        "files": loaded_files,
        "loaded_key_count": loaded,
        "initialized_model_keys": list(initialized),
    }


def _resolve_index(checkpoint: str | Path, index_filename: str | None):
    path = Path(checkpoint)
    if path.is_file():
        return path.parent, path
    if index_filename:
        return path, path / index_filename
    for name in _DEFAULT_INDEX_NAMES:
        candidate = path / name
        if candidate.is_file():
            return path, candidate
    candidate = next(path.glob("*.index.json"), None)
    if candidate is not None:
        return path, candidate
    return path, next(path.glob("*.safetensors"))


def _read_index(index_path: Path) -> Mapping[str, str]:
    if index_path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(index_path), framework="pt", device="cpu") as shard:
            return {key: index_path.name for key in shard.keys()}
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["weight_map"]


def _state_templates(module: Any, templates: Mapping[str, Any] | None):
    if templates is not None:
        return dict(templates)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    state, _ = get_state_dict(module, (), options=StateDictOptions(strict=False))
    return dict(state)


def _load_state(module: Any, state: Mapping[str, Any]):
    if state:
        module.load_state_dict(state, strict=False, assign=True)


def _is_dtensor(value: Any) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(value, DTensor)
