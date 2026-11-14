"""Memory bounded loading of Hugging Face safetensors into FSDP2 models.

The loader in this module deliberately knows nothing about a model's LoRA
layout, positional encodings, or other model-specific state.  A model adapter
may resolve checkpoint names and initialize keys which are intentionally absent
from the pretrained checkpoint, while this module only handles the generic
DTensor/safetensors mechanics.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


_DEFAULT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
    "pytorch_model.safetensors.index.json",
)


def normalize_sharded_load_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize the generic ``fsdp2.sharded_load`` option.

    ``serialized_load`` is retained as a compatibility alias for existing
    configurations.  The returned dictionary is intentionally JSON serializable
    so callers can include it in run metadata.
    """

    fsdp_config = ((config or {}).get("distributed", {}) or {}).get("fsdp2", {}) or {}
    if not isinstance(fsdp_config, Mapping):
        raise ValueError("distributed.fsdp2 must be a mapping")

    has_new = "sharded_load" in fsdp_config
    has_old = "serialized_load" in fsdp_config
    deprecated_alias = False
    if has_new:
        value = fsdp_config["sharded_load"]
        if isinstance(value, bool):
            options: dict[str, Any] = {}
            enabled = value
        elif isinstance(value, Mapping):
            options = dict(value)
            enabled_value = options.pop("enabled", True)
            if not isinstance(enabled_value, bool):
                raise ValueError("distributed.fsdp2.sharded_load.enabled must be a bool")
            enabled = enabled_value
        else:
            raise ValueError("distributed.fsdp2.sharded_load must be a bool or mapping")
        unsupported = set(options) - {"index_filename", "strict", "collect_garbage"}
        if unsupported:
            names = ", ".join(sorted(repr(key) for key in unsupported))
            raise ValueError(f"Unsupported distributed.fsdp2.sharded_load options: {names}.")
        if "index_filename" in options and (
            not isinstance(options["index_filename"], str) or not options["index_filename"]
        ):
            raise ValueError("distributed.fsdp2.sharded_load.index_filename must be a nonempty string")
        for key in ("strict", "collect_garbage"):
            if key in options and type(options[key]) is not bool:
                raise ValueError(f"distributed.fsdp2.sharded_load.{key} must be a bool")
        if has_old:
            old_value = fsdp_config["serialized_load"]
            if not isinstance(old_value, bool):
                raise ValueError("distributed.fsdp2.serialized_load must be a bool")
            if old_value != enabled:
                raise ValueError(
                    "distributed.fsdp2.sharded_load conflicts with deprecated serialized_load"
                )
            deprecated_alias = True
        source = "sharded_load"
    elif has_old:
        old_value = fsdp_config["serialized_load"]
        if not isinstance(old_value, bool):
            raise ValueError("distributed.fsdp2.serialized_load must be a bool")
        enabled = old_value
        options = {}
        source = "serialized_load"
        deprecated_alias = True
    else:
        enabled = False
        options = {}
        source = "default"

    return {
        "enabled": enabled,
        "options": options,
        "source": source,
        "deprecated_alias": deprecated_alias,
    }


def sharded_load_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether FSDP2 sharded loading is enabled for ``config``."""

    return bool(normalize_sharded_load_config(config)["enabled"])


# Transitional descriptive alias used by model adapters.
fsdp2_sharded_load_enabled = sharded_load_enabled


def make_dtensor_from_template(template: Any, local_tensor: Any) -> Any:
    """Create a DTensor that preserves a FSDP2 state-dict template.

    The helper is also useful to model adapters when initializing intentionally
    absent keys (for example LoRA A/B parameters).  It does not initialize or
    alter ``local_tensor`` and does not perform a collective validation.
    """

    from torch.distributed.tensor import DTensor

    if not isinstance(template, DTensor):
        raise TypeError(f"Expected a DTensor template, got {type(template).__name__}")
    local_shape = tuple(_compute_local_shape_and_offset(template)[0])
    if tuple(local_tensor.shape) != local_shape:
        raise ValueError(
            f"Local tensor shape {tuple(local_tensor.shape)} does not match DTensor local shape {local_shape}"
        )
    return DTensor.from_local(
        local_tensor,
        device_mesh=template.device_mesh,
        placements=template.placements,
        run_check=False,
        shape=template.shape,
        stride=template.stride(),
    )


def load_fsdp2_safetensors(
    module: Any,
    checkpoint: str | os.PathLike[str],
    *,
    key_resolver: Callable[[str, Mapping[str, Any]], str | None] | None = None,
    templates: Mapping[str, Any] | None = None,
    index_filename: str | None = None,
    device: Any = None,
    dtype: Any = None,
    strict: bool = True,
    ignored_checkpoint_keys: Iterable[str] = (),
    ignored_model_keys: Iterable[str] = (),
    initialize_missing: Callable[[str, Any], Any] | None = None,
    collect_garbage: bool = True,
) -> dict[str, Any]:
    """Load only each rank's FSDP2 parameter slice from HF safetensors.

    ``templates`` are normally obtained from ``torch.distributed.checkpoint``
    after FSDP2 wrapping.  Supplying them is supported for adapters and tests.
    ``key_resolver`` maps a checkpoint key to the model state-dict key; return
    ``None`` only for a key listed in ``ignored_checkpoint_keys``.  If the
    checkpoint and model dtypes differ, pass an explicit ``dtype`` to request a
    conversion.  With ``dtype=None`` strict mode requires exact dtype equality.

    The function opens and loads one safetensors file at a time.  It does not
    materialize a full checkpoint on CPU, and it leaves model-specific missing
    key initialization to ``initialize_missing``.
    """

    root, index_path = _resolve_index_path(checkpoint, index_filename)
    index = _read_index(index_path)
    weight_map = index["weight_map"]
    ignored_checkpoint = set(ignored_checkpoint_keys)
    ignored_model = set(ignored_model_keys)
    if any(not isinstance(key, str) for key in ignored_checkpoint):
        raise TypeError("ignored_checkpoint_keys must contain strings")
    if any(not isinstance(key, str) for key in ignored_model):
        raise TypeError("ignored_model_keys must contain strings")

    if templates is None:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        templates, _ = get_state_dict(
            module,
            (),
            options=StateDictOptions(strict=False),
        )
    templates = dict(templates)
    if not templates:
        raise ValueError("FSDP2 state-dict templates are empty")

    resolver = key_resolver or (lambda key, _templates: key)
    mapped: dict[str, str] = {}
    mapped_reverse: dict[str, str] = {}
    ignored_checkpoint_found: list[str] = []
    for checkpoint_key in weight_map:
        if checkpoint_key in ignored_checkpoint:
            ignored_checkpoint_found.append(checkpoint_key)
            continue
        model_key = resolver(checkpoint_key, templates)
        if model_key is None:
            raise KeyError(
                f"Checkpoint key {checkpoint_key!r} was not resolved; add it to ignored_checkpoint_keys"
            )
        if not isinstance(model_key, str) or not model_key:
            raise TypeError(f"key_resolver returned invalid key for {checkpoint_key!r}: {model_key!r}")
        if model_key in mapped_reverse:
            previous = mapped_reverse[model_key]
            raise ValueError(
                f"Multiple checkpoint keys resolve to model key {model_key!r}: {previous!r}, {checkpoint_key!r}"
            )
        if model_key not in templates:
            raise KeyError(
                f"Checkpoint key {checkpoint_key!r} resolves to missing model key {model_key!r}"
            )
        mapped[checkpoint_key] = model_key
        mapped_reverse[model_key] = checkpoint_key

    missing = [
        key for key in templates
        if key not in mapped.values() and key not in ignored_model
    ]
    initialized: list[str] = []
    initial_state: dict[str, Any] = {}
    if initialize_missing is not None:
        for key in missing:
            value = initialize_missing(key, templates[key])
            if value is None:
                continue
            _validate_initialized_value(key, templates[key], value)
            initial_state[key] = value
            initialized.append(key)
        missing = [key for key in missing if key not in initial_state]
    missing_model_keys = list(missing)
    if strict and missing:
        raise KeyError(f"Model state-dict keys are absent from checkpoint: {missing[:8]!r}")

    # Validate all paths before touching the module, so a malformed index cannot
    # leave a partially loaded model.
    files = _validate_weight_map(root, weight_map)
    safe_open = _get_safe_open()
    target_device = device
    target_dtype = _normalize_dtype(dtype)
    if initial_state:
        _load_state(module, initial_state)

    loaded = 0
    dtype_casts = 0
    loaded_files: list[str] = []
    for filename, checkpoint_keys in _group_by_file(weight_map, mapped).items():
        shard_state: dict[str, Any] = {}
        shard_path = root / filename
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for checkpoint_key in checkpoint_keys:
                model_key = mapped[checkpoint_key]
                template = templates[model_key]
                tensor_slice = shard.get_slice(checkpoint_key)
                source_shape = tuple(tensor_slice.get_shape())
                expected_shape = tuple(_global_shape(template))
                if source_shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {checkpoint_key!r} -> {model_key!r}: "
                        f"checkpoint={source_shape}, model={expected_shape}"
                    )
                source_dtype = _normalize_dtype(tensor_slice.get_dtype())
                effective_dtype = target_dtype or _template_dtype(template)
                if target_dtype is None and source_dtype != effective_dtype:
                    raise TypeError(
                        f"Dtype mismatch for {checkpoint_key!r} -> {model_key!r}: "
                        f"checkpoint={source_dtype}, model={effective_dtype}; pass dtype to cast explicitly"
                    )
                if source_dtype != effective_dtype:
                    dtype_casts += 1

                if _is_dtensor(template):
                    local_shape, global_offset = _compute_local_shape_and_offset(template)
                    slices = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(global_offset, local_shape, strict=True)
                    )
                    local_tensor = tensor_slice[slices].contiguous()
                    if tuple(local_tensor.shape) != tuple(local_shape):
                        raise ValueError(
                            f"Local slice shape mismatch for {checkpoint_key!r}: "
                            f"actual={tuple(local_tensor.shape)}, expected={tuple(local_shape)}"
                        )
                else:
                    local_tensor = tensor_slice[:] if source_shape else tensor_slice[()]
                if target_device is not None or effective_dtype is not None:
                    local_tensor = local_tensor.to(device=target_device, dtype=effective_dtype)
                shard_state[model_key] = (
                    make_dtensor_from_template(template, local_tensor)
                    if _is_dtensor(template)
                    else local_tensor
                )
                loaded += 1
        _load_state(module, shard_state)
        loaded_files.append(filename)
        del shard_state
        if collect_garbage:
            gc.collect()

    return {
        "enabled": True,
        "index_path": str(index_path),
        "files": loaded_files,
        "checkpoint_key_count": len(weight_map),
        "mapped_key_count": len(mapped),
        "loaded_key_count": loaded,
        "ignored_checkpoint_keys": sorted(ignored_checkpoint_found),
        "initialized_model_keys": initialized,
        "ignored_model_keys": sorted(ignored_model),
        "missing_model_keys": missing_model_keys,
        "dtype_cast_count": dtype_casts,
        "strict": strict,
    }


def _resolve_index_path(checkpoint: str | os.PathLike[str], index_filename: str | None):
    path = Path(checkpoint).expanduser().resolve()
    if path.is_file():
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if index_filename is not None:
        candidate = (path / index_filename).resolve()
        if candidate.parent != path or not candidate.is_file():
            raise FileNotFoundError(f"Safetensors index does not exist below checkpoint directory: {candidate}")
        return path, candidate
    candidates = [path / name for name in _DEFAULT_INDEX_NAMES if (path / name).is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Specify index_filename; expected exactly one index in {path}, found {len(candidates)}"
        )
    return path, candidates[0]


def _read_index(index_path: Path) -> dict[str, Any]:
    # Hugging Face also permits an unindexed, single safetensors file.  Treat
    # every key in that file as belonging to the file itself.  This keeps the
    # generic API useful for small checkpoints while preserving the same strict
    # key/shape/dtype validation as indexed checkpoints.
    if index_path.suffix == ".safetensors":
        safe_open = _get_safe_open()
        with safe_open(str(index_path), framework="pt", device="cpu") as shard:
            keys = list(shard.keys())
        return {"weight_map": {key: index_path.name for key in keys}, "single_file": True}
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors index {index_path}: {exc}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"Safetensors index {index_path} must contain a weight_map object")
    if not index["weight_map"] or any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in index["weight_map"].items()
    ):
        raise ValueError("Safetensors weight_map must be a non-empty string-to-string mapping")
    return index


def _validate_weight_map(root: Path, weight_map: Mapping[str, str]) -> set[str]:
    files = set()
    for filename in weight_map.values():
        path = (root / filename).resolve()
        if path.parent != root or path.name != filename or not path.is_file():
            raise FileNotFoundError(f"Safetensors shard must be a file directly below checkpoint directory: {path}")
        files.add(filename)
    return files


def _group_by_file(weight_map: Mapping[str, str], mapped: Mapping[str, str]):
    grouped: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if key in mapped:
            grouped.setdefault(filename, []).append(key)
    return grouped


def _get_safe_open():
    from safetensors import safe_open

    return safe_open


def _load_state(module: Any, state: Mapping[str, Any]) -> None:
    if not state:
        return
    incompatible = module.load_state_dict(state, strict=False, assign=True)
    unexpected = getattr(incompatible, "unexpected_keys", ())
    if unexpected:
        raise RuntimeError(f"FSDP2 state-dict load returned unexpected keys: {list(unexpected)!r}")


def _is_dtensor(value: Any) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(value, DTensor)


def _global_shape(template: Any) -> tuple[int, ...]:
    shape = getattr(template, "shape", None)
    if shape is None:
        raise TypeError(f"State-dict template {type(template).__name__} has no shape")
    return tuple(shape)


def _template_dtype(template: Any):
    dtype = getattr(template, "dtype", None)
    if dtype is None:
        raise TypeError(f"State-dict template {type(template).__name__} has no dtype")
    return _normalize_dtype(dtype)


def _normalize_dtype(dtype: Any):
    if dtype is None:
        return None
    if not isinstance(dtype, str):
        return dtype
    import torch

    mapping = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
        "C64": torch.complex64,
        "C128": torch.complex128,
    }
    if dtype in mapping:
        return mapping[dtype]
    raise TypeError(f"Unsupported safetensors dtype {dtype!r}")


def _compute_local_shape_and_offset(template: Any):
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    return compute_local_shape_and_global_offset(
        template.shape,
        template.device_mesh,
        template.placements,
    )


def local_layout_from_template(template: Any, global_shape: Iterable[int] | None = None):
    """Return this rank's local shape and global offset for a DTensor template."""
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape = template.shape if global_shape is None else global_shape
    return compute_local_shape_and_global_offset(
        shape,
        template.device_mesh,
        template.placements,
    )


def _validate_initialized_value(name: str, template: Any, value: Any) -> None:
    if _is_dtensor(template):
        if not _is_dtensor(value):
            raise TypeError(f"initialize_missing({name!r}) must return a DTensor")
        if tuple(value.shape) != _global_shape(template):
            raise ValueError(f"Initialized DTensor shape mismatch for {name!r}")
        if value.dtype != template.dtype:
            raise TypeError(f"Initialized DTensor dtype mismatch for {name!r}")
        return
    if tuple(getattr(value, "shape", ())) != _global_shape(template):
        raise ValueError(f"Initialized tensor shape mismatch for {name!r}")
    if getattr(value, "dtype", None) != getattr(template, "dtype", None):
        raise TypeError(f"Initialized tensor dtype mismatch for {name!r}")
