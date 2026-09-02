from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class GenerationShape:
    value: tuple[int, ...]
    ratio: float | None

    @property
    def dimensions(self) -> int:
        return len(self.value)

    @property
    def spatial_size(self) -> tuple[int, int]:
        return self.value[-2], self.value[-1]


def parse_generation_shapes(
    entries,
    *,
    expected_dimensions: int | None = None,
    config_path: str = "training.dmd.generation_shapes",
) -> list[GenerationShape]:
    """Parse output shapes expressed as ``[H, W]`` or ``[T, H, W]``."""
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{config_path} must be a non-empty list.")
    if expected_dimensions not in {None, 2, 3}:
        raise ValueError(f"expected_dimensions must be 2 or 3, got {expected_dimensions!r}.")

    shapes = []
    ratio_modes = set()
    configured_values = set()
    configured_dimensions = set()
    for index, entry in enumerate(entries):
        entry_path = f"{config_path}[{index}]"
        if not isinstance(entry, Mapping):
            raise TypeError(f"{entry_path} must be {{'value': [...]}} or {{'value': [...], 'ratio': number}}.")
        keys = set(entry)
        if keys not in ({"value"}, {"value", "ratio"}):
            raise ValueError(f"{entry_path} must contain exactly 'value' or 'value' and 'ratio', got keys {sorted(keys)}.")

        value = entry["value"]
        if not isinstance(value, list) or len(value) not in {2, 3}:
            raise ValueError(f"{entry_path}.value must be [height, width] or [num_frames, height, width], got {value!r}.")
        if expected_dimensions is not None and len(value) != expected_dimensions:
            expected = "[height, width]" if expected_dimensions == 2 else "[num_frames, height, width]"
            raise ValueError(f"{entry_path}.value must be {expected}, got {value!r}.")
        if any(not isinstance(item, Integral) or isinstance(item, bool) for item in value):
            raise TypeError(f"{entry_path}.value must contain integers, got {value!r}.")
        normalized_value = tuple(int(item) for item in value)
        if any(item <= 0 for item in normalized_value):
            raise ValueError(f"{entry_path}.value must contain positive integers, got {value!r}.")

        has_ratio = "ratio" in entry
        ratio_modes.add(has_ratio)
        ratio = None
        if has_ratio:
            raw_ratio = entry["ratio"]
            if not isinstance(raw_ratio, Real) or isinstance(raw_ratio, bool):
                raise TypeError(f"{entry_path}.ratio must be a positive number, got {raw_ratio!r}.")
            ratio = float(raw_ratio)
            if not math.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"{entry_path}.ratio must be finite and positive, got {raw_ratio!r}.")

        if normalized_value in configured_values:
            raise ValueError(f"{config_path} contains duplicate shape {list(normalized_value)}.")
        configured_values.add(normalized_value)
        configured_dimensions.add(len(normalized_value))
        shapes.append(GenerationShape(value=normalized_value, ratio=ratio))

    if len(configured_dimensions) > 1:
        raise ValueError(f"{config_path} cannot mix two-dimensional image shapes and three-dimensional video shapes.")
    if len(ratio_modes) > 1:
        raise ValueError(f"{config_path} cannot mix entries with and without ratio; use one schema consistently.")
    return shapes


def _scalar(value, key: str) -> int:
    if hasattr(value, "detach"):
        values = value.detach().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = [item.item() if hasattr(item, "item") else item for item in value]
    else:
        values = [value]
    if len(values) != 1:
        raise ValueError(f"Generation shape metadata {key} must contain exactly one value, got {values}.")
    result = int(values[0])
    if result <= 0:
        raise ValueError(f"Generation shape metadata {key} must be positive, got {result}.")
    return result


def resolve_generation_shape(
    entries,
    metadata: Mapping,
    *,
    expected_dimensions: int,
    broadcast: Callable[[int], int],
    config_path: str = "training.dmd.generation_shapes",
) -> tuple[int, ...]:
    """Resolve one configured generation shape for the current sample."""
    shapes = parse_generation_shapes(
        entries,
        expected_dimensions=expected_dimensions,
        config_path=config_path,
    )
    metadata_keys = (
        ("target_height", "target_width")
        if expected_dimensions == 2
        else ("target_num_frames", "target_height", "target_width")
    )
    present = [key in metadata and metadata[key] is not None for key in metadata_keys]
    if any(present) and not all(present):
        missing = [key for key, is_present in zip(metadata_keys, present, strict=True) if not is_present]
        raise ValueError(f"Generation shape metadata is incomplete; missing: {', '.join(missing)}.")

    if all(present):
        selected = tuple(_scalar(metadata[key], key) for key in metadata_keys)
    elif len(shapes) == 1:
        selected = shapes[0].value
    else:
        required = ", ".join(metadata_keys)
        raise ValueError(f"Multiple {config_path} entries require every prompt sample to provide {required}.")

    selected = tuple(int(broadcast(dimension)) for dimension in selected)
    configured = {shape.value for shape in shapes}
    if selected not in configured:
        available = ", ".join(str(list(shape)) for shape in sorted(configured))
        raise ValueError(f"Sample generation shape {list(selected)} is not in {config_path}: [{available}].")
    return selected
