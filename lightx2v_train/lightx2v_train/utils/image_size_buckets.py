from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class ImageSizeBucket:
    value: tuple[int, ...]
    ratio: float | None

    @property
    def spatial_size(self) -> tuple[int, int]:
        return self.value[-2], self.value[-1]


def parse_image_size_buckets(
    entries,
    *,
    config_path: str = "training.dmd.image_sizes",
) -> list[ImageSizeBucket]:
    """Parse the only supported image-size bucket schemas.

    Every entry is either ``{"value": [...]}`` or
    ``{"value": [...], "ratio": number}``. A configuration cannot mix the
    weighted and unweighted forms. The final two values are always interpreted
    as pixel height and width.
    """
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise TypeError(f"{config_path} must be a list, got {type(entries).__name__}.")

    buckets = []
    ratio_modes = set()
    spatial_sizes = set()
    for index, entry in enumerate(entries):
        entry_path = f"{config_path}[{index}]"
        if not isinstance(entry, Mapping):
            raise TypeError(f"{entry_path} must be {{'value': [...]}} or {{'value': [...], 'ratio': number}}; legacy bare lists are not supported.")
        keys = set(entry)
        if keys not in ({"value"}, {"value", "ratio"}):
            raise ValueError(f"{entry_path} must contain exactly 'value' or 'value' and 'ratio', got keys {sorted(keys)}.")

        value = entry["value"]
        if not isinstance(value, list) or len(value) not in {2, 3}:
            raise ValueError(f"{entry_path}.value must be [height, width] or [prefix, height, width], got {value!r}.")
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

        bucket = ImageSizeBucket(value=normalized_value, ratio=ratio)
        if bucket.spatial_size in spatial_sizes:
            height, width = bucket.spatial_size
            raise ValueError(f"{config_path} contains duplicate spatial bucket {height}x{width}.")
        spatial_sizes.add(bucket.spatial_size)
        buckets.append(bucket)

    if len(ratio_modes) > 1:
        raise ValueError(f"{config_path} cannot mix entries with and without ratio; use one schema consistently.")
    return buckets
