from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import torch


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


@dataclass(frozen=True)
class GenerationShapeIndex:
    dataset_index: int
    generation_shape: tuple[int, ...]


class GenerationShapeSampler(torch.utils.data.Sampler):
    """Sample an output shape per item while keeping data-parallel steps aligned."""

    def __init__(
        self,
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        drop_last=False,
        seed=0,
        generation_shapes=None,
    ):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.generation_shapes = parse_generation_shapes(generation_shapes)
        if self.num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {self.num_replicas}.")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")

        if self.drop_last:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = (len(self.dataset) + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        if self.num_samples == 0:
            raise ValueError("Generation-shape sampling produced no steps. Disable drop_last or add more samples.")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.drop_last:
            indices = indices[: self.total_size]
        elif self.total_size > len(indices):
            repeat = (self.total_size + len(indices) - 1) // len(indices)
            indices = (indices * repeat)[: self.total_size]
        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        sampled_shapes = self._sample_shapes(generator)
        return iter(GenerationShapeIndex(dataset_index=index, generation_shape=shape) for index, shape in zip(local_indices, sampled_shapes, strict=True))

    def _sample_shapes(self, generator):
        weights = torch.tensor(
            [shape.ratio if shape.ratio is not None else 1.0 for shape in self.generation_shapes],
            dtype=torch.double,
        )
        exact_counts = weights / weights.sum() * self.num_samples
        step_counts = exact_counts.floor().to(dtype=torch.int64)
        remainder = self.num_samples - int(step_counts.sum().item())
        if remainder:
            fractions = exact_counts - step_counts
            selected = torch.multinomial(
                fractions,
                remainder,
                replacement=False,
                generator=generator,
            )
            step_counts[selected] += 1

        sampled_shapes = []
        for shape, step_count in zip(self.generation_shapes, step_counts.tolist(), strict=True):
            sampled_shapes.extend([shape.value] * step_count)
        if len(sampled_shapes) > 1:
            order = torch.randperm(len(sampled_shapes), generator=generator).tolist()
            sampled_shapes = [sampled_shapes[position] for position in order]
        return sampled_shapes

    def __len__(self):
        return self.num_samples


def split_generation_shape_index(index) -> tuple[int, tuple[int, ...] | None]:
    if isinstance(index, GenerationShapeIndex):
        return index.dataset_index, index.generation_shape
    return int(index), None


def apply_generation_shape(sample: dict, generation_shape: tuple[int, ...] | None) -> None:
    if generation_shape is not None:
        sample["generation_shape"] = tuple(int(dimension) for dimension in generation_shape)


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


def normalize_generation_shape(value, *, key: str = "generation_shape") -> tuple[int, ...]:
    if torch.is_tensor(value):
        dimensions = value.detach().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        dimensions = value
    else:
        raise ValueError(f"Generation shape metadata {key} must be a sequence, got {value!r}.")
    return tuple(_scalar(dimension, key) for dimension in dimensions)


def generation_shape_key(value) -> str:
    return "x".join(str(dimension) for dimension in normalize_generation_shape(value))


def resolve_generation_shape(
    entries,
    sample: Mapping,
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
    sampled_shape = sample.get("generation_shape")
    if sampled_shape is not None:
        selected = normalize_generation_shape(sampled_shape)
        if len(selected) != expected_dimensions:
            raise ValueError(f"Sampled generation shape must have {expected_dimensions} dimensions, got {sampled_shape!r}.")
    elif len(shapes) == 1:
        selected = shapes[0].value
    else:
        raise ValueError(f"Multiple {config_path} entries require generation-shape sampling to be enabled.")

    selected = tuple(int(broadcast(dimension)) for dimension in selected)
    configured = {shape.value for shape in shapes}
    if selected not in configured:
        available = ", ".join(str(list(shape)) for shape in sorted(configured))
        raise ValueError(f"Sample generation shape {list(selected)} is not in {config_path}: [{available}].")
    return selected
