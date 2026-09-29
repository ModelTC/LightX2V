"""Operator-owned storage descriptions, independent of models and devices."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    loaded_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class WeightStorage:
    name: str
    attr: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    transpose: bool = False


@dataclass(frozen=True)
class StorageDescription:
    tensors: tuple[WeightStorage, ...] = ()
    # (attribute, state-dict name) of device-resident state copied with a block.
    auxiliary: tuple[tuple[str, str], ...] = ()


def require_dtype(name, dtype, accepted):
    if dtype not in accepted:
        raise ValueError(f"{name}: expected checkpoint dtype in {accepted}, got {dtype}; implicit quantization/dequantization is not supported")


class FloatingWeightStorage:
    """Storage contract for ordinary floating weights using base_attrs."""

    def validate_checkpoint(self, metadata):
        for name, _, _ in self.base_attrs:
            require_dtype(name, metadata[name].dtype, (torch.float16, torch.bfloat16, torch.float32))

    def describe_storage(self, metadata):
        self.validate_checkpoint(metadata)
        return StorageDescription(tuple(WeightStorage(name, attr, metadata[name].shape, self.infer_dtype, transpose) for name, attr, transpose in self.base_attrs))
