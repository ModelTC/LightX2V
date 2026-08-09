"""Pack block weights into one contiguous buffer for faster offload."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import torch

DEFAULT_SLAB_ALIGNMENT = 16


@dataclass(frozen=True)
class SlabEntry:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class BlockSlabLayout:
    entries: dict[str, SlabEntry]
    nbytes: int
    alignment: int


@dataclass(frozen=True)
class BlockSlab:
    raw: torch.Tensor
    views: dict[str, torch.Tensor]
    layout: BlockSlabLayout

    @property
    def is_pinned(self) -> bool:
        return self.raw.device.type == "cpu" and self.raw.is_pinned()


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _check_buffer(buffer, nbytes, name):
    if buffer.dtype != torch.uint8 or buffer.ndim != 1 or not buffer.is_contiguous():
        raise ValueError(f"{name} must be a contiguous 1-D uint8 tensor")
    if buffer.numel() < nbytes:
        raise ValueError(f"{name} is too small: need {nbytes} bytes, got {buffer.numel()}")


def build_block_slab_layout(
    tensors: Mapping[str, torch.Tensor],
    alignment: int = DEFAULT_SLAB_ALIGNMENT,
) -> BlockSlabLayout:
    if alignment <= 0:
        raise ValueError(f"alignment must be a positive integer, got {alignment!r}")

    entries = {}
    offset = 0
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name!r} is not a tensor")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"{name!r} cannot be packed into a slab")

        item_alignment = math.lcm(alignment, tensor.element_size())
        offset = _align_up(offset, item_alignment)
        nbytes = tensor.numel() * tensor.element_size()
        entries[name] = SlabEntry(
            offset=offset,
            nbytes=nbytes,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
        )
        offset += nbytes

    total_nbytes = max(_align_up(offset, alignment), 1)
    return BlockSlabLayout(entries, total_nbytes, alignment)


def carve_block_slab(raw: torch.Tensor, layout: BlockSlabLayout) -> dict[str, torch.Tensor]:
    """Return tensor views that share storage with ``raw``."""

    _check_buffer(raw, layout.nbytes, "raw slab")
    views = {}
    for name, entry in layout.entries.items():
        data = raw[entry.offset : entry.offset + entry.nbytes]
        views[name] = data.view(entry.dtype).view(entry.shape)
    return views


def _allocate_cpu_buffer(nbytes, pin_memory, strict_pin):
    if not pin_memory:
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu")

    try:
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    except RuntimeError as error:
        message = f"failed to allocate a {nbytes}-byte pinned block slab: {error}"
        if strict_pin:
            raise RuntimeError(message) from error
        warnings.warn(
            f"{message}; falling back to regular CPU memory",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu")


def pack_cpu_block_slab(
    tensors: Mapping[str, torch.Tensor],
    layout: BlockSlabLayout | None = None,
    alignment: int = DEFAULT_SLAB_ALIGNMENT,
    pin_memory: bool = True,
    strict_pin: bool = False,
) -> BlockSlab:
    """Copy a block's CPU tensors into one contiguous buffer."""

    if layout is None:
        layout = build_block_slab_layout(tensors, alignment=alignment)

    names = set(tensors)
    expected_names = set(layout.entries)
    if names != expected_names:
        missing = sorted(expected_names - names)
        unexpected = sorted(names - expected_names)
        raise ValueError(f"tensor names do not match layout (missing={missing}, unexpected={unexpected})")

    for name, entry in layout.entries.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name!r} is not a tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name!r} must be on CPU, got {tensor.device}")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"{name!r} cannot be packed into a slab")
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(f"{name!r} has shape {tuple(tensor.shape)}, expected {entry.shape}")
        if tensor.dtype != entry.dtype:
            raise ValueError(f"{name!r} has dtype {tensor.dtype}, expected {entry.dtype}")

    raw = _allocate_cpu_buffer(layout.nbytes, pin_memory, strict_pin)
    views = carve_block_slab(raw, layout)
    for name, view in views.items():
        view.copy_(tensors[name].detach().contiguous())

    return BlockSlab(raw, views, layout)


def allocate_block_slab_slot(
    layout: BlockSlabLayout,
    device: torch.device | str,
    slot_nbytes: int | None = None,
) -> BlockSlab:
    nbytes = layout.nbytes if slot_nbytes is None else slot_nbytes
    if nbytes < layout.nbytes:
        raise ValueError(f"slot_nbytes is too small: need {layout.nbytes}, got {nbytes!r}")

    raw = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return BlockSlab(raw, carve_block_slab(raw, layout), layout)


def copy_block_slab_(
    destination_raw: torch.Tensor,
    source_raw: torch.Tensor,
    nbytes: int | None = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Copy one slab. Stream ordering is handled by the caller."""

    size = source_raw.numel() if nbytes is None else nbytes
    if size < 0:
        raise ValueError(f"nbytes must be a non-negative integer, got {size!r}")

    _check_buffer(source_raw, size, "source slab")
    _check_buffer(destination_raw, size, "destination slab")
    destination_raw[:size].copy_(source_raw[:size], non_blocking=non_blocking)
    return destination_raw
