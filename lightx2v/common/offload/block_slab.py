"""Contiguous byte slabs for block-level weight offload.

The utilities in this module turn a ``{name: tensor}`` mapping into one
contiguous CPU ``uint8`` allocation.  The same layout can be carved from a raw
slot on any torch device, so transferring a block requires a single raw
``copy_`` instead of one copy per tensor.

Stream and event ordering deliberately live outside this module.  Callers
should invoke :func:`copy_block_slab_` in their device's copy-stream context
and record the ready event after it returns.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import torch

DEFAULT_SLAB_ALIGNMENT = 16


@dataclass(frozen=True)
class SlabEntry:
    """The byte range and tensor metadata for one value in a slab."""

    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class BlockSlabLayout:
    """An aligned, non-overlapping layout for a block of tensors."""

    entries: dict[str, SlabEntry]
    nbytes: int
    alignment: int


@dataclass(frozen=True)
class BlockSlab:
    """A raw slab together with typed views carved from its leading bytes."""

    raw: torch.Tensor
    views: dict[str, torch.Tensor]
    layout: BlockSlabLayout

    @property
    def is_pinned(self) -> bool:
        """Whether the raw allocation is pinned host memory."""

        return self.raw.device.type == "cpu" and self.raw.is_pinned()


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _validate_alignment(alignment: int) -> None:
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise ValueError(f"alignment must be a positive integer, got {alignment!r}")


def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * dtype.itemsize


def build_block_slab_layout(
    tensors: Mapping[str, torch.Tensor],
    *,
    alignment: int = DEFAULT_SLAB_ALIGNMENT,
) -> BlockSlabLayout:
    """Build an aligned layout while preserving the mapping's iteration order.

    Every entry is aligned to both ``alignment`` and its dtype element size.
    This keeps typed views valid even when a caller chooses an alignment that
    is not a multiple of every element size.
    """

    _validate_alignment(alignment)
    entries: dict[str, SlabEntry] = {}
    cursor = 0

    for name, tensor in tensors.items():
        if not isinstance(name, str):
            raise TypeError(f"tensor names must be strings, got {type(name).__name__}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor {name!r} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.layout != torch.strided:
            raise ValueError(f"tensor {name!r} must use torch.strided layout, got {tensor.layout}")
        if tensor.is_quantized:
            raise ValueError(f"quantized tensor {name!r} cannot be represented as a plain typed slab view")

        shape = tuple(tensor.shape)
        entry_alignment = math.lcm(alignment, tensor.element_size())
        cursor = _align_up(cursor, entry_alignment)
        nbytes = tensor.numel() * tensor.element_size()
        entries[name] = SlabEntry(offset=cursor, nbytes=nbytes, shape=shape, dtype=tensor.dtype)
        cursor += nbytes

    # A one-byte allocation keeps an empty layout representable by torch.empty.
    total_nbytes = max(_align_up(cursor, alignment), 1)
    return BlockSlabLayout(entries=entries, nbytes=total_nbytes, alignment=alignment)


def _validate_layout(layout: BlockSlabLayout) -> None:
    if not isinstance(layout, BlockSlabLayout):
        raise TypeError(f"layout must be a BlockSlabLayout, got {type(layout).__name__}")
    _validate_alignment(layout.alignment)
    if layout.nbytes < 1:
        raise ValueError(f"layout.nbytes must be at least 1, got {layout.nbytes}")

    previous_end = 0
    for name, entry in layout.entries.items():
        if not isinstance(name, str):
            raise TypeError(f"layout names must be strings, got {type(name).__name__}")
        if entry.offset < previous_end:
            raise ValueError(f"layout entry {name!r} overlaps a preceding entry")
        if entry.offset % layout.alignment:
            raise ValueError(f"layout entry {name!r} offset {entry.offset} is not aligned to {layout.alignment} bytes")
        if entry.offset % entry.dtype.itemsize:
            raise ValueError(f"layout entry {name!r} offset {entry.offset} is not aligned for {entry.dtype}")

        expected_nbytes = _tensor_nbytes(entry.shape, entry.dtype)
        if entry.nbytes != expected_nbytes:
            raise ValueError(f"layout entry {name!r} has {entry.nbytes} bytes, expected {expected_nbytes}")
        if entry.offset + entry.nbytes > layout.nbytes:
            raise ValueError(f"layout entry {name!r} exceeds the {layout.nbytes}-byte slab")
        previous_end = entry.offset + entry.nbytes


def _validate_raw_buffer(raw: torch.Tensor, *, minimum_nbytes: int, label: str) -> None:
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor, got {type(raw).__name__}")
    if raw.dtype != torch.uint8 or raw.dim() != 1 or not raw.is_contiguous():
        raise ValueError(f"{label} must be a contiguous 1-D uint8 tensor")
    if raw.numel() < minimum_nbytes:
        raise ValueError(f"{label} is too small: need {minimum_nbytes} bytes, got {raw.numel()}")


def carve_block_slab(raw: torch.Tensor, layout: BlockSlabLayout) -> dict[str, torch.Tensor]:
    """Carve zero-copy typed views from the leading bytes of ``raw``."""

    _validate_layout(layout)
    _validate_raw_buffer(raw, minimum_nbytes=layout.nbytes, label="raw slab")

    views: dict[str, torch.Tensor] = {}
    for name, entry in layout.entries.items():
        byte_view = raw.narrow(0, entry.offset, entry.nbytes)
        views[name] = byte_view.view(entry.dtype).view(entry.shape)
    return views


def _allocate_pinned_uint8(nbytes: int) -> torch.Tensor:
    return torch.empty((nbytes,), dtype=torch.uint8, device="cpu", pin_memory=True)


def _allocate_cpu_uint8(nbytes: int, *, pin_memory: bool, strict_pin: bool) -> torch.Tensor:
    if not pin_memory:
        return torch.empty((nbytes,), dtype=torch.uint8, device="cpu")

    try:
        raw = _allocate_pinned_uint8(nbytes)
        if not raw.is_pinned():
            raise RuntimeError("the allocator returned non-pinned memory")
        return raw
    except Exception as exc:
        message = f"failed to allocate a {nbytes}-byte pinned block slab: {exc}"
        if strict_pin:
            raise RuntimeError(message) from exc
        warnings.warn(f"{message}; falling back to regular CPU memory", RuntimeWarning, stacklevel=3)
        return torch.empty((nbytes,), dtype=torch.uint8, device="cpu")


def _validate_tensors_against_layout(tensors: Mapping[str, torch.Tensor], layout: BlockSlabLayout) -> None:
    tensor_names = set(tensors)
    layout_names = set(layout.entries)
    if tensor_names != layout_names:
        missing = sorted(layout_names - tensor_names)
        unexpected = sorted(tensor_names - layout_names)
        raise ValueError(f"tensor names do not match layout (missing={missing}, unexpected={unexpected})")

    for name, entry in layout.entries.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"tensor {name!r} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.device.type != "cpu":
            raise ValueError(f"tensor {name!r} must be on CPU before packing, got {tensor.device}")
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError(f"tensor {name!r} cannot be packed from layout {tensor.layout}")
        if tuple(tensor.shape) != entry.shape:
            raise ValueError(f"tensor {name!r} shape {tuple(tensor.shape)} does not match layout shape {entry.shape}")
        if tensor.dtype != entry.dtype:
            raise ValueError(f"tensor {name!r} dtype {tensor.dtype} does not match layout dtype {entry.dtype}")


def pack_cpu_block_slab(
    tensors: Mapping[str, torch.Tensor],
    *,
    layout: BlockSlabLayout | None = None,
    alignment: int = DEFAULT_SLAB_ALIGNMENT,
    pin_memory: bool = True,
    strict_pin: bool = False,
) -> BlockSlab:
    """Pack CPU tensors into one contiguous raw slab.

    Non-contiguous inputs are copied in logical tensor order into contiguous
    typed views.  If pinned allocation fails, ``strict_pin=True`` raises;
    otherwise a warning is emitted and regular CPU memory is used.
    """

    if layout is None:
        layout = build_block_slab_layout(tensors, alignment=alignment)
    else:
        _validate_layout(layout)
        if alignment != DEFAULT_SLAB_ALIGNMENT and alignment != layout.alignment:
            raise ValueError(f"alignment {alignment} does not match supplied layout alignment {layout.alignment}")

    _validate_tensors_against_layout(tensors, layout)
    raw = _allocate_cpu_uint8(layout.nbytes, pin_memory=pin_memory, strict_pin=strict_pin)
    views = carve_block_slab(raw, layout)

    for name, destination in views.items():
        source = tensors[name].detach().contiguous()
        destination.copy_(source)

    return BlockSlab(raw=raw, views=views, layout=layout)


def allocate_block_slab_slot(
    layout: BlockSlabLayout,
    device: torch.device | str,
    *,
    slot_nbytes: int | None = None,
) -> BlockSlab:
    """Allocate a raw slot on ``device`` and carve ``layout`` from its front.

    ``slot_nbytes`` may exceed the layout size, allowing a largest-block pool
    slot to serve smaller heterogeneous blocks.
    """

    _validate_layout(layout)
    allocation_nbytes = layout.nbytes if slot_nbytes is None else slot_nbytes
    if isinstance(allocation_nbytes, bool) or not isinstance(allocation_nbytes, int):
        raise TypeError(f"slot_nbytes must be an integer, got {type(allocation_nbytes).__name__}")
    if allocation_nbytes < layout.nbytes:
        raise ValueError(f"slot_nbytes is too small: need {layout.nbytes}, got {allocation_nbytes}")

    raw = torch.empty((allocation_nbytes,), dtype=torch.uint8, device=device)
    return BlockSlab(raw=raw, views=carve_block_slab(raw, layout), layout=layout)


def copy_block_slab_(
    destination_raw: torch.Tensor,
    source_raw: torch.Tensor,
    *,
    nbytes: int | None = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    """Copy a slab with exactly one raw ``copy_`` operation.

    The operation is device-agnostic.  To make an H2D copy asynchronous, the
    source must be pinned and the caller must invoke this function in the
    appropriate accelerator copy-stream context.
    """

    copy_nbytes = source_raw.numel() if nbytes is None else nbytes
    if isinstance(copy_nbytes, bool) or not isinstance(copy_nbytes, int) or copy_nbytes < 0:
        raise ValueError(f"nbytes must be a non-negative integer, got {copy_nbytes!r}")

    _validate_raw_buffer(source_raw, minimum_nbytes=copy_nbytes, label="source slab")
    _validate_raw_buffer(destination_raw, minimum_nbytes=copy_nbytes, label="destination slab")
    destination_raw.narrow(0, 0, copy_nbytes).copy_(source_raw.narrow(0, 0, copy_nbytes), non_blocking=non_blocking)
    return destination_raw
