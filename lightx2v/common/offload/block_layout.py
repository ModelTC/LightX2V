"""Persistent, rank-local block storage populated during weight loading."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod

import torch


@dataclass(frozen=True)
class TensorSpec:
    key: tuple[str, str]
    name: str = field(compare=False)
    shape: tuple[int, ...]
    dtype: torch.dtype
    transpose: bool
    offset: int

    @property
    def nbytes(self):
        return prod(self.shape) * self.dtype.itemsize

    def view(self, storage, operator=False):
        tensor = storage.narrow(0, self.offset, self.nbytes).view(self.dtype).view(self.shape)
        return tensor.t() if operator and self.transpose else tensor


@dataclass(frozen=True)
class BlockLayout:
    tensors: tuple[TensorSpec, ...]
    nbytes: int

    @classmethod
    def build(cls, entries):
        """Entries describe loading shapes and the operators' final dtypes."""
        specs, names, keys = [], set(), set()
        offset = 0
        for key, name, shape, dtype, transpose in sorted(entries, key=lambda entry: entry[0]):
            if key in keys or name in names or prod(shape) <= 0:
                raise ValueError(f"Invalid or duplicate block tensor: {name}")
            if transpose and len(shape) != 2:
                raise ValueError(f"Expected a matrix for transposed weight: {name}")
            offset = (offset + 255) // 256 * 256
            spec = TensorSpec(key, name, tuple(shape), dtype, transpose, offset)
            specs.append(spec)
            keys.add(key)
            names.add(name)
            offset += spec.nbytes
        if not specs:
            raise ValueError("A contiguous block must contain weights")
        return cls(tuple(specs), offset)


class BlockBuffer:
    def __init__(self, layout, device):
        self.layout = layout
        device = torch.device(device)
        self.storage = torch.empty(layout.nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")
        # Padding is initialized once; the actual weights are filled directly.
        if device.type == "cpu":
            end = 0
            for spec in layout.tensors:
                self.storage[end : spec.offset].zero_()
                end = spec.offset + spec.nbytes
        self.views = {spec.name: spec.view(self.storage) for spec in layout.tensors}


class BlockLoadContext(Mapping):
    """Bind preallocated views during one block load.

    CPU loads consume checkpoint tensors; device loads only bind views.
    Do not retain this context after loading: it references the checkpoint.
    """

    def __init__(self, sources, buffer):
        self.sources = sources
        self.buffer = buffer
        self.consumed = set()

    def __getitem__(self, key):
        return self.sources[key]

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def take(self, name, transpose=False):
        view = self.buffer.views[name]
        if view.device.type == "cpu":
            view.copy_(self.sources[name])
            self.sources.pop(name)
        self.consumed.add(name)
        return view.t() if transpose else view

    def finish(self):
        if self.consumed != self.buffer.views.keys():
            raise ValueError("Operators did not load the complete block layout")

    def bind(self, operator):
        """Load platform operators directly into their planned block views."""
        for name, attr, transpose in operator.base_attrs:
            tensor = self.take(name, transpose)
            buffer_attr = f"{attr}_cuda_buffer" if operator.create_cuda_buffer else f"pin_{attr}"
            setattr(operator, buffer_attr, tensor)


class ContiguousBlockTransfer:
    """Copy immutable CPU blocks to two persistent device slots, without packing."""

    def __init__(self, blocks, slots):
        if not blocks or len(slots) != 2:
            raise ValueError("Contiguous transfer requires CPU blocks and two GPU slots")
        sources = tuple(block.block_buffer for block in blocks)
        device = slots[0].block_buffer.storage.device
        layout = sources[0].layout
        for source in sources:
            if source.layout != layout or source.storage.device.type != "cpu" or not source.storage.is_pinned():
                raise ValueError("Contiguous CPU block layouts or pin states differ")
        targets = {}
        for slot in slots:
            target = slot.block_buffer
            if target.layout != layout or target.storage.device != device:
                raise ValueError("Contiguous GPU slot layout differs from CPU blocks")
            auxiliary = []
            for block in blocks:
                if slot.block_auxiliary.keys() != block.block_auxiliary.keys():
                    raise ValueError("Contiguous block auxiliary weights differ")
                pairs = []
                for key, destination in slot.block_auxiliary.items():
                    source = block.block_auxiliary[key]
                    if source.device != device or destination.device != device or source.shape != destination.shape or source.dtype != destination.dtype:
                        raise ValueError(f"Unsupported auxiliary weight: {key}")
                    pairs.append((destination, source))
                auxiliary.append(tuple(pairs))
            targets[id(slot)] = (target, tuple(auxiliary))
        self.sources = sources
        self.targets = targets
        self.device = device
        self.device_module = getattr(torch, device.type)
        self.streams = set()
        self.closed = False
        self.ready = self.device_module.Event()
        self.ready.record(self.device_module.current_stream(device))

    @torch.no_grad()
    def copy(self, block_idx, target):
        if self.closed:
            raise RuntimeError("Contiguous transfer has been closed")
        if not 0 <= block_idx < len(self.sources):
            raise IndexError(block_idx)
        destination, auxiliary = self.targets[id(target)]
        stream = self.device_module.current_stream(self.device)
        if stream not in self.streams:
            stream.wait_event(self.ready)
            self.streams.add(stream)
        destination.storage.copy_(self.sources[block_idx].storage, non_blocking=True)
        destination.storage.record_stream(stream)
        for dst, src in auxiliary[block_idx]:
            dst.copy_(src, non_blocking=True)
            dst.record_stream(stream)
            src.record_stream(stream)

    def close(self):
        for stream in self.streams:
            stream.synchronize()
        self.closed = True
