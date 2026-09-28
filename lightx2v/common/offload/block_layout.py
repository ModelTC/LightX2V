"""Persistent, rank-local block storage populated during weight loading."""

from dataclasses import dataclass, field
from math import prod

import torch

ALIGNMENT_BYTES = 256


def align_up(nbytes):
    return (nbytes + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES


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
            shape = tuple(torch.Size(shape))
            if key in keys or name in names or any(dim <= 0 for dim in shape):
                raise ValueError(f"Invalid or duplicate block tensor: {name}")
            if not isinstance(dtype, torch.dtype):
                raise TypeError(f"Expected a torch.dtype for block tensor: {name}")
            if transpose and len(shape) != 2:
                raise ValueError(f"Expected a matrix for transposed weight: {name}")
            offset = align_up(offset)
            spec = TensorSpec(key, name, shape, dtype, transpose, offset)
            specs.append(spec)
            keys.add(key)
            names.add(name)
            offset += spec.nbytes
        if not specs:
            raise ValueError("A contiguous block must contain weights")
        return cls(tuple(specs), offset)


class BlockBuffer:
    """A block-sized view of storage; tensor views keep its allocation alive."""

    def __init__(self, layout, storage):
        if storage.dtype != torch.uint8 or storage.ndim != 1 or not storage.is_contiguous():
            raise ValueError("Block storage must be a contiguous one-dimensional uint8 tensor")
        if layout.nbytes <= 0 or storage.numel() < layout.nbytes:
            raise ValueError(f"Block requires {layout.nbytes} bytes; storage provides {storage.numel()} bytes")
        if storage.data_ptr() % ALIGNMENT_BYTES:
            raise ValueError(f"Block storage address must be aligned to {ALIGNMENT_BYTES} bytes")
        self.layout = layout
        # Bound whole-block copies even when the supplied storage is larger.
        self.storage = storage.narrow(0, 0, layout.nbytes)

    @classmethod
    def allocate(cls, layout, device, backend=None):
        """Allocate independent storage, pinned on CPU, with initialized padding."""
        device = torch.device(device)
        from lightx2v_platform.base.offload import TorchBlockOffload

        allocator = backend or TorchBlockOffload
        storage = allocator.allocate(layout.nbytes, device)
        if storage.data_ptr() % ALIGNMENT_BYTES:
            # Only overallocate when the backend did not align the allocation.
            storage = allocator.allocate(layout.nbytes + ALIGNMENT_BYTES - 1, device)
            offset = (-storage.data_ptr()) % ALIGNMENT_BYTES
            storage = storage.narrow(0, offset, layout.nbytes)
        buffer = cls(layout, storage)
        # Padding is initialized once; the actual weights are filled directly.
        if device.type == "cpu":
            end = 0
            for spec in layout.tensors:
                buffer.storage[end : spec.offset].zero_()
                end = spec.offset + spec.nbytes
        return buffer

    def view(self, spec, operator=False):
        """Interpret a planned byte range, optionally using the operator's transpose."""
        if spec.offset < 0 or spec.offset % ALIGNMENT_BYTES:
            raise ValueError(f"Invalid aligned offset for block tensor: {spec.name}")
        if spec.nbytes <= 0 or spec.offset + spec.nbytes > self.storage.numel():
            raise ValueError(f"Block tensor exceeds its storage: {spec.name}")
        tensor = self.storage.narrow(0, spec.offset, spec.nbytes).view(spec.dtype).view(spec.shape)
        return tensor.t() if operator and spec.transpose else tensor


class BlockLoadContext:
    """Bind preallocated views during one block load.

    CPU loads consume checkpoint tensors; device loads only bind views.
    Do not retain this context after loading: it references the checkpoint.
    """

    def __init__(self, sources, buffer):
        self.sources = sources
        self.views = {spec.name: buffer.view(spec) for spec in buffer.layout.tensors}

    def take(self, name, transpose=False):
        view = self.views.pop(name)
        if view.device.type == "cpu":
            view.copy_(self.sources[name])
            self.sources.pop(name)
        return view.t() if transpose else view

    def finish(self):
        if self.views:
            raise ValueError("Operators did not load the complete block layout")

    def bind(self, operator):
        """Load platform operators directly into their planned block views."""
        for name, attr, transpose in operator.base_attrs:
            tensor = self.take(name, transpose)
            buffer_attr = f"{attr}_cuda_buffer" if operator.create_cuda_buffer else f"pin_{attr}"
            setattr(operator, buffer_attr, tensor)


class ContiguousBlockTransfer:
    """Copy immutable CPU blocks to two persistent device slots, without packing."""

    def __init__(self, blocks, slots, backend=None):
        from lightx2v_platform.base.offload import get_block_offload_backend

        if not blocks or len(slots) != 2:
            raise ValueError("Contiguous transfer requires CPU blocks and two GPU slots")
        sources = tuple(block.block_buffer for block in blocks)
        device = slots[0].block_buffer.storage.device
        layout = sources[0].layout
        for source in sources:
            if source.layout != layout or source.storage.device.type != "cpu" or not source.storage.is_pinned():
                raise ValueError("Contiguous CPU block layouts or pin states differ")
        for slot in slots:
            target = slot.block_buffer
            if target.layout != layout or target.storage.device != device:
                raise ValueError("Contiguous GPU slot layout differs from CPU blocks")

        reference = blocks[0].block_auxiliary
        auxiliary = {}
        for block in (*blocks, *slots):
            if block.block_auxiliary.keys() != reference.keys():
                raise ValueError("Contiguous block auxiliary weights differ")
            for key, expected in reference.items():
                tensor = block.block_auxiliary[key]
                if tensor.device != device or tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                    raise ValueError(f"Unsupported auxiliary weight: {key}")
            auxiliary[id(block)] = tuple(block.block_auxiliary[key] for key in reference)
        self.sources = tuple((block.block_buffer, auxiliary[id(block)]) for block in blocks)
        self.targets = {id(slot): (slot.block_buffer, auxiliary[id(slot)]) for slot in slots}
        self.device = device
        self.backend = backend or get_block_offload_backend()
        self.device_module = self.backend.device_module(device)
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
        source, source_auxiliary = self.sources[block_idx]
        destination, target_auxiliary = self.targets[id(target)]
        stream = self.device_module.current_stream(self.device)
        if stream not in self.streams:
            stream.wait_event(self.ready)
            self.streams.add(stream)
        self.backend.copy(destination.storage, source.storage, stream)
        for dst, src in zip(target_auxiliary, source_auxiliary):
            self.backend.copy(dst, src, stream)

    def close(self):
        for stream in self.streams:
            stream.synchronize()
        self.closed = True
