"""Shared storage for checkpoint-backed Q/K/V projections."""

import torch


class FusedQKVStorage:
    def __init__(self, sources, target):
        self.sources = sources
        self.target = target
        module = sources[0]
        # Default MM stores [K, N] but does not define weight_need_transpose.
        transpose = getattr(module, "weight_need_transpose", None)
        if transpose is None:
            transpose = next((transposed for _, attr, transposed in getattr(module, "base_attrs", ()) if attr == "weight"), False)
        self.weight_dim = 1 if transpose else 0

    @staticmethod
    def _scale_dim(tensor):
        return 0 if tensor.ndim == 1 or tensor.shape[-1] == 1 else tensor.ndim - 1

    @staticmethod
    def _join(tensors, dim):
        # Use the native Linear [N, K] layout so each original projection is
        # still contiguous (or a transpose of contiguous) inside the allocation.
        transpose = tensors[0].ndim == 2 and dim == 1
        native = [tensor.t() if transpose else tensor for tensor in tensors]
        first = native[0]
        offset = first.storage_offset()
        shared = True
        for tensor in native:
            shared &= tensor.is_contiguous() and tensor.untyped_storage().data_ptr() == first.untyped_storage().data_ptr() and tensor.storage_offset() == offset
            offset += tensor.numel()
        shape = (sum(tensor.shape[0] for tensor in native), *first.shape[1:])
        if shared:
            # Also recognizes slices of an existing offload CUDA buffer.
            joined = first.as_strided(shape, first.stride())
        else:
            joined = torch.empty(shape, dtype=first.dtype, device=first.device, pin_memory=first.device.type == "cpu" and first.is_pinned())
            torch.cat(native, dim=0, out=joined)
        return joined.t() if transpose else joined

    def _bind(self, attr, tensor, dim):
        setattr(self.target, attr, tensor)
        if tensor is None:
            for module in self.sources:
                setattr(module, attr, None)
            return
        for module, view in zip(self.sources, tensor.chunk(3, dim=dim)):
            setattr(module, attr, view)

    def refresh(self):
        # Never fold adapter diffs into base storage: the unfused fallback
        # applies them itself, and removing an adapter must restore the base.
        for suffix in ("weight", "weight_scale"):
            for attr in (f"pin_{suffix}", f"{suffix}_cuda_buffer", suffix):
                tensors = [getattr(module, attr, None) for module in self.sources]
                if all(tensor is not None for tensor in tensors):
                    dim = self.weight_dim if suffix == "weight" else self._scale_dim(tensors[0])
                    self._bind(attr, self._join(tensors, dim), dim)
                else:
                    setattr(self.target, attr, None)
        self.target.bias = None
        self.target.has_lora_branch = False

    @property
    def can_move(self):
        return getattr(self.target, "weight", None) is not None or getattr(self.target, "pin_weight", None) is not None

    def move(self, device, non_blocking=False):
        for attr in ("weight", "weight_scale"):
            tensor = getattr(self.target, attr, None)
            pinned = getattr(self.target, f"pin_{attr}", None)
            if tensor is None and pinned is None:
                continue
            if device == "cpu" and pinned is not None:
                if tensor is not None and tensor.data_ptr() != pinned.data_ptr():
                    pinned.copy_(tensor, non_blocking=non_blocking)
                moved = pinned
            else:
                source = pinned if pinned is not None else tensor
                moved = source.to(device, non_blocking=non_blocking)
            dim = self.weight_dim if attr == "weight" else self._scale_dim(moved)
            self._bind(attr, moved, dim)
        for module in self.sources:
            for attr in getattr(module, "lora_attrs", {}):
                tensor = getattr(module, attr, None)
                if isinstance(tensor, torch.Tensor):
                    setattr(module, attr, tensor.to(device, non_blocking=non_blocking))
