"""
Mixin that gives any Runner tensor-parallel rank-0 I/O support:

  - rank predicates (_use_tp_rank0_io, _is_rank0, _rank_device)
  - broadcast any nested dict/list/tuple-of-tensors from rank 0 to all TP ranks
  - rank-0-only text-encoder load / unload helpers
"""

import gc

import torch
import torch.distributed as dist

from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

# ---------------------------------------------------------------------------
# Module-level helpers for the broadcast primitive.
# These are pure functions (no Runner state needed), so they live at module
# level rather than as instance methods.
# ---------------------------------------------------------------------------

_TENSOR_TAG = "__T__"
_OBJECT_TAG = "__O__"


def _pack(obj):
    """Recursively replace tensors with (shape, dtype) metadata stubs."""
    if torch.is_tensor(obj):
        return {_TENSOR_TAG: (tuple(obj.shape), obj.dtype)}
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        packed = [_pack(v) for v in obj]
        return packed if isinstance(obj, list) else tuple(packed)
    return {_OBJECT_TAG: obj}


def _alloc(meta, device):
    """Allocate empty tensors to mirror the structure returned by _pack."""
    if isinstance(meta, dict):
        if _TENSOR_TAG in meta:
            shape, dtype = meta[_TENSOR_TAG]
            return torch.empty(shape, dtype=dtype, device=device)
        if _OBJECT_TAG in meta:
            return meta[_OBJECT_TAG]
        return {k: _alloc(v, device) for k, v in meta.items()}
    if isinstance(meta, list):
        return [_alloc(v, device) for v in meta]
    if isinstance(meta, tuple):
        return tuple(_alloc(v, device) for v in meta)
    return meta


def _broadcast(skeleton, source, src_rank, rank0, device):
    """Fill skeleton with tensors broadcast from rank src_rank.

    skeleton: pre-allocated structure (from _alloc) on every non-rank0 rank;
              on rank0 it is the original source.
    source:   original payload on rank0, None on other ranks.
    """
    if torch.is_tensor(skeleton):
        if rank0:
            skeleton = source.to(device, non_blocking=True)
        dist.broadcast(skeleton, src=src_rank)
        return skeleton
    if isinstance(skeleton, dict):
        return {k: _broadcast(skeleton[k], source[k] if rank0 else None, src_rank, rank0, device) for k in skeleton}
    if isinstance(skeleton, list):
        return [_broadcast(skeleton[i], source[i] if rank0 else None, src_rank, rank0, device) for i in range(len(skeleton))]
    if isinstance(skeleton, tuple):
        return tuple(_broadcast(skeleton[i], source[i] if rank0 else None, src_rank, rank0, device) for i in range(len(skeleton)))
    return skeleton


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class TPRunnerMixin:
    """Rank-0 I/O helpers for tensor-parallel runners.

    Mix in before DefaultRunner / BaseRunner so that any runner subclass
    automatically gets TP support without touching its own load_model or
    run_input_encoder logic.
    """

    # --- predicates ---------------------------------------------------------

    def _use_tp_rank0_io(self):
        return self.config.get("tensor_parallel", False) and dist.is_initialized()

    def _is_rank0(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def _rank_device(self):
        if dist.is_initialized():
            return torch.device(f"{AI_DEVICE}:{dist.get_rank()}")
        return torch.device(AI_DEVICE)

    # --- broadcast ----------------------------------------------------------

    def _broadcast_rank0_payload(self, payload):
        """Broadcast a nested dict/list/tuple-of-tensors from rank 0 to all ranks.

        Uses broadcast_object_list for the metadata (fast, zero-copy for small
        dicts) and dist.broadcast for each tensor buffer (direct NCCL transfer).
        """
        if not self._use_tp_rank0_io():
            return payload

        device = self._rank_device()
        meta = [_pack(payload) if self._is_rank0() else None]
        dist.broadcast_object_list(meta, src=0)
        skeleton = payload if self._is_rank0() else _alloc(meta[0], device)
        result = _broadcast(skeleton, payload if self._is_rank0() else None, 0, self._is_rank0(), device)
        dist.barrier()
        return result

    # --- encoder lifecycle --------------------------------------------------

    def _load_rank0_text_encoder(self):
        if not self._is_rank0():
            return
        if self.text_encoders is None or self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.text_encoders = self.load_text_encoder()

    def _unload_rank0_text_encoder(self):
        if not self._is_rank0():
            return
        if self.text_encoders is not None and (self._use_tp_rank0_io() or self.config.get("lazy_load", False) or self.config.get("unload_modules", False)):
            del self.text_encoders
            self.text_encoders = None
            torch_device_module.empty_cache()
            gc.collect()
