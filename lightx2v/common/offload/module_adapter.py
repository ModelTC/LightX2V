"""Explicit CPU-source/GPU-slot adapters for immutable native nn.Modules.

No hooks or global patches are installed. Callers own stream ordering and the
shared arena lifetime; restore/release only after outstanding device work ends.
"""

from __future__ import annotations

import copy
import weakref
from itertools import chain

import torch
from torch import nn


def assign_module_tensor(module, name, tensor):
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = tensor
    else:
        raise KeyError(f"Not a parameter or buffer: {name}")


def module_tensors(module):
    # Include runtime buffers and tied names so no alias keeps a stale device copy.
    return dict(chain(module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)))


class ModuleCPUWeights:
    """Retain CPU sources while selected native submodules execute on a GPU."""

    def __init__(self, module):
        self._module = weakref.ref(module)
        self.sources = {name: tensor.detach() for name, tensor in module_tensors(module).items()}
        if any(t.device.type != "cpu" for t in self.sources.values()):
            raise ValueError("ModuleCPUWeights requires materialized CPU tensors")

    @property
    def module(self):
        module = self._module()
        if module is None:
            raise RuntimeError("The native module has already been released")
        return module

    def activate(self, prefixes, device):
        module = self.module
        child_prefixes = tuple(prefix + "." for prefix in prefixes)
        for name, source in self.sources.items():
            if name in prefixes or name.startswith(child_prefixes):
                assign_module_tensor(module, name, source.to(device, non_blocking=source.is_pinned()))

    def restore(self):
        module = self.module
        for name, source in self.sources.items():
            assign_module_tensor(module, name, source)


class NativeModuleBlockSlot:
    """Adapt a native block to EventSlotWeightAsyncStreamManager's interface."""

    def __init__(self, source, device):
        # Deep-copy a meta skeleton, never a full CPU checkpoint block.
        tensors = module_tensors(source)
        memo = {}
        for tensor in tensors.values():
            meta = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="meta")
            memo[id(tensor)] = nn.Parameter(meta, requires_grad=False) if isinstance(tensor, nn.Parameter) else meta
        self.module = copy.deepcopy(source, memo)
        for name, tensor in tensors.items():
            target = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
            assign_module_tensor(self.module, name, target)
        self.module.eval().requires_grad_(False)
        self.targets = module_tensors(self.module)
        self.nbytes = sum(t.numel() * t.element_size() for t in self.targets.values())

    @torch.no_grad()
    def load_state_dict(self, source, block_index, adapter_block_index=None):
        del block_index, adapter_block_index
        if source.keys() != self.targets.keys():
            raise ValueError("Native block slot parameter/buffer keys differ from source")
        for name, target in self.targets.items():
            value = source[name]
            if value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError(f"Native block slot shape/dtype mismatch: {name}")
            target.copy_(value, non_blocking=True)


class NativeModuleBlockSource:
    """Expose CPU parameters and all buffers without copying their storage."""

    def __init__(self, module):
        self.tensors = {name: t.detach() for name, t in module_tensors(module).items()}

    def state_dict(self):
        return self.tensors
