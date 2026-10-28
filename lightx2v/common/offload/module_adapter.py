"""Explicit CPU-source/GPU-slot adapters for immutable native nn.Modules.

No hooks or global patches are installed. Callers own stream ordering and the
shared arena lifetime; restore/release only after outstanding device work ends.
"""

from __future__ import annotations

import copy
import weakref
from itertools import chain

import torch
from loguru import logger
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


class PinnedModuleCPUWeights:
    """Pinned immutable parameters for an eval CPU module, without hooks.

    ``activate`` uses the module's normal device transfer. ``restore`` waits for
    that CUDA device, restores the original CPU Parameter objects and copies
    current buffers back to CPU, including buffers registered during forward.
    Call restore in a finally block, including when activation fails. Parameter
    storage aliases/layout are retained on CPU; device layout follows Module.to.
    """

    def __init__(self, module):
        if module.training:
            raise ValueError("PinnedModuleCPUWeights requires an evaluation-mode module")
        self._module = weakref.ref(module)
        self._device = None
        self._slots = []
        self._parameters = {}
        self.sources = {}
        self.parameter_bytes = self.pinned_parameter_bytes = 0
        groups = {}
        for owner in module.modules():
            for name, parameter in owner._parameters.items():
                if parameter is None:
                    continue
                if parameter.device.type != "cpu" or parameter.layout != torch.strided or parameter.is_quantized:
                    raise ValueError("PinnedModuleCPUWeights requires materialized, unquantized strided CPU parameters")
                self._slots.append((weakref.ref(owner), name, parameter))
                key = id(parameter)
                if key not in self._parameters:
                    self._parameters[key] = parameter
                    groups.setdefault(parameter.untyped_storage()._cdata, []).append(parameter)

        with torch.no_grad():
            for parameters in groups.values():
                storage = parameters[0].untyped_storage()
                # Pin each storage once, preserving aliases, strides and offsets.
                raw = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (storage.nbytes(),), (1,))
                try:
                    pinned = raw if raw.is_pinned() else raw.pin_memory()
                except Exception as error:
                    logger.warning(f"Failed to allocate pinned module weights: {error}. Falling back to regular CPU memory.")
                    pinned = raw
                self.parameter_bytes += storage.nbytes()
                if pinned.is_pinned():
                    self.pinned_parameter_bytes += storage.nbytes()
                for parameter in parameters:
                    source = torch.empty(0, dtype=parameter.dtype, device="cpu").set_(pinned.untyped_storage(), parameter.storage_offset(), parameter.shape, parameter.stride())
                    self.sources[id(parameter)] = source
                    parameter.data = source
                # Do not retain the old pageable checkpoint storage after pinning.
                del storage, raw, pinned, source
        logger.info("Module CPU weights: {:.2f}/{:.2f} GiB pinned (rank-local)", self.pinned_parameter_bytes / 2**30, self.parameter_bytes / 2**30)

    @property
    def module(self):
        module = self._module()
        if module is None:
            raise RuntimeError("The native module has already been released")
        return module

    def activate(self, device):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("PinnedModuleCPUWeights only activates on CUDA")
        if self._device is not None:
            raise RuntimeError("Restore the CPU weights before activating again")
        # Mark first: a failed Module.to may have already moved some parameters.
        self._device = device
        return self.module.to(device=device, non_blocking=all(source.is_pinned() for source in self.sources.values()))

    def restore(self):
        if self._device is not None:
            torch.cuda.synchronize(self._device)
        with torch.no_grad():
            for key, parameter in self._parameters.items():
                parameter.data = self.sources[key]
            for owner, name, parameter in self._slots:
                owner()._parameters[name] = parameter
            # Forward may replace/add buffers; never restore a stale snapshot.
            buffers = {}
            for owner in self.module.modules():
                for name, buffer in owner._buffers.items():
                    if buffer is not None:
                        key = id(buffer)
                        if key not in buffers:
                            buffers[key] = buffer.to(device="cpu")
                        owner._buffers[name] = buffers[key]
        self._device = None


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
