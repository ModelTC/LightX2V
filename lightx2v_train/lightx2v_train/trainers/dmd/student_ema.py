"""EMA of student trainable weights, including FSDP2 parameter shards."""

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FSDPModule


class StudentWeightEMA:
    """Construct and update outside forward/backward; frozen weights stay shared."""

    def __init__(self, module, decay=0.99):
        if not 0 <= decay <= 1:
            raise ValueError("EMA decay must be in [0, 1].")
        self.module = module
        self.decay = decay
        self.num_updates = 0
        self._average_depth = 0
        self.shadow = {name: parameter.detach().float().clone() for name, parameter in self._parameters().items() if parameter.requires_grad}

    def _parameters(self):
        # Preview may leave full parameters registered when root reshard is off.
        for module in self.module.modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        return dict(self.module.named_parameters())

    @torch.no_grad()
    def update(self):
        if self._average_depth:
            raise RuntimeError("Cannot update EMA while averaged weights are installed.")
        parameters = self._parameters()
        for name, shadow in self.shadow.items():
            shadow.lerp_(parameters[name].detach().float(), 1 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self):
        parameters = self._parameters()
        for name, shadow in self.shadow.items():
            parameters[name].copy_(shadow)

    @contextmanager
    def average_parameters(self):
        with torch.no_grad():
            parameters = self._parameters()
            backup = {name: parameters[name].detach().clone() for name in self.shadow}
            self._average_depth += 1
            try:
                self.copy_to()
                yield self.module
            finally:
                parameters = self._parameters()
                for name, original in backup.items():
                    parameters[name].copy_(original)
                self._average_depth -= 1

    def state_dict(self):
        # Retain DTensor layout so DCP saves rank-local shards, not full copies.
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": dict(self.shadow),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["decay"] != self.decay:
            raise RuntimeError("Checkpoint EMA decay does not match the current recipe.")
        saved = state["shadow"]
        if saved.keys() != self.shadow.keys():
            raise RuntimeError("Checkpoint EMA parameter names do not match the student.")
        for name, shadow in self.shadow.items():
            value = saved[name]
            if value.shape != shadow.shape or value.dtype != torch.float32:
                raise RuntimeError(f"Checkpoint EMA shape or FP32 dtype mismatch for {name}.")
        if state["num_updates"] < 0:
            raise RuntimeError("Checkpoint EMA update count cannot be negative.")
        for name, shadow in self.shadow.items():
            shadow.copy_(saved[name])
        self.num_updates = state["num_updates"]
