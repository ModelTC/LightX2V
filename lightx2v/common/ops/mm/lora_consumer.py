"""Keep base/LoRA projections separate until their activation or residual consumer."""

from copy import copy
from typing import NamedTuple

import torch
import torch.nn.functional as F


class ProjectionParts(NamedTuple):
    base: torch.Tensor
    delta: torch.Tensor | None = None

    def materialize(self):
        return self.base if self.delta is None else (self.base + self.delta).to(self.base.dtype)


def base_projection(module):
    # Share the current weight storage, not a cached pointer or a merged weight.
    # The original module remains untouched during offload and adapter switching.
    base_module = copy(module)
    base_module.has_lora_branch = False
    return base_module


def project_parts(module, hidden, base_module):
    if not module.has_lora_branch:
        return ProjectionParts(module.apply(hidden))
    return ProjectionParts(base_module.apply(hidden), module.apply_lora(hidden))


def swiglu_consumer(parts):
    if parts.delta is None:
        value, gate = parts.base.chunk(2, dim=-1)
        return value * F.silu(gate)
    from .lora_consumer_kernels import swiglu

    return swiglu(parts.base, parts.delta)


def gated_residual_consumer(parts, residual, gate, indices):
    if parts.delta is None:
        return residual + gate.index_select(0, indices) * parts.base
    from .lora_consumer_kernels import gated_residual

    return gated_residual(parts.base, parts.delta, residual, gate, indices)
