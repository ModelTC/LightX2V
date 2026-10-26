"""Compile dispatch, sequence-only dynamic shapes and native RelayVSR norms."""

from functools import partial

import torch
import torch.nn.functional as F


class RelayCompileMixin:
    def get_compiled_block(self, block_idx, block):
        key = self.get_compile_block_key(block_idx, block)
        cached = self.compiled_blocks.get(key)
        if cached is not None and cached[0] is block:
            return cached[1]

        # Compile the family method before binding the block so Wan and Flash
        # do not share a wrapper's Dynamo cache and recompile limit.
        if self.compile_backend == "default":
            options = {"options": self.compile_options} if self.compile_options else {}
            compiled = torch.compile(self.infer_block, dynamic=None, **options)
        else:
            compiled = torch.compile(self.infer_block, dynamic=None, backend=self._compile_backend_obj)
        bound = partial(compiled, block)
        self.compiled_blocks[key] = (block, bound)
        return bound


def mark_sequence_dynamic(*tensors):
    for tensor in tensors:
        torch._dynamo.mark_dynamic(tensor, 0)
        torch._dynamo.mark_static(tensor, list(range(1, tensor.ndim)))


# Preserve the checkpoint's native reductions and low-precision rounding while
# Inductor fuses the surrounding projections, modulation and residuals.
@torch.library.custom_op("lightx2v::relayvsr_rms_norm", mutates_args=(), device_types="cuda")
def native_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


@native_rms_norm.register_fake
def _rms_norm_fake(x, weight, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("lightx2v::relayvsr_layer_norm", mutates_args=(), device_types="cuda")
def native_layer_norm(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), None if weight is None else weight.float(), None if bias is None else bias.float(), eps)


@native_layer_norm.register_fake
def _layer_norm_fake(x, weight, bias, eps):
    return torch.empty_like(x, dtype=torch.float32, memory_format=torch.contiguous_format)


# Keep MUSA SDPA's static meta constraints outside the dynamic Wan block.
@torch.library.custom_op("lightx2v::relayvsr_sdpa", mutates_args=(), device_types="cuda")
def native_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    from lightx2v.common.ops.attn.torch_sdpa import TorchSDPAWeight

    return TorchSDPAWeight().apply(query, key, value)


@native_sdpa.register_fake
def _sdpa_fake(query, key, value):
    return query.new_empty((query.shape[0], query.shape[1] * query.shape[2]))
