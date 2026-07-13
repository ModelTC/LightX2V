from __future__ import annotations

import torch

from lightx2v.utils.registry_factory import ROPE_REGISTER

from .chunked_rope import ChunkedRope
from .template import RopeTemplate

ROPE_TYPE_ALIASES = {
    "torch_complex": "torch_complex_rope",
    "torch_naive": "torch_real_rope",
    "torch_real": "torch_real_rope",
    "flashinfer": "flashinfer_rope",
}


def get_rope_module(rope_type: str, *, layout="interleaved", compute_dtype=torch.float32):
    rope_type = ROPE_TYPE_ALIASES.get(rope_type, rope_type)
    if rope_type not in ROPE_REGISTER:
        raise ValueError(f"Unsupported rope_type: {rope_type}")
    rope_class = ROPE_REGISTER[rope_type]
    if rope_type == "torch_complex_rope":
        if layout != "interleaved":
            raise ValueError("TorchComplexRope only supports interleaved layout.")
        return rope_class(compute_dtype=compute_dtype)
    if issubclass(rope_class, RopeTemplate):
        return rope_class(layout=layout, compute_dtype=compute_dtype)
    return rope_class()


def build_rope_module(config, *, layout="interleaved", default="flashinfer_rope", compute_dtype=torch.float32):
    module = get_rope_module(config.get("rope_type", default), layout=layout, compute_dtype=compute_dtype)
    if hasattr(module, "set_config"):
        module.set_config(config)
    if config.get("rope_chunk", False):
        module = ChunkedRope(module, config.get("rope_chunk_size", 100))
        module.set_config(config)
    return module
