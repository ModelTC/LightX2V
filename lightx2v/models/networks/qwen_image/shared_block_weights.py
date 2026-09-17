"""Qwen-Image Diffusers BF16 checkpoints backed by shared CPU block weights."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.offload.checkpoint_metadata import checkpoint_content_digest, read_checkpoint_json, read_safetensors_header
from lightx2v.common.offload.shared_pinned_arena import SharedWeightManifest
from lightx2v.common.offload.shared_weight_coordinator import materialize_shared_weight_arena, validate_shared_weight_config
from lightx2v.common.offload.shared_weight_map import SharedWeightViewMap, validate_shared_operator_views
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

_PRIVATE_KEYS = {
    "img_in.weight",
    "img_in.bias",
    "txt_in.weight",
    "txt_in.bias",
    "txt_norm.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_1.bias",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "time_text_embed.timestep_embedder.linear_2.bias",
    "norm_out.linear.weight",
    "norm_out.linear.bias",
    "proj_out.weight",
    "proj_out.bias",
}


@dataclass(frozen=True)
class CheckpointTensor:
    name: str
    path: Path
    shape: tuple[int, ...]


class QwenBf16SharedBlockAdapter:
    FORMAT_VERSION = "qwen-image-bf16-shared-block-v1"

    def __init__(self, model_path, config, *, lora_path=None):
        self.config = config
        self.checkpoint_dir = Path(model_path).resolve()
        self._validate_config(lora_path)
        self.tensors, self.private_tensors, signature = self._inspect_checkpoint()
        metadata = {item.name: torch.empty(item.shape, dtype=torch.bfloat16, device="meta") for item in self.tensors}
        self.manifest = SharedWeightManifest.from_tensors(metadata, weight_signature=signature, alignment=4096)
        self.allocation = None

    def _validate_config(self, lora_path):
        validate_shared_weight_config(self.config)
        errors = []
        if not self.config.get("cpu_offload", False):
            errors.append("cpu_offload must be true")
        if self.config.get("offload_granularity") != "block":
            errors.append("offload_granularity must be 'block'")
        for key in ("lazy_load", "unload_modules", "dit_quantized", "weight_auto_quant", "tensor_parallel", "layered", "dummy_model"):
            if self.config.get(key, False):
                errors.append(f"{key} is not supported")
        if self.config.get("dit_quantized_ckpt") or self.config.get("dit_quant_scheme", "Default") != "Default":
            errors.append("only unquantized BF16 weights are supported")
        for key in ("dit_original_ckpt", "transformer_model_path"):
            if self.config.get(key):
                errors.append(f"{key} overrides are not supported; select the Diffusers directory with model_path")
        if self.config.get("lora_configs") or self.config.get("lora_dynamic_apply") or lora_path or self.config.get("adapter_model_path"):
            errors.append("LoRA/diff/adapter weights are not supported")
        if self.config.get("task") != "t2i":
            errors.append("the BF16 shared adapter currently supports task='t2i'")
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            errors.append("feature_caching must be 'NoCaching'")
        if GET_DTYPE() != torch.bfloat16 or GET_SENSITIVE_DTYPE() != torch.bfloat16:
            errors.append("DTYPE and SENSITIVE_LAYER_DTYPE must both resolve to BF16")
        layers = self.config.get("num_layers")
        if type(layers) is not int or layers <= 0:
            errors.append("num_layers must be a positive integer")
        if not self.checkpoint_dir.is_dir():
            errors.append(f"transformer directory does not exist: {self.checkpoint_dir}")
        if errors:
            raise ValueError("Invalid Qwen shared_cpu_weights configuration: " + "; ".join(errors))

    def _inspect_checkpoint(self):
        index = read_checkpoint_json(self.checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Qwen checkpoint index must contain a non-empty weight_map")
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
            raise ValueError("Qwen weight_map keys and shard filenames must be strings")
        by_file = defaultdict(set)
        for name, filename in weight_map.items():
            path = (self.checkpoint_dir / filename).resolve()
            if not path.is_relative_to(self.checkpoint_dir) or path.suffix != ".safetensors":
                raise ValueError(f"Invalid Qwen shard path: {filename!r}")
            by_file[path].add(name)

        tensors, private, signatures = [], [], []
        block_schemas = defaultdict(dict)
        for path, names in sorted(by_file.items()):
            header = read_safetensors_header(path)
            entries = {key: value for key, value in header.items() if key != "__metadata__"}
            if names != set(entries):
                raise ValueError(f"Qwen index/header keys differ in {path.name}: missing={sorted(names - entries.keys())}, unexpected={sorted(entries.keys() - names)}")
            for name in sorted(names):
                entry = entries[name]
                shape = entry.get("shape")
                if entry.get("dtype") != "BF16":
                    raise ValueError(f"Qwen shared weight {name!r} must be BF16")
                if not isinstance(shape, list) or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
                    raise ValueError(f"Invalid Qwen shape for {name!r}: {shape!r}")
                offsets = entry.get("data_offsets", [])
                if len(offsets) != 2 or any(type(v) is not int for v in offsets) or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * 2:
                    raise ValueError(f"Invalid Qwen data offsets for {name!r}")
                item = CheckpointTensor(name, path, tuple(shape))
                if name.startswith("transformer_blocks."):
                    parts = name.split(".", 2)
                    if len(parts) != 3 or not parts[1].isdigit() or str(int(parts[1])) != parts[1]:
                        raise ValueError(f"Invalid Qwen block key: {name!r}")
                    block_schemas[int(parts[1])][parts[2]] = item.shape
                    tensors.append(item)
                else:
                    private.append(item)
            signatures.append({"name": path.name, "size": path.stat().st_size, "sha256": checkpoint_content_digest(path)})

        expected = set(range(self.config["num_layers"]))
        if set(block_schemas) != expected:
            raise ValueError(f"Qwen block indices differ from num_layers: found={sorted(block_schemas)}, expected={sorted(expected)}")
        for block_idx in sorted(expected):
            if block_schemas[block_idx] != block_schemas[0]:
                raise ValueError(f"Qwen block {block_idx} key/shape schema differs from block 0")
        private_keys = {item.name for item in private}
        if private_keys != _PRIVATE_KEYS:
            raise ValueError(f"Qwen non-block keys differ: missing={sorted(_PRIVATE_KEYS - private_keys)}, unexpected={sorted(private_keys - _PRIVATE_KEYS)}")
        signature_data = {"format": self.FORMAT_VERSION, "files": signatures, "weight_map": weight_map, "dtype": "bfloat16", "num_layers": self.config["num_layers"]}
        signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return tuple(tensors), tuple(private), signature

    @staticmethod
    def _read_tensors(items):
        by_file = defaultdict(list)
        for item in items:
            by_file[item.path].append(item)
        for path, file_items in sorted(by_file.items()):
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for item in file_items:
                    tensor = checkpoint.get_tensor(item.name)
                    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != item.shape:
                        raise ValueError(f"Qwen tensor changed after preflight: {item.name}")
                    yield item.name, tensor

    def load_private_weights(self):
        # Clone only the small private tensors, releasing their shard mappings.
        return {name: tensor.clone() for name, tensor in self._read_tensors(self.private_tensors)}

    def _populate(self, views):
        logger.info(f"[SharedCPUWeightsInfo] Populating {len(self.tensors)} Qwen block tensors from {self.checkpoint_dir}")
        for name, tensor in self._read_tensors(self.tensors):
            views[name].copy_(tensor)

    def materialize(self):
        if self.allocation is not None:
            raise RuntimeError("Qwen shared block arena has already been materialized")
        self.allocation = materialize_shared_weight_arena(
            self.manifest,
            self._populate,
            scope=self.config.get("shared_cpu_weight_scope", "auto"),
            strict_numa=self.config.get("shared_cpu_weight_strict_numa", True),
            register_chunk_bytes=self.config.get("shared_cpu_weight_register_chunk_mb", 128) * 1024 * 1024,
        )
        return self.allocation

    def build_weight_map(self, private_weights):
        allocation = self.allocation if self.allocation is not None else self.materialize()
        return SharedWeightViewMap(private_weights, allocation.tensor_views(), owner=allocation)


def validate_qwen_shared_block_views(weight_map, blocks):
    """Check final operator views, including each MM's transposed layout."""
    state = blocks.state_dict()
    validate_shared_operator_views(weight_map, state)
    arena = weight_map.owner.arena
    expected = set(arena.manifest.by_name)
    checked = set()

    def visit(module):
        for name, _, transpose in getattr(module, "base_attrs", ()):
            if name not in expected:
                continue
            source = weight_map[name]
            wanted = source.t() if transpose else source
            actual = state[name]
            if actual.shape != wanted.shape or actual.stride() != wanted.stride():
                raise RuntimeError(f"Qwen shared tensor has the wrong operator layout: {name}")
            checked.add(name)
        for child in getattr(module, "_modules", {}).values():
            if child is not None:
                visit(child)

    visit(blocks)
    if checked != expected:
        raise RuntimeError(f"Qwen shared validation did not inspect every block tensor: missing={sorted(expected - checked)}")
    logger.info(f"[SharedCPUWeightsInfo] Validated {len(expected)} Qwen block tensor views in {arena.nbytes / 1024**3:.3f} GiB arena")
