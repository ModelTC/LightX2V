"""Wan adapter for NUMA-shared FP8 block-offload weights."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.common.offload.checkpoint_metadata import checkpoint_content_digest as _checkpoint_content_digest
from lightx2v.common.offload.checkpoint_metadata import read_safetensors_header as _read_header
from lightx2v.common.offload.shared_pinned_arena import SharedWeightManifest
from lightx2v.common.offload.shared_weight_coordinator import SharedArenaAllocation, materialize_shared_weight_arena, validate_shared_weight_config
from lightx2v.common.offload.shared_weight_map import SharedWeightViewMap
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

_BLOCK_FILE_RE = re.compile(r"^block_(\d+)\.safetensors$")
_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.")
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FN": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class _CheckpointTensor:
    name: str
    path: str
    target_dtype: torch.dtype
    shape: tuple[int, ...]
    cast_scale_through_infer_dtype: bool


def _target_dtype(name: str, source_dtype: torch.dtype, infer_dtype: torch.dtype) -> tuple[torch.dtype, bool]:
    # Match BaseTransformerModel._load_quant_ckpt followed by
    # MMWeightQuantTemplate.post_process.  The released fp8-vllm scales are
    # F32, but the legacy path first rounds them to inference dtype and then
    # stores the CPU staging tensor as F32.
    if source_dtype == torch.float8_e4m3fn:
        return source_dtype, False
    if name.endswith(".weight_scale"):
        return torch.float32, source_dtype != infer_dtype
    if source_dtype in (torch.float16, torch.bfloat16, torch.float32):
        return infer_dtype, source_dtype != infer_dtype
    return source_dtype, False


class WanFp8VllmSharedBlockAdapter:
    """Describe and materialize Wan DiT blocks without private rank copies."""

    FORMAT_VERSION = "wan-fp8-vllm-block-arena-v2"

    def __init__(self, config, *, lora_path=None):
        self.config = config
        self.infer_dtype = GET_DTYPE()
        self._validate_config(lora_path)
        self.checkpoint_dir = Path(config["dit_quantized_ckpt"]).resolve()
        self.block_files = self._discover_block_files()
        self.tensors, signature = self._inspect_checkpoint()
        metadata_tensors = {tensor.name: torch.empty(tensor.shape, dtype=tensor.target_dtype, device="meta") for tensor in self.tensors}
        self.manifest = SharedWeightManifest.from_tensors(
            metadata_tensors,
            weight_signature=signature,
            alignment=4096,
        )
        self.allocation: SharedArenaAllocation | None = None

    def _validate_config(self, lora_path) -> None:
        validate_shared_weight_config(self.config)
        errors = []
        if not self.config.get("cpu_offload", False):
            errors.append("cpu_offload must be true")
        if self.config.get("offload_granularity") != "block":
            errors.append("offload_granularity must be 'block'")
        if self.config.get("lazy_load", False):
            errors.append("lazy_load must be false")
        if not self.config.get("dit_quantized", False) or self.config.get("dit_quant_scheme") != "fp8-vllm":
            errors.append("Wan shared adapter requires dit_quantized=true and dit_quant_scheme='fp8-vllm'")
        if self.config.get("tensor_parallel", False):
            errors.append("tensor_parallel is not supported")
        if self.config.get("weight_auto_quant", False):
            errors.append("weight_auto_quant is not supported")
        if self.config.get("adapter_model_path"):
            errors.append("adapter checkpoints are not supported")
        if self.config.get("lora_configs") or self.config.get("lora_dynamic_apply", False) or lora_path:
            errors.append("LoRA/diff weights are not supported")
        num_layers = self.config.get("num_layers")
        if type(num_layers) is not int or num_layers <= 0:
            errors.append("num_layers must be a positive integer")
        if self.infer_dtype != GET_SENSITIVE_DTYPE():
            errors.append("Wan shared adapter requires DTYPE and SENSITIVE_LAYER_DTYPE to match")
        checkpoint = self.config.get("dit_quantized_ckpt")
        if not checkpoint or not os.path.isdir(checkpoint):
            errors.append("dit_quantized_ckpt must be a directory containing block_*.safetensors")
        if errors:
            raise ValueError("Invalid shared_cpu_weights configuration: " + "; ".join(errors))

    def _discover_block_files(self) -> dict[int, Path]:
        result = {}
        for path in self.checkpoint_dir.iterdir():
            match = _BLOCK_FILE_RE.match(path.name)
            if match:
                result[int(match.group(1))] = path
        expected = set(range(self.config["num_layers"]))
        found = set(result)
        if found != expected:
            raise ValueError(f"Wan shared block checkpoint files do not match num_layers (missing={sorted(expected - found)}, unexpected={sorted(found - expected)})")
        return result

    def _inspect_checkpoint(self) -> tuple[tuple[_CheckpointTensor, ...], str]:
        tensors = []
        signature_files = []
        canonical_keys = None

        for block_index in sorted(self.block_files):
            path = self.block_files[block_index]
            header = _read_header(str(path))
            entries = {name: value for name, value in header.items() if name != "__metadata__"}
            if not entries:
                raise ValueError(f"No tensors found in {path}")
            block_canonical_keys = set()
            for name in sorted(entries):
                key_match = _BLOCK_KEY_RE.match(name)
                if key_match is None or int(key_match.group(1)) != block_index:
                    raise ValueError(f"Tensor {name!r} in {path} does not belong to block {block_index}")
                block_canonical_keys.add(_BLOCK_KEY_RE.sub("blocks.*.", name, count=1))

                raw_dtype = entries[name]["dtype"]
                try:
                    source_dtype = _SAFETENSORS_DTYPES[raw_dtype]
                except (KeyError, TypeError) as error:
                    raise ValueError(f"Unsupported safetensors dtype {raw_dtype!r} for {name}") from error
                target_dtype, converted = _target_dtype(name, source_dtype, self.infer_dtype)
                tensors.append(
                    _CheckpointTensor(
                        name=name,
                        path=str(path),
                        target_dtype=target_dtype,
                        shape=tuple(entries[name]["shape"]),
                        cast_scale_through_infer_dtype=name.endswith(".weight_scale") and converted,
                    )
                )

            if canonical_keys is None:
                canonical_keys = block_canonical_keys
            elif block_canonical_keys != canonical_keys:
                raise ValueError(
                    f"Wan block {block_index} tensor schema differs from block 0 (missing={sorted(canonical_keys - block_canonical_keys)}, unexpected={sorted(block_canonical_keys - canonical_keys)})"
                )
            signature_files.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "payload_sha256": _checkpoint_content_digest(path),
                }
            )

        signature_payload = {
            "format": self.FORMAT_VERSION,
            "files": signature_files,
            "infer_dtype": str(self.infer_dtype),
            "num_layers": self.config["num_layers"],
        }
        signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return tuple(tensors), signature

    def _populate(self, views: Mapping[str, torch.Tensor]) -> None:
        specs_by_file: dict[str, list[_CheckpointTensor]] = {}
        for tensor in self.tensors:
            specs_by_file.setdefault(tensor.path, []).append(tensor)

        logger.info(f"[SharedCPUWeightsInfo] Populating {len(self.block_files)} Wan block files from {self.checkpoint_dir}")
        for _, path in sorted(self.block_files.items()):
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                for tensor_spec in specs_by_file[str(path)]:
                    source = checkpoint.get_tensor(tensor_spec.name)
                    if tensor_spec.cast_scale_through_infer_dtype:
                        source = source.to(self.infer_dtype).to(torch.float32)
                    elif source.dtype != tensor_spec.target_dtype:
                        source = source.to(tensor_spec.target_dtype)
                    destination = views[tensor_spec.name]
                    if tuple(source.shape) != tuple(destination.shape) or source.dtype != destination.dtype:
                        raise ValueError(f"Converted tensor {tensor_spec.name} is {tuple(source.shape)} {source.dtype}, expected {tuple(destination.shape)} {destination.dtype}")
                    destination.copy_(source)

    def materialize(self) -> SharedArenaAllocation:
        if self.allocation is not None:
            raise RuntimeError("Wan shared block arena has already been materialized")
        chunk_mb = self.config.get("shared_cpu_weight_register_chunk_mb", 128)
        self.allocation = materialize_shared_weight_arena(
            self.manifest,
            self._populate,
            scope=self.config.get("shared_cpu_weight_scope", "auto"),
            strict_numa=self.config.get("shared_cpu_weight_strict_numa", True),
            register_chunk_bytes=chunk_mb * 1024 * 1024,
        )
        return self.allocation

    def build_weight_map(self, private_weights: Mapping[str, torch.Tensor]) -> SharedWeightViewMap[torch.Tensor]:
        allocation = self.allocation if self.allocation is not None else self.materialize()
        shared_views = allocation.tensor_views()
        return SharedWeightViewMap(private_weights, shared_views, owner=allocation)
