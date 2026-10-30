"""Optional DiffSynth DeCFG preset for H3 score-network experiments.

By default the teacher and fake-score networks receive DeCFG; the deployable
student stays on the original base and its learned LoRA has no preset
dependency. Explicit legacy/custom roles containing student remain supported
and are recorded as preset-dependent exports.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from torch.distributed.tensor import DTensor

_ROLES = ("student", "fake", "teacher")
_SUFFIXES = {"attn.out_proj": "attn.to_out.0", "mlp.fc1": "ff.net.0.proj", "mlp.fc2": "ff.net.2"}


@lru_cache(maxsize=8)
def _file_digest(path, size, mtime_ns):
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AdapterOptions:
    enabled: bool = False
    path: str = ""
    scale: float = 1.0
    roles: tuple[str, ...] = ("fake", "teacher")
    qkv_layout: str = "comfy"

    @classmethod
    def from_mapping(cls, value=None):
        if value is None:
            value = {}
        if not isinstance(value, dict) or set(value) - {"enabled", "path", "scale", "roles", "qkv_layout"}:
            raise ValueError("model.training_adapter must contain enabled/path/scale/roles/qkv_layout only")
        enabled = value.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("model.training_adapter.enabled must be a boolean")
        if not enabled:
            return cls()  # Disabled presets never inspect a path or touch weights.
        path = value.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Enabled model.training_adapter requires a safetensors path")
        path = Path(path).expanduser().resolve()
        if not path.is_file() or path.suffix != ".safetensors":
            raise FileNotFoundError(f"TrainingAdapter safetensors not found: {path}")
        scale = value.get("scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale):
            raise ValueError("model.training_adapter.scale must be finite")
        roles = value.get("roles", ["fake", "teacher"])
        if not isinstance(roles, (tuple, list)) or not roles or not all(isinstance(role, str) for role in roles) or len(set(roles)) != len(roles) or set(roles) - set(_ROLES):
            raise ValueError("model.training_adapter.roles must be a nonempty unique subset of student/fake/teacher")
        layout = value.get("qkv_layout", "comfy")
        if layout not in {"comfy", "official"}:
            raise ValueError("TrainingAdapter qkv_layout must be comfy or official")
        known_original = {"model.safetensors", "model_ref2va.safetensors"}
        known_comfy = {"model_for_comfy_dit.safetensors", "model_ref2va_for_comfy_dit.safetensors"}
        if (path.name in known_original and layout != "official") or (path.name in known_comfy and layout != "comfy"):
            raise ValueError(f"TrainingAdapter filename/layout mismatch: {path.name}, qkv_layout={layout}")
        return cls(True, str(path), float(scale), tuple(role for role in _ROLES if role in roles), layout)

    def metadata(self):
        if not self.enabled:
            return {"enabled": False}
        stat = Path(self.path).stat()
        return {
            "enabled": True,
            "schema_version": 1,
            "path": self.path,
            "sha256": _file_digest(self.path, stat.st_size, stat.st_mtime_ns),
            "scale": self.scale,
            "roles": list(self.roles),
            "qkv_layout": self.qkv_layout,
            "inference_requires_preset": "student" in self.roles,
            "merge_precision": "fp32_product_then_base_dtype",
        }


def _base_linear(module):
    return module.get_base_layer() if hasattr(module, "get_base_layer") else module


def _source_blocks(transformer):
    for index in range(len(transformer.transformer_blocks)):
        yield f"blocks.{index}", f"transformer_blocks.{index}"
    for index in range(len(transformer.token_refiner.refiner_blocks)):
        yield f"token_refiner.blocks.{index}", f"token_refiner.refiner_blocks.{index}"


def _factor_names(handle):
    pairs = {}
    for raw in handle.keys():
        key = raw.removeprefix("diffusion_model.").removeprefix("transformer.")
        match = re.fullmatch(r"(.+)\.lora_([AB])(?:\.default)?\.weight", key)
        if match is None:
            raise ValueError(f"Unexpected TrainingAdapter tensor: {raw}")
        name, side = match.groups()
        if side in pairs.setdefault(name, {}):
            raise ValueError(f"Duplicate TrainingAdapter factor: {raw}")
        pairs[name][side] = raw
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("TrainingAdapter contains an incomplete A/B pair")
    return pairs


def _plans(transformer, handle, pairs):
    expected = {f"{source}.{suffix}" for source, _ in _source_blocks(transformer) for suffix in (*_SUFFIXES, "attn.qkv_proj")}
    if expected != set(pairs):
        raise ValueError(f"TrainingAdapter module coverage mismatch: missing={sorted(expected - pairs.keys())[:8]}, unexpected={sorted(pairs.keys() - expected)[:8]}")
    plans = []
    for source, target in _source_blocks(transformer):
        for suffix in (*_SUFFIXES, "attn.qkv_proj"):
            key = f"{source}.{suffix}"
            a_shape = tuple(handle.get_slice(pairs[key]["A"]).get_shape())
            b_shape = tuple(handle.get_slice(pairs[key]["B"]).get_shape())
            targets = [(f"{target}.{_SUFFIXES[suffix]}", None)] if suffix != "attn.qkv_proj" else [(f"{target}.attn.to_{role}", role_index) for role_index, role in enumerate(("q", "k", "v"))]
            for target_name, qkv_role in targets:
                weight = _base_linear(transformer.get_submodule(target_name)).weight
                out_features, in_features = tuple(weight.shape)
                expected_rows = out_features * (3 if qkv_role is not None else 1)
                if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] <= 0 or a_shape[1] != in_features or b_shape != (expected_rows, a_shape[0]):
                    raise ValueError(f"TrainingAdapter factor shapes do not match {target_name}: A={a_shape}, B={b_shape}, weight={tuple(weight.shape)}")
                if weight.is_meta:
                    raise RuntimeError("TrainingAdapter must be applied after H3 streamed FSDP materialization")
                plans.append((key, target_name, weight, qkv_role, suffix == "mlp.fc1"))
    return plans


def _local_weight(weight):
    if not isinstance(weight, DTensor):
        return weight, (0, 0)
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    shape, offset = compute_local_shape_and_global_offset(weight.shape, weight.device_mesh, weight.placements)
    local = weight.to_local()
    if tuple(local.shape) != tuple(shape):
        raise RuntimeError("Unexpected FSDP local TrainingAdapter weight shape")
    return local, offset


def _source_rows(rows, *, output_rows, qkv_role, swap_halves, layout, head_dim):
    if swap_halves:
        return (rows + output_rows // 2) % output_rows
    if qkv_role is None:
        return rows
    if layout == "comfy":
        return rows + qkv_role * output_rows
    return (rows // head_dim * 3 + qkv_role) * head_dim + rows % head_dim


@torch.no_grad()
def merge_training_adapter(transformer, options):
    """Add one bounded FP32 delta tile to each destination's local shard.

    Only the small A/B matrices are read. Base tensors are never
    gathered, parameter objects are preserved, and no preset PEFT parameters
    enter the trainable adapter export. FFN fc1 halves are swapped to match
    Diffusers SwiGLU; official QKV rows are also deinterleaved when requested.
    """
    if not options.enabled:
        return 0
    with safe_open(options.path, framework="pt", device="cpu") as handle:
        pairs = _factor_names(handle)
        plans = _plans(transformer, handle, pairs)
        # Validate every factor before changing any destination parameter.
        for pair in pairs.values():
            for key in pair.values():
                tensor = handle.get_tensor(key)
                if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f"TrainingAdapter contains nonfinite/nonfloating factors: {key}")
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            cached_key, a, b = None, None, None
            for key, target_name, weight, qkv_role, swap_halves in plans:
                if key != cached_key:
                    a = handle.get_tensor(pairs[key]["A"]).float()
                    b = handle.get_tensor(pairs[key]["B"]).float()
                    cached_key = key
                local, offsets = _local_weight(weight)
                if local.numel() == 0:
                    continue
                row_offset, column_offset = offsets
                a_local = a[:, column_offset : column_offset + local.shape[1]].to(device=local.device)
                output_rows = int(weight.shape[0])
                head_dim = int(getattr(transformer.config, "attention_head_dim", 128))
                for start in range(0, local.shape[0], 256):
                    stop = min(start + 256, local.shape[0])
                    rows = torch.arange(row_offset + start, row_offset + stop)
                    selected = _source_rows(rows, output_rows=output_rows, qkv_role=qkv_role, swap_halves=swap_halves, layout=options.qkv_layout, head_dim=head_dim)
                    delta = b.index_select(0, selected).to(device=local.device) @ a_local
                    merged = local[start:stop].float().add_(delta, alpha=options.scale)
                    local[start:stop].copy_(merged.to(dtype=local.dtype))
                del a_local
            del a, b
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    logger.info("[training_adapter] merged preset path={} targets={} scale={} layout={} local_shards_only=true", options.path, len(plans), options.scale, options.qkv_layout)
    return len(plans)


def write_export_metadata(directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    semantics = "learned_lora_only_requires_preset" if metadata.get("enabled") else "learned_lora_only"
    payload = {**metadata, "export_semantics": semantics}
    temporary = directory / ".training_adapter.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(directory / "training_adapter.json")
