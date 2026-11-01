"""Read-only provenance checks for offline DeCFG score checkpoints.

This module deliberately uses only the standard library. Launchers can audit
the merged base before importing Torch or creating any distributed process.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

MANIFEST_NAME = "decfg_merged.json"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_MODELS = {"transformer": "minimax_h3_t2av", "transformer_ref": "minimax_h3_ref2av"}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read premerged DeCFG metadata: {path}") from error


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _adapter_identity(value):
    _require(isinstance(value, dict) and value.get("enabled") is True, "Premerged score base must identify its enabled DeCFG adapter")
    digest, scale = value.get("sha256"), value.get("scale")
    roles, layout = value.get("roles"), value.get("qkv_layout")
    _require(_valid_digest(digest), "Premerged DeCFG adapter has no valid SHA256 identity")
    _require(not isinstance(scale, bool) and isinstance(scale, (int, float)) and math.isfinite(scale) and scale > 0, "Premerged DeCFG scale must be finite and positive")
    _require(isinstance(roles, list) and len(roles) == 2 and set(roles) == {"fake", "teacher"}, "Premerged DeCFG must be restricted to fake/teacher scores")
    _require(layout in {"comfy", "official"}, "Premerged DeCFG has an invalid QKV layout")
    _require(value.get("inference_requires_preset") is False, "Premerged DeCFG must not declare a student inference dependency")
    return digest, float(scale), tuple(sorted(roles)), layout


def _relative_file(directory, name):
    _require(isinstance(name, str) and bool(name) and not Path(name).is_absolute(), "Premerged DeCFG shard paths must be relative")
    relative = Path(name)
    _require(".." not in relative.parts, "Premerged DeCFG shard path escapes its transformer directory")
    path = directory / relative
    _require(path.resolve().is_relative_to(directory.resolve()), "Premerged DeCFG shard path escapes its transformer directory")
    return path


def validate_premerged_score_base(model_path, *, component_name, model_name=None, expected_adapter=None, required=False, verify_weights=False):
    """Validate the published manifest and small files without loading tensors.

    Shard sizes and index membership are always checked. Full shard SHA256
    verification is optional, so 32 training ranks do not reread an entire H3
    checkpoint just to validate provenance. ``expected_adapter`` compares the
    preset identity independently of its old machine's absolute file path.
    """
    _require(component_name in _MODELS, f"Unknown H3 transformer component: {component_name}")
    root = Path(model_path).expanduser().resolve()
    directory = root / component_name if (root / component_name / "config.json").is_file() else root
    candidates = [root / MANIFEST_NAME, directory / MANIFEST_NAME]
    if directory == root and (root / "config.json").is_file():
        candidates.append(root.parent / MANIFEST_NAME)
    manifests = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if not manifests:
        _require(not required and expected_adapter is None, f"Missing premerged DeCFG manifest below {root}")
        return None
    manifest = _json(manifests[0])
    _require(isinstance(manifest, dict), f"Invalid premerged DeCFG manifest: {manifests[0]}")
    _require(
        manifest.get("schema_version") == 1 and manifest.get("format") == "lightx2v_h3_score_decfg" and manifest.get("complete") is True,
        "Premerged DeCFG checkpoint is incomplete or has an unsupported format",
    )
    _require(manifest.get("precision") == "fp32_merged_targets" and manifest.get("student_uses_decfg") is False, "Premerged DeCFG must preserve FP32 merged score targets and exclude the student")
    _require(isinstance(manifest.get("source_model_path"), str) and Path(manifest["source_model_path"]).is_absolute(), "Premerged DeCFG manifest has no absolute original model identity")
    for other in manifests[1:]:
        _require(_json(other) == manifest, f"Root/component premerged DeCFG manifests disagree: {other}")
    components = manifest.get("components")
    _require(isinstance(components, dict) and set(components) <= set(_MODELS) and component_name in components, f"Premerged DeCFG has no matching component {component_name}")
    component = components[component_name]
    expected_name = _MODELS[component_name]
    _require(isinstance(component, dict) and component.get("model_name") == expected_name and model_name in {None, expected_name}, "Premerged DeCFG model/task mismatch")
    identity = _adapter_identity(component.get("training_adapter"))
    if expected_adapter is not None:
        _require(identity == _adapter_identity(expected_adapter), "Premerged DeCFG adapter differs from the CM score teacher (SHA256/scale/roles/layout)")
    _require(type(component.get("adapted_weights")) is int and component["adapted_weights"] > 0, "Premerged DeCFG manifest contains no adapted score weights")
    for filename, key in (("config.json", "config_sha256"), (INDEX_NAME, "index_sha256")):
        path = directory / filename
        _require(path.is_file() and _valid_digest(component.get(key)), f"Missing or invalid premerged DeCFG {filename}")
        _require(_digest(path) == component[key], f"Premerged DeCFG {filename} SHA256 mismatch")
    index = _json(directory / INDEX_NAME)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    _require(isinstance(weight_map, dict) and bool(weight_map) and all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()), "Invalid premerged DeCFG weight index")
    shards = component.get("shards")
    _require(isinstance(shards, list) and bool(shards), "Premerged DeCFG has no published weight shards")
    found = set()
    for shard in shards:
        _require(isinstance(shard, dict), "Invalid premerged DeCFG shard entry")
        path = _relative_file(directory, shard.get("path"))
        _require(
            shard["path"] not in found and _valid_digest(shard.get("sha256")) and type(shard.get("size_bytes")) is int and shard["size_bytes"] > 0,
            "Invalid or duplicate premerged DeCFG shard identity",
        )
        found.add(shard["path"])
        _require(path.is_file() and path.stat().st_size == shard["size_bytes"], f"Missing or truncated premerged DeCFG shard: {path}")
        if verify_weights:
            _require(_digest(path) == shard["sha256"], f"Premerged DeCFG shard SHA256 mismatch: {path}")
    _require(found == set(weight_map.values()), "Premerged DeCFG shards do not match the weight index")
    return {"manifest_path": str(manifests[0]), "transformer_dir": str(directory), "manifest": manifest, "component": component}


def validate_premerged_score_role(metadata, role):
    if metadata is not None:
        _require(role in {"fake", "teacher"}, "A premerged DeCFG base is only for fake/teacher scores; student rollout and inference require the original H3 base")
