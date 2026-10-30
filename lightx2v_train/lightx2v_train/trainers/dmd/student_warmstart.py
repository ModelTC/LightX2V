"""Load an exported student adapter without resuming its training state."""

import json
import math
import re
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict


def student_adapter_path(checkpoint_path):
    """Recognize explicit adapter exports; never infer EMA from a parent path."""
    path = Path(checkpoint_path)
    if path.is_dir():
        candidate = path / "pytorch_lora_weights.safetensors"
        return candidate if candidate.is_file() else None
    return path if path.suffix == ".safetensors" else None


def _preset_identity(metadata):
    """Compare the effective student base independently of preset file paths."""
    if not isinstance(metadata, dict) or type(metadata.get("enabled", False)) is not bool:
        raise RuntimeError("Student TrainingAdapter metadata must contain a boolean enabled flag")
    if not metadata.get("enabled", False):
        return None
    roles = metadata.get("roles")
    if not isinstance(roles, (tuple, list)) or not roles or not all(isinstance(role, str) for role in roles) or len(set(roles)) != len(roles) or set(roles) - {"student", "fake", "teacher"}:
        raise RuntimeError("Student TrainingAdapter metadata has invalid roles")
    if "student" not in roles:
        return None  # Score-only presets do not modify the student base.
    digest = metadata.get("sha256")
    scale = metadata.get("scale")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise RuntimeError("Student TrainingAdapter metadata has no valid SHA256 identity")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise RuntimeError("Student TrainingAdapter metadata has no finite scale")
    layout = metadata.get("qkv_layout")
    if layout not in {"comfy", "official"}:
        raise RuntimeError("Student TrainingAdapter metadata has an invalid QKV layout")
    return digest.lower(), float(scale), tuple(sorted(roles)), layout


def _preset_model_name(metadata):
    name = metadata.get("model_name")
    if name:
        return name
    component = metadata.get("transformer_component")
    if component in {"transformer", "transformer_ref"}:
        return "minimax_h3_ref2av" if component == "transformer_ref" else "minimax_h3_t2av"
    # Backward compatibility with the original, task-specific DeCFG exports.
    basename = Path(metadata.get("path", "")).name
    if basename in {"model.safetensors", "model_for_comfy_dit.safetensors"}:
        return "minimax_h3_t2av"
    if basename in {"model_ref2va.safetensors", "model_ref2va_for_comfy_dit.safetensors"}:
        return "minimax_h3_ref2av"
    return None


def validate_student_adapter_preset(checkpoint_path, *, expected_preset=None, model_name=None):
    """Validate an exported LoRA's frozen-base dependency before tensor loading.

    Old exports without a sidecar remain valid for an unmodified student base.
    Student-side DeCFG exports require matching preset identity and task.
    Score-only DeCFG is audit metadata, never an inference dependency of the
    student. Relocating a byte-identical preset between machines is valid.
    This helper reads only JSON and can be used by stage-transition preflights.
    """
    path = student_adapter_path(checkpoint_path)
    if path is None or not path.is_file():
        raise RuntimeError(f"Student LoRA adapter not found: {checkpoint_path}")
    expected = _preset_identity(expected_preset or {"enabled": False})
    sidecar = path.parent / "training_adapter.json"
    if not sidecar.is_file():
        if expected is not None:
            raise RuntimeError(f"Student TrainingAdapter warm-start requires training_adapter.json: {sidecar}")
        return None
    try:
        source = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid student TrainingAdapter sidecar: {sidecar}") from error
    source_identity = _preset_identity(source)
    if source.get("export_role", "student") != "student":
        raise RuntimeError(f"Student warm-start cannot load a non-student TrainingAdapter export: {sidecar}")
    if source.get("enabled") and source_identity is None:
        raise RuntimeError(f"Student TrainingAdapter export does not identify an adapted student base: {sidecar}")
    if source_identity != expected:
        raise RuntimeError(f"Student TrainingAdapter preset differs (enabled/SHA256/scale/roles/layout): {sidecar}; keep the same frozen student base across stages")
    if source_identity is not None:
        if source.get("schema_version") != 1 or source.get("export_semantics") != "learned_lora_only_requires_preset":
            raise RuntimeError(f"Unsupported student TrainingAdapter export semantics: {sidecar}")
        source_name = _preset_model_name(source)
        if model_name is None or source_name is None or source_name != model_name:
            raise RuntimeError(f"Student TrainingAdapter model/task mismatch: exported={source_name!r}, configured={model_name!r}; {sidecar}")
    else:
        if source.get("inference_requires_preset", False) is not False or source.get("export_semantics") not in {None, "learned_lora_only"}:
            raise RuntimeError(f"Unmodified student export must not require a TrainingAdapter at inference: {sidecar}")
        if source.get("schema_version", 1) != 1:
            raise RuntimeError(f"Unsupported student export schema: {sidecar}")
        source_name = _preset_model_name(source)
        if model_name is not None and source_name is not None and source_name != model_name:
            raise RuntimeError(f"Student TrainingAdapter model/task mismatch: exported={source_name!r}, configured={model_name!r}; {sidecar}")
    return source


def _adapter_key(name):
    name = name.removeprefix("transformer.")
    name = name.replace(".lora.down.weight", ".lora_A.weight")
    name = name.replace(".lora.up.weight", ".lora_B.weight")
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def _pattern_value(pattern, module_name, default):
    # Match PEFT's suffix patterns, including its first-matching-key rule.
    key = next((key for key in pattern if re.match(rf"(.*\.)?({key})$", module_name)), None)
    return pattern.get(key, default)


def _validate_scaling(module, adapter_path):
    config_path = adapter_path.parent / "adapter_config.json"
    if not config_path.is_file():
        logger.warning("[train] student adapter has no adapter_config.json; ranks are checked from tensors and alpha comes from the current training config: {}", adapter_path)
        return
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    mismatches = []
    for name, layer in module.named_modules():
        if not hasattr(layer, "lora_A") or "default" not in layer.lora_A:
            continue
        rank = int(_pattern_value(config.get("rank_pattern", {}), name, config["r"]))
        alpha = float(_pattern_value(config.get("alpha_pattern", {}), name, config["lora_alpha"]))
        source_scale = alpha / (math.sqrt(rank) if config.get("use_rslora", False) else rank)
        if rank != layer.r["default"] or not math.isclose(source_scale, layer.scaling["default"], rel_tol=1e-9, abs_tol=1e-12):
            mismatches.append(f"{name}: checkpoint rank/scale={rank}/{source_scale}, configured={layer.r['default']}/{layer.scaling['default']}")
    if mismatches:
        raise RuntimeError(f"Student adapter rank/alpha configuration mismatch in {config_path}: {mismatches[:10]}")


@torch.no_grad()
def load_student_adapter(module, checkpoint_path, *, strict=True, expected_preset=None, model_name=None):
    """Validate adapter tensors, then copy them into dense or FSDP2 parameters.

    Call after adapter injection AND parallel.apply: H3's streamed loader
    materializes the base and initializes LoRA during FSDP setup. Loading before
    that point would either write to meta tensors or be overwritten. Only rank
    zero reads full weights; no frozen base tensors are gathered or replaced.
    Optimizers, fake/teacher, counters and RNG are deliberately untouched.
    """
    path = student_adapter_path(checkpoint_path)
    if path is None or not path.is_file():
        raise RuntimeError(f"Student LoRA adapter not found: {checkpoint_path}")
    validate_student_adapter_preset(path, expected_preset=expected_preset, model_name=model_name)

    expected = {}
    for name, parameter in module.named_parameters():
        if re.search(r"\.lora_[AB]\.default\.weight$", name):
            expected[_adapter_key(name)] = (name, tuple(parameter.shape))
    if not expected:
        raise RuntimeError("Student LoRA warm-start requires injected default LoRA A/B parameters; configure training.student.train_type=lora and matching target modules first.")

    source_keys = {}
    mismatches = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for raw_key in handle.keys():
            key = _adapter_key(raw_key)
            if key in source_keys:
                raise RuntimeError(f"Student adapter contains duplicate canonical key {key!r}: {path}")
            source_keys[key] = raw_key
            if key in expected:
                shape = tuple(handle.get_slice(raw_key).get_shape())
                if shape != expected[key][1]:
                    mismatches.append(f"{key}: checkpoint={shape}, configured={expected[key][1]}")
    missing = sorted(expected.keys() - source_keys.keys())
    unexpected = sorted(source_keys.keys() - expected.keys())
    if mismatches or (strict and (missing or unexpected)):
        raise RuntimeError(f"Student adapter does not match configured LoRA targets/ranks: missing={missing[:10]}, unexpected={unexpected[:10]}, shape_mismatches={mismatches[:10]}; checkpoint={path}")
    selected = sorted(expected.keys() & source_keys.keys())
    if not selected:
        raise RuntimeError(f"Student adapter has no keys matching the configured LoRA: {path}")
    if missing or unexpected:
        logger.warning("[train] partial student adapter warm-start missing={} unexpected={}", missing, unexpected)
    _validate_scaling(module, path)

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank_zero = not distributed or dist.get_rank() == 0
    state = {}
    if rank_zero:
        raw = load_file(str(path), device="cpu")
        state = {expected[key][0]: raw[source_keys[key]] for key in selected}
    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=distributed,
        strict=False,  # Only adapter keys are intentionally supplied.
    )
    set_model_state_dict(module, state, options=options)
    logger.info("[train] student LoRA warm-start path={} tensors={} strict={} weights_only=true; optimizer/fake/teacher/counters start fresh", path, len(selected), strict)
