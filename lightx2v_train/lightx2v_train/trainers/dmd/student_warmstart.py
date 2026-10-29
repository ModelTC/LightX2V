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
def load_student_adapter(module, checkpoint_path, *, strict=True):
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
