import json
import re
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.tensor import DTensor


def _adapter_key(name, adapter_name):
    return (
        name.removeprefix("transformer.")
        .replace(".lora.down.weight", ".lora_A.weight")
        .replace(".lora.up.weight", ".lora_B.weight")
        .replace(f".lora_A.{adapter_name}.weight", ".lora_A.weight")
        .replace(f".lora_B.{adapter_name}.weight", ".lora_B.weight")
    )


def _pattern_value(pattern, name, default):
    return next((value for key, value in pattern.items() if re.fullmatch(rf"(.*\.)?({key})", name)), default)


def _validate_adapter_config(model, path, adapter_name):
    preset = path / "training_adapter.json"
    if preset.is_file() and json.loads(preset.read_text()).get("inference_requires_preset", False):
        raise ValueError("This H3 LoRA requires a different frozen student base.")

    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or adapter_name not in module.lora_A:
            continue
        rank = _pattern_value(config.get("rank_pattern", {}), name, config["r"])
        alpha = _pattern_value(config.get("alpha_pattern", {}), name, config["lora_alpha"])
        scale = alpha / (rank**0.5 if config.get("use_rslora", False) else rank)
        if rank != module.r[adapter_name] or scale != module.scaling[adapter_name]:
            raise ValueError(f"H3 LoRA rank/alpha mismatch for {name} in {config_path}.")


@torch.no_grad()
def load_h3_lora(model, path: Path, adapter_name: str):
    file = path / "pytorch_lora_weights.safetensors" if path.is_dir() else path
    _validate_adapter_config(model, file.parent, adapter_name)
    expected = {
        _adapter_key(name, adapter_name): (name, parameter) for name, parameter in model.named_parameters() if name.endswith((f".lora_A.{adapter_name}.weight", f".lora_B.{adapter_name}.weight"))
    }
    sharded = any(isinstance(parameter, DTensor) for _, parameter in expected.values())
    with safe_open(str(file), framework="pt", device="cpu") as weights:
        paths = {_adapter_key(key, adapter_name): key for key in weights.keys()}
        if not expected or len(paths) != len(weights.keys()) or paths.keys() != expected.keys():
            missing = sorted(expected.keys() - paths.keys())
            extra = sorted(paths.keys() - expected.keys())
            raise ValueError(f"H3 LoRA keys do not match {file}: missing={missing[:5]}, extra={extra[:5]}.")
        for key, (_, parameter) in expected.items():
            if tuple(weights.get_slice(paths[key]).get_shape()) != tuple(parameter.shape):
                raise ValueError(f"H3 LoRA tensor {key} has a different rank or shape in {file}.")
        if sharded:
            state = {name: weights.get_tensor(paths[key]) for key, (name, _) in expected.items()} if dist.get_rank() == 0 else {}
            # This checkpoint contains only the adapter parameters.
            options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
            set_model_state_dict(model, state, options=options)
        else:
            for key, (_, parameter) in expected.items():
                parameter.copy_(weights.get_tensor(paths[key]))
