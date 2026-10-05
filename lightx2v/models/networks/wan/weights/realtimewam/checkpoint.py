"""Read-only loading of native WAM checkpoints and PEFT linear LoRA adapters.

No dependency on the training repository or PEFT's model constructors. Base
weights and adapters are never modified on disk. Unsupported PEFT variants fail
explicitly rather than producing a partially merged policy.
"""

import json
import math
import re
from pathlib import Path

import torch


def merge_linear_lora(base, adapter, config, prefix):
    if config.get("peft_type", "LORA") != "LORA":
        raise ValueError("Only linear LoRA adapters are supported")
    for option in ("use_dora", "use_qalora", "fan_in_fan_out", "lora_bias", "layer_replication"):
        if config.get(option):
            raise ValueError(f"Unsupported adapter option: {option}")
    if config.get("bias", "none") != "none":
        raise ValueError("LoRA bias training is not supported")
    state = {k.removeprefix("base_model.model."): v for k, v in adapter.items()}
    consumed = set()
    merged_count = 0

    def pattern_value(patterns, module, default):
        for pattern, value in (patterns or {}).items():
            if re.fullmatch(r"(?:.*\.)?" + pattern, module):
                return value
        return default

    for key, a in state.items():
        match = re.fullmatch(r"(.+)\.lora_A(?:\.default)?\.weight", key)
        if not match:
            continue
        module = match.group(1)
        b_key = key.replace(".lora_A", ".lora_B")
        if b_key not in state:
            raise ValueError(f"Missing LoRA B for {key}")
        b = state[b_key]
        target = f"{prefix}.{module}.weight"
        if target not in base:
            raise ValueError(f"Adapter target absent from base checkpoint: {target}")
        rank = int(pattern_value(config.get("rank_pattern"), module, config["r"]))
        alpha = float(pattern_value(config.get("alpha_pattern"), module, config["lora_alpha"]))
        if rank <= 0 or a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"Invalid LoRA rank/shapes for {target}")
        delta = b.float() @ a.float()
        if delta.shape != base[target].shape:
            raise ValueError(f"LoRA/base shape mismatch: {target}")
        scale = alpha / (math.sqrt(rank) if config.get("use_rslora") else rank)
        # Match PEFT merge_and_unload(safe_merge=True): cast the delta to the
        # base dtype BEFORE addition. A single final cast changes BF16 rounding.
        merged = base[target] + (delta * scale).to(base[target].dtype)
        if not torch.isfinite(merged).all():
            raise ValueError(f"Non-finite merged weights: {target}")
        base[target] = merged
        consumed.update((key, b_key))
        merged_count += 1
    saved = config.get("modules_to_save") or []
    for key in state.keys() - consumed:
        target_key = key.replace(".modules_to_save.default.", ".")
        module = target_key.rsplit(".", 1)[0]
        target = f"{prefix}.{target_key}"
        if not any(module == m or module.endswith("." + m) for m in saved):
            raise ValueError(f"Unconsumed adapter key: {key}")
        if target not in base or state[key].shape != base[target].shape or not torch.isfinite(state[key]).all():
            raise ValueError(f"Invalid saved module: {target}")
        base[target] = state[key].to(base[target].dtype)
    if not merged_count:
        raise ValueError("Adapter contains no LoRA pairs")
    return merged_count


def load_checkpoint(base_path, lora_path=None, weights="ema"):
    payload = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ValueError("base_ckpt must be a native WAM checkpoint containing 'mot'")
    state = dict(payload["mot"])
    for key, value in payload.get("proprio_encoder", {}).items():
        state[f"proprio_encoder.{key}"] = value
    if lora_path:
        root = Path(lora_path)
        if (root / "adapter_config.json").is_file():
            roles = [("action", root)]
        else:
            roles = [(role, root / role) for role in ("action", "video") if (root / role / "adapter_config.json").is_file()]
        if roles:
            from safetensors.torch import load_file

            for role, folder in roles:
                config = json.loads((folder / "adapter_config.json").read_text())
                if (folder / "adapter_model.safetensors").is_file():
                    adapter = load_file(str(folder / "adapter_model.safetensors"))
                else:
                    adapter = torch.load(folder / "adapter_model.bin", map_location="cpu", weights_only=True)
                merge_linear_lora(state, adapter, config, f"mixtures.{role}")
        elif (root / "config.yaml").is_file():
            import yaml

            training = yaml.safe_load((root / "config.yaml").read_text())["training"]
            role_names = [("action", "student", f"{weights}_action.pt")]
            if training.get("unfreeze_video", False):
                role_names.append(("video", "video", "video.pt"))
            for role, cfg_key, filename in role_names:
                role_config = training[cfg_key]
                if role_config["train_type"] != "lora":
                    raise ValueError("lora_path must contain LoRA weights, not a full fine-tune")
                lc = role_config["lora"]
                cfg = {"r": lc["rank"], "lora_alpha": lc.get("alpha", lc["rank"]), "modules_to_save": lc.get("modules_to_save", [])}
                adapter = torch.load(root / filename, map_location="cpu", weights_only=True)
                merge_linear_lora(state, adapter, cfg, f"mixtures.{role}")
        else:
            raise ValueError(f"No PEFT adapter or training checkpoint found in {root}")
    return state
