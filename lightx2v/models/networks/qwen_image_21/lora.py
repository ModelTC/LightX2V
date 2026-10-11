"""Qwen-Image-2.1 metadata and TP adapters for the shared LoRA loader."""

import json
import math
import os

import torch
from safetensors import safe_open

from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v.utils.lora_loader import LoRALoader


def validate_lora_config(config):
    loras = config.get("lora_configs") or []
    if config.get("lora_dynamic_apply", False) and len(loras) != 1:
        raise ValueError("qwen_image_21 dynamic LoRA requires exactly one lora_configs entry")
    if loras:
        for key in ("dit_quantized", "weight_auto_quant", "shared_cpu_weights", "dit_disk_streaming"):
            if config.get(key):
                raise ValueError(f"qwen_image_21 LoRA does not support {key}")


def prepare_lora_weights(model, path, weights):
    weights = {key.replace(".default.weight", ".weight"): value for key, value in weights.items()}
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = json.loads((source.metadata() or {}).get("lora_adapter_metadata", "{}"))
    if any(metadata.get("transformer." + key) for key in ("use_rslora", "use_dora", "alpha_pattern")):
        raise ValueError("Qwen-Image-2.1 LoRA does not support rsLoRA, DoRA or alpha_pattern")
    alpha = next((item.get("alpha") for item in model.config.get("lora_configs") or [] if os.path.abspath(item["path"]) == os.path.abspath(path)), None)
    if alpha is None:
        alpha = metadata.get("transformer.lora_alpha")
    pairs = LoRALoader().extract_lora_pairs(weights)
    expected = {pair[name] for pair in pairs.values() for name in ("down_key", "up_key")} | {pair["base_key"] + ".alpha" for pair in pairs.values()}
    if not pairs or weights.keys() - expected:
        raise ValueError("Qwen-Image-2.1 LoRA contains unsupported or incomplete pairs")
    shapes = model._lora_weight_shapes if hasattr(model, "_lora_weight_shapes") else {key: value.shape for key, value in model.original_weight_dict.items()}
    for key, pair in pairs.items():
        down, up = weights[pair["down_key"]], weights[pair["up_key"]]
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or down.shape[0] != up.shape[1]:
            raise ValueError(f"Invalid Qwen-Image-2.1 LoRA pair shapes: {key}")
        # Validate against the local checkpoint, leaving factors unsharded for
        # dynamic MMWeightTP registration. Merged loading shards them below.
        shape = [up.shape[0], down.shape[1]]
        split = model._tp_split_type(key) if model.use_tp else None
        if split:
            dim = 1 if split == "row" else 0
            if shape[dim] % model.tp_size:
                raise ValueError(f"Cannot shard Qwen-Image-2.1 LoRA: {key}")
            shape[dim] //= model.tp_size
        if tuple(shape) != tuple(shapes.get(key, ())):
            raise ValueError(f"Qwen-Image-2.1 LoRA target or shape mismatch: {key}")
        pair_alpha = pair["alpha"] if pair["alpha"] is not None else alpha
        if pair_alpha is None or not math.isfinite(float(pair_alpha)) or float(pair_alpha) <= 0:
            raise ValueError(f"Qwen-Image-2.1 LoRA requires a positive finite alpha: {key}")
        weights[pair["base_key"] + ".alpha"] = torch.tensor(pair_alpha, dtype=torch.float32, device=down.device)
    return weights


class QwenImage21LoraAdapter(LoraAdapter):
    def _load_lora_file(self, file_path):
        weights = self.model._load_lora_file(file_path, dtype=torch.float32 if self.merge_force_fp32 else None)
        if self.model.use_tp:
            for key, pair in self.lora_loader.extract_lora_pairs(weights).items():
                split = self.model._tp_split_type(key)
                if split:
                    factor = pair["down_key"] if split == "row" else pair["up_key"]
                    weights[factor] = weights[factor].chunk(self.model.tp_size, dim=1 if split == "row" else 0)[self.model.tp_rank].contiguous()
        return weights
