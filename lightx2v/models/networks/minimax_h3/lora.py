"""LoRA merging for the native MiniMax-H3 inference implementation."""

import gc
import math
from collections import Counter

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v_platform.base.global_var import AI_DEVICE


def read_lora_pairs(source, config_alpha=None):
    """Read the published native H3 format; alpha tensors override metadata/config."""
    pairs = {}
    alpha_sources = Counter()
    keys = set(source.keys())
    down_suffix, up_suffix = ".lora_A.default.weight", ".lora_B.default.weight"
    prefixes = sorted(key[: -len(down_suffix)] for key in keys if key.endswith(down_suffix))
    expected = {prefix + suffix for prefix in prefixes for suffix in (down_suffix, up_suffix)}
    expected.update(prefix + ".alpha" for prefix in prefixes if prefix + ".alpha" in keys)
    if not prefixes or keys != expected:
        raise ValueError(f"Invalid native H3 LoRA keys: missing={sorted(expected - keys)[:3]}, unexpected={sorted(keys - expected)[:3]}")

    metadata_alpha = (source.metadata() or {}).get("alpha")
    for prefix in prefixes:
        down, up = prefix + down_suffix, prefix + up_suffix
        down_shape, up_shape = source.get_slice(down).get_shape(), source.get_slice(up).get_shape()
        if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1] or down_shape[0] < 1:
            raise ValueError(f"Invalid H3 LoRA factors for {prefix}: {down_shape}, {up_shape}")
        if prefix + ".alpha" in keys:
            alpha, origin = source.get_tensor(prefix + ".alpha").item(), "tensor"
        elif metadata_alpha is not None:
            alpha, origin = float(metadata_alpha), "metadata"
        elif config_alpha is not None:
            alpha, origin = float(config_alpha), "config"
        else:
            raise ValueError(f"{prefix} has no embedded alpha; set lora_configs[].alpha")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid H3 LoRA alpha for {prefix}: {alpha}")
        pairs[prefix] = (down, up, alpha)
        alpha_sources[origin, alpha] += 1
    for (origin, alpha), count in sorted(alpha_sources.items()):
        logger.info("H3 LoRA alpha: source={}, alpha={}, layers={}", origin, alpha, count)
    return pairs


def cached_modulation_shapes(config):
    """Projections replaced by the persistent AdaLN tables."""
    hidden = int(config.get("hidden_size", 5376))
    time_dim = int(config.get("time_embed_dim", 2688))
    shapes = {f"transformer_blocks.{i}.adaln_proj.linear.weight": (18 * hidden, time_dim) for i in range(int(config.get("num_layers", 50)))}
    shapes["norm_out.linear.weight"] = (2 * hidden, time_dim)
    return shapes


def merge_lora_weight(parameter, up, down, alpha, strength, *, lora_dtype, device):
    """Share the online/offline merge order, including the final BF16 rounding."""
    dtype = lora_dtype
    up, down = up.to(device=device, dtype=dtype), down.to(device=device, dtype=dtype)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        delta = torch.mm(up, down)
    if delta.shape != parameter.shape:
        raise ValueError(f"LoRA delta shape {tuple(delta.shape)} != weight {tuple(parameter.shape)}")
    merged = parameter.to(device=device, dtype=torch.promote_types(parameter.dtype, dtype))
    merged.add_(delta, alpha=alpha / down.shape[0] * float(strength))
    parameter.copy_(merged)


class MiniMaxH3LoraAdapter(LoraAdapter):
    """Merge Diffusers/PEFT LoRA weights into H3 checkpoint tensors.

    H3 checkpoints are large, so LoRA pairs are read and merged one linear
    layer at a time.  For tensor parallel inference the low-rank factors are
    sharded before multiplication, avoiding construction of a full dense
    delta on every rank.
    """

    def _merge_device(self, parameter):
        requested = self.model.config.get("lora_merge_device")
        if requested:
            return torch.device(requested)
        if parameter.device.type != "cpu":
            return parameter.device
        device_module = getattr(torch, AI_DEVICE, None)
        is_available = getattr(device_module, "is_available", lambda: False)
        if is_available():
            return torch.device(AI_DEVICE, device_module.current_device())
        return parameter.device

    def _shard_factors(self, model_key, lora_up, lora_down):
        if not self.model.use_tp:
            return lora_up, lora_down

        split_type = self.model._tp_split_type(model_key)
        if split_type is None:
            return lora_up, lora_down

        tp_size = self.model.tp_size
        tp_rank = self.model.tp_rank
        if split_type == "row":
            if lora_down.shape[1] % tp_size:
                raise ValueError(f"Cannot row-shard LoRA {model_key} with down shape {tuple(lora_down.shape)} across TP size {tp_size}")
            lora_down = torch.chunk(lora_down, tp_size, dim=1)[tp_rank]
        elif split_type == "ff_fused_col":
            if lora_up.shape[0] % 2:
                raise ValueError(f"Invalid fused SwiGLU LoRA up tensor for {model_key}: {tuple(lora_up.shape)}")
            value, gate = lora_up.chunk(2, dim=0)
            if value.shape[0] % tp_size:
                raise ValueError(f"Cannot fused-column-shard LoRA {model_key} with up shape {tuple(lora_up.shape)} across TP size {tp_size}")
            value = torch.chunk(value, tp_size, dim=0)[tp_rank]
            gate = torch.chunk(gate, tp_size, dim=0)[tp_rank]
            lora_up = torch.cat((value, gate), dim=0)
        else:
            if lora_up.shape[0] % tp_size:
                raise ValueError(f"Cannot column-shard LoRA {model_key} with up shape {tuple(lora_up.shape)} across TP size {tp_size}")
            lora_up = torch.chunk(lora_up, tp_size, dim=0)[tp_rank]
        return lora_up.contiguous(), lora_down.contiguous()

    @torch.no_grad()
    def _merge_file(self, path, strength=1.0, alpha=None):
        with safe_open(path, framework="pt", device="cpu") as source:
            pairs = read_lora_pairs(source, alpha)
            cached_shapes = cached_modulation_shapes(self.model.config) if self.model.use_adaln_cache else {}
            for index, (prefix, (down_key, up_key, pair_alpha)) in enumerate(pairs.items(), start=1):
                model_key = prefix + ".weight"
                if model_key in cached_shapes:
                    shape = (source.get_slice(up_key).get_shape()[0], source.get_slice(down_key).get_shape()[1])
                    if shape != cached_shapes[model_key]:
                        raise ValueError(f"Cached LoRA shape mismatch for {model_key}: {shape}")
                    continue
                parameter = self.model.original_weight_dict[model_key]
                up, down = self._shard_factors(model_key, source.get_tensor(up_key), source.get_tensor(down_key))
                merge_lora_weight(parameter, up, down, pair_alpha, strength, lora_dtype=self.lora_dtype, device=self._merge_device(parameter))
                del up, down
                if index % 24 == 0 or index == len(pairs):
                    logger.info("Merged MiniMax-H3 LoRA layers: {}/{}", index, len(pairs))

        logger.info(
            "Successfully merged MiniMax-H3 LoRA: {} (layers={}, strength={})",
            path,
            len(pairs),
            strength,
        )
        return len(pairs)

    def apply_lora(self, lora_configs):
        if not hasattr(self.model, "original_weight_dict"):
            raise RuntimeError("MiniMax-H3 base weights are no longer available for LoRA merging")
        if self.model.config.get("dit_quantized", False):
            raise ValueError("MiniMax-H3 merged LoRA inference requires original, non-quantized DiT weights")
        if self.model.config.get("lazy_load", False):
            raise ValueError("MiniMax-H3 lazy loading does not support LoRA merging")

        total = 0
        for lora_config in lora_configs:
            total += self._merge_file(
                lora_config["path"],
                strength=lora_config.get("strength", 1.0),
                alpha=lora_config.get("alpha"),
            )
        self.model._apply_weights(self.model.original_weight_dict)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total


__all__ = ["MiniMaxH3LoraAdapter"]
