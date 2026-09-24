"""Read H3-World's legacy, head-interleaved LoRA into native H3 layers."""

import re

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.minimax_h3.lora import MiniMaxH3LoraAdapter


def normalize_minimax_h3_lora_sources(keys):
    """Map native/PEFT and H3-World tensors to native keys without reading weights.

    An entry is ``(source_key, qkv_component)``. H3-World packs Q/K/V per
    attention head; only the up factor needs the corresponding row selection.
    Down factors and optional alpha scalars are shared by the three branches.
    """
    sources = {}
    unsupported = []
    for source_key in keys:
        key = MiniMaxH3LoraAdapter._normalize_lora_key(source_key)
        if key is None:
            unsupported.append(source_key)
            continue
        match = re.match(r"^(blocks\.\d+|token_refiner\.blocks\.\d+)\.attn\.(qkv_proj|out_proj)(\..+)$", key)
        if match:
            prefix, projection, suffix = match.groups()
            prefix = prefix.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
            if prefix.startswith("blocks."):
                prefix = "transformer_" + prefix
            if projection == "qkv_proj":
                entries = [(f"{prefix}.attn.to_{component}{suffix}", index if suffix == ".lora_up.weight" else None) for index, component in enumerate("qkv")]
            else:
                entries = [(f"{prefix}.attn.to_out.0{suffix}", None)]
        else:
            entries = [(key, None)]
        for normalized_key, component in entries:
            if normalized_key in sources:
                raise ValueError(f"MiniMax-H3 LoRA keys collide after normalization: {source_key} and {sources[normalized_key][0]}")
            sources[normalized_key] = (source_key, component)
    if unsupported:
        raise ValueError(f"MiniMax-H3 LoRA contains {len(unsupported)} unsupported tensors: {unsupported[:4]}")
    return sources


def minimax_h3_lora_shape(source, entry, num_heads=56, head_dim=128):
    source_key, component = entry
    shape = tuple(source.get_slice(source_key).get_shape())
    if component is not None:
        if len(shape) != 2 or shape[0] != num_heads * 3 * head_dim:
            raise ValueError(f"Invalid head-interleaved H3-World QKV LoRA shape for {source_key}: {shape}")
        return (num_heads * head_dim, shape[1])
    return shape


def read_minimax_h3_lora_tensor(source, entry, num_heads=56, head_dim=128):
    source_key, component = entry
    shape = minimax_h3_lora_shape(source, entry, num_heads, head_dim)
    tensor = source.get_tensor(source_key)
    if component is not None:
        tensor = tensor.reshape(num_heads, 3, head_dim, shape[1])[:, component].reshape(shape).contiguous()
    return tensor


class MiniMaxH3WorldLoraAdapter(MiniMaxH3LoraAdapter):
    """Reuse native merge/device/sharding logic with legacy QKV row selection."""

    @torch.no_grad()
    def _merge_file(self, path, strength=1.0, alpha=None):
        with safe_open(path, framework="pt", device="cpu") as source:
            normalized_sources = normalize_minimax_h3_lora_sources(source.keys())
            num_heads = int(self.model.config.get("num_attention_heads", 56))
            head_dim = int(self.model.config.get("attention_head_dim", 128))

            down_names = {key for key in normalized_sources if key.endswith(".lora_down.weight")}
            up_names = {key for key in normalized_sources if key.endswith(".lora_up.weight")}
            expected_up_names = {key[: -len(".lora_down.weight")] + ".lora_up.weight" for key in down_names}
            if not down_names or up_names != expected_up_names:
                missing_up = sorted(expected_up_names - up_names)
                orphan_up = sorted(up_names - expected_up_names)
                raise ValueError(f"MiniMax-H3 LoRA has incomplete pairs: missing_up={missing_up[:3]}, orphan_up={orphan_up[:3]}")

            expected_alpha_names = {key[: -len(".lora_down.weight")] + ".alpha" for key in down_names}
            alpha_names = {key for key in normalized_sources if key.endswith(".alpha")}
            orphan_alpha = sorted(alpha_names - expected_alpha_names)
            if orphan_alpha:
                raise ValueError(f"MiniMax-H3 LoRA contains alpha tensors without matching pairs: {orphan_alpha[:3]}")
            if alpha is None and expected_alpha_names - alpha_names:
                raise ValueError("MiniMax-H3 merged LoRA requires lora_configs[].alpha when the checkpoint has no per-layer alpha tensors")

            pairs = {}
            for down_name in sorted(down_names):
                base_name = down_name[: -len(".lora_down.weight")]
                model_key = base_name + ".weight"
                if model_key not in self.model.original_weight_dict:
                    raise KeyError(f"MiniMax-H3 LoRA target does not exist in the loaded transformer: {model_key}")
                pairs[model_key] = {
                    "down_key": normalized_sources[down_name],
                    "up_key": normalized_sources[base_name + ".lora_up.weight"],
                    "alpha_key": normalized_sources.get(base_name + ".alpha"),
                }

            if not pairs:
                raise ValueError(f"No supported LoRA pairs found in {path}")

            for index, (model_key, pair) in enumerate(pairs.items(), start=1):
                parameter = self.model.original_weight_dict[model_key]
                lora_up = read_minimax_h3_lora_tensor(source, pair["up_key"], num_heads, head_dim)
                lora_down = read_minimax_h3_lora_tensor(source, pair["down_key"], num_heads, head_dim)
                lora_up, lora_down = self._shard_factors(model_key, lora_up, lora_down)
                merge_device = self._merge_device(parameter)
                lora_up = lora_up.to(device=merge_device, dtype=parameter.dtype)
                lora_down = lora_down.to(device=merge_device, dtype=parameter.dtype)

                pair_alpha = read_minimax_h3_lora_tensor(source, pair["alpha_key"], num_heads, head_dim).item() if pair["alpha_key"] is not None else None
                effective_alpha = pair_alpha if pair_alpha is not None else alpha
                scale = float(effective_alpha) / lora_down.shape[0] if effective_alpha is not None else 1.0
                delta = torch.mm(lora_up, lora_down)
                if delta.shape != parameter.shape:
                    raise ValueError(f"LoRA delta shape mismatch for {model_key}: delta={tuple(delta.shape)}, weight={tuple(parameter.shape)}")
                parameter.add_(
                    delta.to(parameter.device),
                    alpha=scale * float(strength),
                )
                del lora_up, lora_down, delta

                if index % 24 == 0 or index == len(pairs):
                    logger.info(
                        "Merged MiniMax-H3 LoRA layers: {}/{}",
                        index,
                        len(pairs),
                    )

        logger.info(
            "Successfully merged MiniMax-H3 LoRA: {} (layers={}, strength={})",
            path,
            len(pairs),
            strength,
        )
        return len(pairs)
