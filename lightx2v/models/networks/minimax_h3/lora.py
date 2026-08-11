"""LoRA merging for the native MiniMax-H3 inference implementation."""

import gc

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.lora_adapter import LoraAdapter
from lightx2v_platform.base.global_var import AI_DEVICE


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

    @staticmethod
    def _normalize_lora_key(key: str) -> str | None:
        """Map supported H3 LoRA layouts to native down/up tensor names."""
        for prefix in (
            "base_model.model.",
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "model.",
        ):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break

        suffixes = {
            ".lora_A.default.weight": ".lora_down.weight",
            ".lora_B.default.weight": ".lora_up.weight",
            ".lora_A.weight": ".lora_down.weight",
            ".lora_B.weight": ".lora_up.weight",
            ".lora.down.weight": ".lora_down.weight",
            ".lora.up.weight": ".lora_up.weight",
            ".lora_down.weight": ".lora_down.weight",
            ".lora_up.weight": ".lora_up.weight",
        }
        for suffix, replacement in suffixes.items():
            if key.endswith(suffix):
                return key[: -len(suffix)] + replacement
        if key.endswith(".alpha"):
            return key
        return None

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
            normalized_sources = {}
            unsupported = []
            for source_key in source.keys():
                normalized_key = self._normalize_lora_key(source_key)
                if normalized_key is None:
                    unsupported.append(source_key)
                    continue
                if normalized_key in normalized_sources:
                    raise ValueError(f"MiniMax-H3 LoRA keys collide after normalization: {source_key} and {normalized_sources[normalized_key]}")
                normalized_sources[normalized_key] = source_key

            if unsupported:
                preview = ", ".join(unsupported[:4])
                raise ValueError(f"MiniMax-H3 LoRA contains {len(unsupported)} unsupported tensors: {preview}")

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
                lora_up = source.get_tensor(pair["up_key"])
                lora_down = source.get_tensor(pair["down_key"])
                lora_up, lora_down = self._shard_factors(model_key, lora_up, lora_down)
                merge_device = self._merge_device(parameter)
                lora_up = lora_up.to(device=merge_device, dtype=parameter.dtype)
                lora_down = lora_down.to(device=merge_device, dtype=parameter.dtype)

                pair_alpha = source.get_tensor(pair["alpha_key"]).item() if pair["alpha_key"] is not None else None
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
