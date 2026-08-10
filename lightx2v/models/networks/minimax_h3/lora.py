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
            keys = list(source.keys())
            # LoRALoader only needs values for optional scalar alpha tensors;
            # using None for matrices keeps the 1.38 GB H3 adapter streaming.
            key_index = {key: source.get_tensor(key) if key.endswith(".alpha") else None for key in keys}
            pairs = self.lora_loader.extract_lora_pairs(key_index)
            if not pairs:
                raise ValueError(f"No supported LoRA pairs found in {path}")

            paired_keys = {tensor_key for pair in pairs.values() for tensor_key in (pair["up_key"], pair["down_key"], pair["mid_key"]) if tensor_key is not None}
            unused = [key for key in keys if not key.endswith(".alpha") and key not in paired_keys]
            if unused:
                preview = ", ".join(unused[:4])
                raise ValueError(f"MiniMax-H3 LoRA contains {len(unused)} unsupported tensors: {preview}")

            missing = [model_key for model_key in pairs if model_key not in self.model.original_weight_dict]
            if missing:
                preview = ", ".join(missing[:4])
                raise KeyError(f"MiniMax-H3 LoRA contains {len(missing)} tensors that do not match the loaded transformer: {preview}")

            for index, (model_key, pair) in enumerate(pairs.items(), start=1):
                parameter = self.model.original_weight_dict[model_key]
                lora_up = source.get_tensor(pair["up_key"])
                lora_down = source.get_tensor(pair["down_key"])
                lora_up, lora_down = self._shard_factors(model_key, lora_up, lora_down)
                merge_device = self._merge_device(parameter)
                lora_up = lora_up.to(device=merge_device, dtype=parameter.dtype)
                lora_down = lora_down.to(device=merge_device, dtype=parameter.dtype)

                pair_alpha = pair["alpha"]
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
