"""H3-World action-conditioned transformer using native MiniMax-H3 weights."""

import math
import os

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3_world.infer.pre_infer import MiniMaxH3WorldPreInfer
from lightx2v.models.networks.minimax_h3_world.infer.transformer_infer import MiniMaxH3WorldOffloadTransformerInfer, MiniMaxH3WorldTransformerInfer
from lightx2v.models.networks.minimax_h3_world.lora import minimax_h3_lora_shape, normalize_minimax_h3_lora_sources, read_minimax_h3_lora_tensor
from lightx2v.utils.envs import GET_DTYPE


class MiniMaxH3WorldModel(MiniMaxH3Model):
    def _init_infer_class(self):
        super()._init_infer_class()
        if self.config.get("seq_parallel", False):
            raise NotImplementedError("H3-World ia2av requires the full packed sequence; sequence parallel is not supported")
        self.pre_infer_class = MiniMaxH3WorldPreInfer
        self.transformer_infer_class = MiniMaxH3WorldOffloadTransformerInfer if self.cpu_offload else MiniMaxH3WorldTransformerInfer

    def _validate_dynamic_lora_shapes(self, source, normalized_sources, down_names):
        model_keys = set()
        ranks = set()
        for down_name in sorted(down_names):
            base_name = down_name[: -len(".lora_down.weight")]
            up_name = base_name + ".lora_up.weight"
            model_key = base_name + ".weight"
            num_heads = int(self.config.get("num_attention_heads", 56))
            head_dim = int(self.config.get("attention_head_dim", 128))
            down_shape = minimax_h3_lora_shape(source, normalized_sources[down_name], num_heads, head_dim)
            up_shape = minimax_h3_lora_shape(source, normalized_sources[up_name], num_heads, head_dim)
            if len(down_shape) != 2 or len(up_shape) != 2 or down_shape[0] != up_shape[1]:
                raise ValueError(f"Invalid MiniMax-H3 LoRA pair for {model_key}: down={down_shape}, up={up_shape}")

            expected_shape = [up_shape[0], down_shape[1]]
            if self.use_tp:
                split_type = self._tp_split_type(model_key)
                if split_type == "row":
                    if expected_shape[1] % self.tp_size:
                        raise ValueError(f"Cannot row-shard MiniMax-H3 LoRA {model_key} shape {tuple(expected_shape)} across TP size {self.tp_size}")
                    expected_shape[1] //= self.tp_size
                elif split_type is not None:
                    if expected_shape[0] % self.tp_size:
                        raise ValueError(f"Cannot column-shard MiniMax-H3 LoRA {model_key} shape {tuple(expected_shape)} across TP size {self.tp_size}")
                    expected_shape[0] //= self.tp_size

            base_shape = self._h3_weight_shapes.get(model_key)
            if base_shape is None:
                raise KeyError(f"MiniMax-H3 LoRA target does not exist in the loaded model: {model_key}")
            if tuple(expected_shape) != base_shape:
                raise ValueError(f"MiniMax-H3 LoRA shape mismatch for {model_key}: LoRA={tuple(expected_shape)}, base={base_shape}")
            model_keys.add(model_key)
            ranks.add(down_shape[0])
        return model_keys, ranks

    def _load_lora_file(self, file_path, alpha=None):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"MiniMax-H3 LoRA file not found: {file_path}")

        effective_alpha = self.lora_alpha if alpha is None else alpha
        if effective_alpha is not None:
            effective_alpha = float(effective_alpha)
            if not math.isfinite(effective_alpha) or effective_alpha <= 0:
                raise ValueError(f"MiniMax-H3 LoRA alpha must be finite and positive, got {effective_alpha}")

        load_device = self._checkpoint_load_device()
        with safe_open(file_path, framework="pt", device=load_device) as source:
            normalized_sources = normalize_minimax_h3_lora_sources(source.keys())

            down_names = {key for key in normalized_sources if key.endswith(".lora_down.weight")}
            up_names = {key for key in normalized_sources if key.endswith(".lora_up.weight")}
            expected_up_names = {key[: -len(".lora_down.weight")] + ".lora_up.weight" for key in down_names}
            if not down_names or up_names != expected_up_names:
                missing_up = sorted(expected_up_names - up_names)
                orphan_up = sorted(up_names - expected_up_names)
                raise ValueError(f"MiniMax-H3 dynamic LoRA has incomplete pairs: missing_up={missing_up[:3]}, orphan_up={orphan_up[:3]}")

            model_keys, ranks = self._validate_dynamic_lora_shapes(source, normalized_sources, down_names)
            expected_alpha_names = {key[: -len(".lora_down.weight")] + ".alpha" for key in down_names}
            alpha_names = {key for key in normalized_sources if key.endswith(".alpha")}
            orphan_alpha = sorted(alpha_names - expected_alpha_names)
            if orphan_alpha:
                raise ValueError(f"MiniMax-H3 dynamic LoRA contains alpha tensors without matching pairs: {orphan_alpha[:3]}")
            missing_alpha = expected_alpha_names - alpha_names
            if missing_alpha and effective_alpha is None:
                raise ValueError("MiniMax-H3 dynamic LoRA requires an alpha in lora_configs because the checkpoint has no per-layer alpha tensors")

            lora_weights = {}
            for normalized_key, entry in normalized_sources.items():
                tensor = read_minimax_h3_lora_tensor(source, entry, int(self.config.get("num_attention_heads", 56)), int(self.config.get("attention_head_dim", 128))).to(GET_DTYPE())
                lora_weights[normalized_key] = tensor.pin_memory() if torch.device(load_device).type == "cpu" else tensor
            for alpha_name in missing_alpha:
                alpha_tensor = torch.tensor(effective_alpha, dtype=GET_DTYPE(), device=load_device)
                lora_weights[alpha_name] = alpha_tensor.pin_memory() if alpha_tensor.device.type == "cpu" else alpha_tensor

        self._pending_dynamic_lora_model_keys = model_keys
        if effective_alpha is not None:
            self.lora_alpha = effective_alpha
        logger.info(
            "Loaded MiniMax-H3 dynamic LoRA {} (pairs={}, ranks={}, alpha={}, device={})",
            file_path,
            len(model_keys),
            sorted(ranks),
            effective_alpha,
            load_device,
        )
        return lora_weights
