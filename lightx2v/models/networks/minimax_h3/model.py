import glob
import math
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState
from lightx2v.models.networks.minimax_h3.infer.offload import MiniMaxH3OffloadTransformerInfer
from lightx2v.models.networks.minimax_h3.infer.post_infer import MiniMaxH3PostInfer
from lightx2v.models.networks.minimax_h3.infer.pre_infer import MiniMaxH3PreInfer
from lightx2v.models.networks.minimax_h3.infer.transformer_infer import MiniMaxH3TransformerInfer
from lightx2v.models.networks.minimax_h3.weights import (
    MiniMaxH3PostWeights,
    MiniMaxH3PreWeights,
    MiniMaxH3TransformerWeights,
)
from lightx2v.models.networks.minimax_h3.weights.tensor_parallel import unwrap_tp_linear
from lightx2v.utils.envs import GET_DTYPE

H3_CHANNEL_QUANT_SCHEMES = {
    "fp8-q8f",
    "fp8-sgl",
    "fp8-torchao",
    "fp8-triton",
    "fp8-vllm",
    "int8-q8f",
    "int8-sgl",
    "int8-torchao",
    "int8-triton",
    "int8-vllm",
}


class MiniMaxH3Model(BaseTransformerModel):
    """LightX2V-native MiniMax-H3 joint audio/video transformer."""

    pre_weight_class = MiniMaxH3PreWeights
    transformer_weight_class = MiniMaxH3TransformerWeights
    post_weight_class = MiniMaxH3PostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0, lora_alpha=None):
        self.lora_alpha = lora_alpha
        if GET_DTYPE() != torch.bfloat16:
            raise ValueError(
                "MiniMax-H3 requires DTYPE=BF16. The native loader preserves the released checkpoint's 626 BF16 tensors and 12 FP32 projection/time/head tensors without dtype conversion."
            )
        if config.get("cfg_parallel", False) or config.get("enable_cfg", False):
            raise ValueError("MiniMax-H3 is guidance-distilled and does not have a CFG/unconditional branch")
        if config.get("dit_quantized", False):
            quant_scheme = config.get("dit_quant_scheme", "Default")
            if quant_scheme not in H3_CHANNEL_QUANT_SCHEMES:
                raise NotImplementedError(f"MiniMax-H3 quantized inference requires a per-output-channel FP8/INT8 scheme; got {quant_scheme!r}. Supported schemes: {sorted(H3_CHANNEL_QUANT_SCHEMES)}")
            if not config.get("dit_quantized_ckpt"):
                raise ValueError("MiniMax-H3 quantized inference requires dit_quantized_ckpt")
        elif config.get("dit_quant_scheme", "Default") != "Default":
            raise ValueError("MiniMax-H3 dit_quant_scheme requires a dit_quantized_ckpt")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") not in {"model", "block"}:
            raise NotImplementedError("MiniMax-H3 supports model and block CPU offload")

        transformer_path = config.get("dit_original_ckpt") or os.path.join(model_path, "transformer")
        super().__init__(transformer_path, config, device, lora_path=lora_path, lora_strength=lora_strength)
        self._validate_tensor_parallel_config()
        if config.get("seq_parallel", False):
            parallel_type = config.get("parallel", {}).get("seq_p_attn_type", "ulysses")
            if parallel_type != "ulysses":
                raise NotImplementedError(f"MiniMax-H3 sequence parallel currently supports Ulysses, got {parallel_type!r}")
            world_size = dist.get_world_size(self.seq_p_group)
            local_heads = int(config.get("num_attention_heads", 56)) // self.tp_size
            if local_heads % world_size:
                raise ValueError(
                    "MiniMax-H3 Ulysses requires TP-local attention heads to be divisible by seq_p_size: "
                    f"global_heads={config.get('num_attention_heads', 56)}, tp_size={self.tp_size}, "
                    f"local_heads={local_heads}, seq_p_size={world_size}"
                )
        self.sensitive_layer = {
            "proj_in",
            "audio_proj_in",
            "time_embedder",
            "proj_out",
            "audio_proj_out",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _apply_weights(self, weight_dict=None):
        if self.config.get("lora_dynamic_apply", False):
            source = weight_dict if weight_dict is not None else self.original_weight_dict
            self._h3_weight_shapes = {key: tuple(tensor.shape) for key, tensor in source.items() if isinstance(tensor, torch.Tensor) and tensor.ndim == 2}
        return super()._apply_weights(weight_dict)

    @staticmethod
    def _normalize_dynamic_lora_key(key):
        for prefix in ("base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
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

    def _validate_dynamic_lora_shapes(self, source, normalized_sources, down_names):
        model_keys = set()
        ranks = set()
        for down_name in sorted(down_names):
            base_name = down_name[: -len(".lora_down.weight")]
            up_name = base_name + ".lora_up.weight"
            model_key = base_name + ".weight"
            down_shape = tuple(source.get_slice(normalized_sources[down_name]).get_shape())
            up_shape = tuple(source.get_slice(normalized_sources[up_name]).get_shape())
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
            normalized_sources = {}
            unsupported = []
            for source_key in source.keys():
                normalized_key = self._normalize_dynamic_lora_key(source_key)
                if normalized_key is None:
                    unsupported.append(source_key)
                    continue
                if normalized_key in normalized_sources:
                    raise ValueError(f"MiniMax-H3 LoRA keys collide after normalization: {source_key} and {normalized_sources[normalized_key]}")
                normalized_sources[normalized_key] = source_key
            if unsupported:
                raise ValueError(f"MiniMax-H3 dynamic LoRA contains {len(unsupported)} unsupported tensors: {unsupported[:4]}")

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
            for normalized_key, source_key in normalized_sources.items():
                tensor = source.get_tensor(source_key).to(GET_DTYPE())
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

    @staticmethod
    def _iter_weight_objects(*roots):
        stack = list(roots)
        visited = set()
        while stack:
            obj = stack.pop()
            if obj is None or id(obj) in visited:
                continue
            visited.add(id(obj))
            yield unwrap_tp_linear(obj)
            stack.extend(getattr(obj, "_modules", {}).values())
            stack.extend(getattr(obj, "_parameters", {}).values())

    def _register_dynamic_lora_weights(self, lora_weights, strength):
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError(f"MiniMax-H3 LoRA strength must be finite, got {strength}")
        self.pre_weight.register_lora(lora_weights, strength)
        self.transformer_weights.register_lora(lora_weights, strength)
        self.post_weight.register_lora(lora_weights, strength)

        weights = list(self._iter_weight_objects(self.pre_weight, self.transformer_weights, self.post_weight))
        for weight in weights:
            if getattr(weight, "pin_weight", None) is None or getattr(weight, "weight", None) is not None:
                continue
            for attr_name in ("lora_down", "lora_up", "lora_alpha", "lora_scale"):
                tensor = getattr(weight, attr_name, None)
                if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                    tensor = tensor.to("cpu")
                    setattr(weight, attr_name, tensor.pin_memory())

        registered = {weight.weight_name for weight in weights if getattr(weight, "has_lora_branch", False)}
        missing = sorted(self._pending_dynamic_lora_model_keys - registered)
        if missing:
            self._remove_lora()
            raise RuntimeError(f"MiniMax-H3 failed to register {len(missing)} dynamic LoRA branches: {missing[:4]}")
        logger.info("Registered {} MiniMax-H3 dynamic LoRA branches with strength={}", len(self._pending_dynamic_lora_model_keys), strength)

    def _register_lora(self, lora_path, strength):
        lora_weights = self._load_lora_file(lora_path)
        self._register_dynamic_lora_weights(lora_weights, strength)
        self.lora_path = lora_path
        self.lora_strength = float(strength)
        offload_manager = getattr(getattr(self, "transformer_infer", None), "offload_manager", None)
        if offload_manager is not None:
            offload_manager.need_init_first_buffer = True

    def _update_lora(self, lora_path, strength, alpha=None):
        if isinstance(lora_path, dict):
            raise NotImplementedError("MiniMax-H3 dynamic LoRA switching expects one checkpoint path, not a merged tensor dictionary")
        lora_weights = self._load_lora_file(lora_path, alpha=alpha)
        self._remove_lora()
        self._register_dynamic_lora_weights(lora_weights, strength)
        self.lora_path = lora_path
        self.lora_strength = float(strength)
        offload_manager = getattr(getattr(self, "transformer_infer", None), "offload_manager", None)
        if offload_manager is not None:
            offload_manager.need_init_first_buffer = True

    def _validate_tensor_parallel_config(self):
        if not self.use_tp:
            return
        checks = {
            "num_attention_heads": int(self.config.get("num_attention_heads", 56)),
            "ffn_hidden_size": int(self.config.get("ffn_hidden_size", 14336)),
            "adaln_output_size": 18 * int(self.config.get("hidden_size", 5376)),
        }
        invalid = {name: value for name, value in checks.items() if value % self.tp_size}
        if invalid:
            details = ", ".join(f"{name}={value}" for name, value in invalid.items())
            raise ValueError(f"MiniMax-H3 TP size {self.tp_size} must divide {details}")

    @staticmethod
    def _tp_split_type(key):
        if ".attn.to_q." in key or ".attn.to_k." in key or ".attn.to_v." in key:
            return "col"
        if ".attn.to_out.0." in key:
            return "row"
        if ".ff.net.0.proj." in key:
            return "ff_fused_col"
        if ".ff.net.2." in key:
            return "row"
        if ".adaln_proj.linear." in key:
            return "col"
        return None

    def _select_tensor_parallel_shard(self, key, tensor):
        """Select this rank's checkpoint shard without materializing peers."""
        if not self.config.get("tensor_parallel", False):
            return tensor
        split_type = self._tp_split_type(key)
        if split_type is None:
            return tensor

        if split_type == "row":
            # Per-output-channel quantization scales remain replicated for a
            # row-parallel weight; only the weight's input dimension is split.
            if key.endswith(".weight_scale") or tensor.ndim < 2:
                return tensor
            split_dim = 1
            if tensor.shape[split_dim] % self.tp_size:
                raise ValueError(f"Cannot row-shard {key} shape {tuple(tensor.shape)} across TP size {self.tp_size}")
            return torch.chunk(tensor, self.tp_size, dim=split_dim)[self.tp_rank].contiguous()

        if split_type == "ff_fused_col":
            if tensor.shape[0] % 2:
                raise ValueError(f"Invalid H3 fused SwiGLU tensor {key} with shape {tuple(tensor.shape)}")
            half = tensor.shape[0] // 2
            if half % self.tp_size:
                raise ValueError(f"Cannot fused-column-shard {key} shape {tuple(tensor.shape)} across TP size {self.tp_size}")
            value, gate = tensor.split(half, dim=0)
            value = torch.chunk(value, self.tp_size, dim=0)[self.tp_rank]
            gate = torch.chunk(gate, self.tp_size, dim=0)[self.tp_rank]
            return torch.cat((value, gate), dim=0).contiguous()

        if tensor.shape[0] % self.tp_size:
            raise ValueError(f"Cannot column-shard {key} shape {tuple(tensor.shape)} across TP size {self.tp_size}")
        return torch.chunk(tensor, self.tp_size, dim=0)[self.tp_rank].contiguous()

    def _should_load_weights(self):
        # Each TP rank reads its own slices.  This is also correct for TP+SP,
        # where the same TP coordinate is replicated in every SP lane.
        if self.use_tp:
            return True
        return super()._should_load_weights()

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if self.use_tp:
            if not is_weight_loader:
                raise RuntimeError("MiniMax-H3 TP expects every rank to load its local checkpoint shards")
            return weight_dict
        return super()._load_weights_from_rank0(weight_dict, is_weight_loader)

    def _load_dummy_ckpt(self, unified_dtype, sensitive_layer):
        weight_dict = super()._load_dummy_ckpt(unified_dtype, sensitive_layer)
        if not self.use_tp:
            return weight_dict
        return {key: self._select_tensor_parallel_shard(key, tensor) for key, tensor in weight_dict.items()}

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        # BaseTransformerModel forces rank-0 TP loading through CPU. H3 shards
        # locally instead, so retain the runner-selected CPU/GPU target.
        if not self.use_tp:
            return super()._load_ckpt(unified_dtype, sensitive_layer)
        load_device = self._checkpoint_load_device()
        logger.info(
            "MiniMax-H3 rank {} (TP rank {}) loading TP checkpoint shards on {}",
            dist.get_rank() if dist.is_initialized() else 0,
            self.tp_rank,
            load_device,
        )
        use_tp = self.use_tp
        self.use_tp = False
        try:
            return super()._load_ckpt(unified_dtype, sensitive_layer)
        finally:
            self.use_tp = use_tp

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        if not self.use_tp:
            return super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        checkpoint_path = self.config["dit_quantized_ckpt"]
        files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))) if os.path.isdir(checkpoint_path) else [checkpoint_path]
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        weight_dict = {}
        load_device = self._checkpoint_load_device()
        logger.info(f"MiniMax-H3 rank {dist.get_rank() if dist.is_initialized() else 0} loading TP checkpoint shards on {load_device}")
        for file_path in files:
            with safe_open(file_path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if any(remove_key in key for remove_key in remove_keys):
                        continue
                    weight_dict[key] = self._load_local_tensor(source, key, load_device)
        self._validate_checkpoint_devices(weight_dict, load_device)
        return weight_dict

    def _checkpoint_load_device(self):
        """Resolve an indexed accelerator device for safetensors."""
        device = torch.device(self.device)
        if device.type == "cpu":
            return "cpu"
        if device.index is not None:
            return str(device)
        device_module = getattr(torch, device.type, None)
        if device_module is None or not hasattr(device_module, "current_device"):
            raise RuntimeError(f"Cannot resolve current MiniMax-H3 checkpoint device from {device}")
        return f"{device.type}:{device_module.current_device()}"

    @staticmethod
    def _validate_checkpoint_devices(weight_dict, load_device):
        expected = torch.device(load_device)
        if expected.type == "cpu":
            return
        misplaced = [key for key, tensor in weight_dict.items() if tensor.device != expected]
        if misplaced:
            preview = ", ".join(misplaced[:4])
            raise RuntimeError(f"MiniMax-H3 checkpoint tensors were not loaded on {expected}: {preview}")

    def _load_local_tensor(self, source, key, load_device):
        """Shard on CPU before copying only this TP rank's tensor to the accelerator."""
        tensor = self._select_tensor_parallel_shard(key, source.get_tensor(key))
        if torch.device(load_device).type != "cpu":
            tensor = tensor.to(load_device)
        return tensor

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        """Load the released mixed-precision tensors without generic dtype coercion."""
        del unified_dtype, sensitive_layer
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"MiniMax-H3 native loading expects the released safetensors checkpoint; got {file_path}")
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        load_device = self._checkpoint_load_device()
        # Reading a full tensor directly on the accelerator and then slicing it
        # can retain the full safetensors storage behind a small TP view.  Shard
        # on CPU first so accelerator memory contains only this rank's weights.
        with safe_open(file_path, framework="pt", device="cpu") as source:
            weight_dict = {
                key: self._load_local_tensor(source, key, load_device)
                for key in source.keys()
                if not any(remove_key in key for remove_key in remove_keys) and (preserve_keys is None or any(preserve_key in key for preserve_key in preserve_keys))
            }
        self._validate_checkpoint_devices(weight_dict, load_device)
        return weight_dict

    def _init_infer_class(self):
        if self.config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("MiniMax-H3 feature caching is not implemented")
        self.pre_infer_class = MiniMaxH3PreInfer
        self.transformer_infer_class = MiniMaxH3OffloadTransformerInfer if self.cpu_offload else MiniMaxH3TransformerInfer
        self.post_infer_class = MiniMaxH3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if not infer_condition:
            raise ValueError("MiniMax-H3 does not execute an unconditional pass")
        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        pre = self.pre_infer.infer(self.pre_weight, prompt_embeds)
        if self.config.get("seq_parallel", False):
            pre = self._seq_parallel_pre_process(pre)
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre)
        if self.config.get("seq_parallel", False):
            hidden_states = self._seq_parallel_post_process(hidden_states, pre)
        return self.post_infer.infer(self.post_weight, hidden_states, pre)

    @torch.no_grad()
    def infer(self, inputs):
        block_offload = self.cpu_offload and self.offload_granularity == "block"
        if block_offload and self.scheduler.step_index == 0:
            self.pre_weight.to_cuda()
            self.post_weight.to_cuda()
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        self.scheduler.video_noise_pred = output.video
        self.scheduler.audio_noise_pred = output.audio
        if block_offload and self.scheduler.step_index == self.scheduler.infer_steps - 1:
            self.pre_weight.to_cpu()
            self.post_weight.to_cpu()

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        total_length = pre_infer_out.hidden_states.shape[0]
        text_length = pre_infer_out.text_indices.numel()
        expected_text_indices = torch.arange(text_length, device=pre_infer_out.text_indices.device)
        if not torch.equal(pre_infer_out.text_indices, expected_text_indices):
            raise ValueError("MiniMax-H3 sequence parallel requires the conditioner rows to be a contiguous prefix")

        # Keep the largest possible conditioner prefix replicated on every
        # rank, and shard the remaining packed sequence without padding. This
        # preserves exact dense-attention semantics for audio/video rows.
        remainder = total_length % world_size
        if text_length < remainder:
            raise ValueError(f"MiniMax-H3 cannot split packed length {total_length} over {world_size} ranks with only {text_length} prefix rows")
        aux_length = text_length - ((text_length - remainder) % world_size)
        main_length = total_length - aux_length
        if main_length <= 0 or main_length % world_size:
            raise ValueError(f"MiniMax-H3 sequence split is invalid: total={total_length}, aux={aux_length}, seq_p_size={world_size}")
        main_shard_length = main_length // world_size
        shard_start = aux_length + rank * main_shard_length
        shard_end = shard_start + main_shard_length

        def shard_sequence(tensor):
            return torch.cat((tensor[:aux_length], tensor[shard_start:shard_end]), dim=0)

        pre_infer_out.sequence_parallel_state = MiniMaxH3SequenceParallelState(
            aux_length=aux_length,
            main_shard_length=main_shard_length,
            timestep_indices=pre_infer_out.timestep_indices,
            adaln_indices=pre_infer_out.adaln_indices,
            rotary_emb=pre_infer_out.rotary_emb,
        )
        pre_infer_out.hidden_states = shard_sequence(pre_infer_out.hidden_states)
        pre_infer_out.timestep_indices = shard_sequence(pre_infer_out.timestep_indices)
        pre_infer_out.adaln_indices = shard_sequence(pre_infer_out.adaln_indices)
        pre_infer_out.rotary_emb = tuple(shard_sequence(tensor) for tensor in pre_infer_out.rotary_emb)
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, output, pre_infer_out):
        state = pre_infer_out.sequence_parallel_state
        if state is None:
            raise RuntimeError("MiniMax-H3 sequence-parallel metadata is missing")
        local_main = output[state.aux_length :]
        if local_main.shape[0] != state.main_shard_length:
            raise RuntimeError(f"MiniMax-H3 local sequence length changed from {state.main_shard_length} to {local_main.shape[0]}")
        world_size = dist.get_world_size(self.seq_p_group)
        gathered = [torch.empty_like(local_main) for _ in range(world_size)]
        dist.all_gather(gathered, local_main.contiguous(), group=self.seq_p_group)
        output = torch.cat((output[: state.aux_length], *gathered), dim=0)
        pre_infer_out.timestep_indices = state.timestep_indices
        pre_infer_out.adaln_indices = state.adaln_indices
        pre_infer_out.rotary_emb = state.rotary_emb
        pre_infer_out.sequence_parallel_state = None
        return output

    def to_cpu(self):
        super().to_cpu()
        if hasattr(self.transformer_infer, "offload_manager"):
            # Full teardown moves the active aliases away from the persistent
            # device buffers. Force buffer 0 to be populated again next run.
            self.transformer_infer.offload_manager.need_init_first_buffer = True
