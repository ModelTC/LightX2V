import glob
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE

from .infer.offload import QwenImage21OffloadTransformerInfer
from .infer.post_infer import QwenImage21PostInfer
from .infer.pre_infer import QwenImage21PreInfer
from .infer.transformer_infer import QwenImage21TransformerInfer
from .weights.post_weights import QwenImage21PostWeights
from .weights.pre_weights import QwenImage21PreWeights
from .weights.transformer_weights import QwenImage21TransformerWeights


class QwenImage21TransformerModel(BaseTransformerModel):
    pre_weight_class = QwenImage21PreWeights
    transformer_weight_class = QwenImage21TransformerWeights
    post_weight_class = QwenImage21PostWeights

    def __init__(self, model_path, config, device):
        if config.get("cpu_offload", False) and config.get("offload_granularity", "model") not in {
            "model",
            "block",
        }:
            raise NotImplementedError("Qwen-Image-2.1 supports model and block CPU offload")
        self.block_offload = config.get("cpu_offload", False) and config.get("offload_granularity", "model") == "block"
        super().__init__(model_path, config, device)
        self._validate_tensor_parallel_config()
        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self.kv_cache_manager = None

    def _load_shared_cpu_weights(self, unified_dtype, sensitive_layer):
        from lightx2v.common.offload.shared_weight_coordinator import coordinate_rank_local_error

        from .shared_block_weights import QwenImage21SharedBlockAdapter

        error = None
        try:
            adapter = QwenImage21SharedBlockAdapter(self, unified_dtype, sensitive_layer)
            private = adapter.load_private_weights()
        except Exception as exc:
            error = exc
        coordinate_rank_local_error("Qwen-Image-2.1 checkpoint preflight", error)
        return adapter.materialize(private)

    def _validate_tensor_parallel_config(self):
        if not self.use_tp:
            return
        hidden_size = int(self.config["num_attention_heads"]) * int(self.config["attention_head_dim"])
        checks = {
            "num_attention_heads": int(self.config["num_attention_heads"]),
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * int(self.config["mlp_ratio"]),
        }
        for name, value in checks.items():
            if value % self.tp_size:
                raise ValueError(f"Qwen-Image-2.1 TP size {self.tp_size} must divide {name}={value}")

    @staticmethod
    def _tp_split_type(key):
        if any(f".{name}." in key for name in ("attn.to_q", "attn.to_k", "attn.to_v", "img_mlp.proj", "img_mlp.gate_layer")):
            return "col"
        if any(f".{name}." in key for name in ("attn.to_out.0", "img_mlp.out")):
            return "row"
        return None

    def _tensor_parallel_split_dim(self, key, tensor):
        split_type = self._tp_split_type(key)
        if not self.config.get("tensor_parallel", False) or split_type is None or tensor.ndim == 0:
            return None
        if split_type == "row":
            # Row parallelism splits input channels; output-channel scales remain replicated.
            if tensor.ndim < 2 or key.endswith(".weight_scale"):
                return None
            split_dim = 1
        else:
            split_dim = 0
        if tensor.shape[split_dim] % self.tp_size:
            raise ValueError(f"Cannot {split_type}-shard {key} shape {tuple(tensor.shape)} across TP size {self.tp_size}")
        return split_dim

    def _select_tensor_parallel_shard(self, key, tensor):
        split_dim = self._tensor_parallel_split_dim(key, tensor)
        if split_dim is None:
            return tensor
        return torch.chunk(tensor, self.tp_size, dim=split_dim)[self.tp_rank].contiguous()

    def _should_load_weights(self):
        # Every rank reads its own TP shard. This also covers repeated TP
        # coordinates in different SP lanes without cross-mesh broadcasts.
        if self.use_tp:
            return True
        return super()._should_load_weights()

    def _load_dummy_ckpt(self, unified_dtype, sensitive_layer):
        weight_dict = super()._load_dummy_ckpt(unified_dtype, sensitive_layer)
        if not self.use_tp:
            return weight_dict
        return {key: self._select_tensor_parallel_shard(key, tensor) for key, tensor in weight_dict.items()}

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        if self.use_tp:
            if not is_weight_loader:
                raise RuntimeError("Qwen-Image-2.1 TP expects every rank to load its local checkpoint shards")
            return weight_dict
        return super()._load_weights_from_rank0(weight_dict, is_weight_loader)

    def _checkpoint_load_device(self):
        device = torch.device(self.device)
        if device.type == "cpu":
            return device
        if device.index is not None:
            return device
        device_module = getattr(torch, device.type, None)
        if device_module is None or not hasattr(device_module, "current_device"):
            raise RuntimeError(f"Cannot resolve current Qwen-Image-2.1 checkpoint device from {device}")
        return torch.device(device.type, device_module.current_device())

    @staticmethod
    def _validate_checkpoint_devices(weight_dict, load_device):
        if load_device.type == "cpu":
            return
        misplaced = [key for key, tensor in weight_dict.items() if tensor.device != load_device]
        if misplaced:
            preview = ", ".join(misplaced[:4])
            raise RuntimeError(f"Qwen-Image-2.1 checkpoint tensors were not loaded on {load_device}: {preview}")

    @staticmethod
    def _checkpoint_dtype(key, tensor, unified_dtype, sensitive_layer, quantized):
        if not tensor.is_floating_point():
            return None
        if quantized and tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return None
        return GET_DTYPE() if unified_dtype or all(pattern not in key for pattern in sensitive_layer) else GET_SENSITIVE_DTYPE()

    def _load_local_tensor(self, source, key, load_device, unified_dtype, sensitive_layer, quantized):
        tensor = self._select_tensor_parallel_shard(key, source.get_tensor(key))
        target_dtype = self._checkpoint_dtype(key, tensor, unified_dtype, sensitive_layer, quantized)
        if target_dtype is not None:
            tensor = tensor.to(target_dtype)
        if load_device.type != "cpu":
            tensor = tensor.to(load_device)
        return tensor

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if not self.use_tp:
            return super()._load_ckpt(unified_dtype, sensitive_layer)
        load_device = self._checkpoint_load_device()
        logger.info("Qwen-Image-2.1 rank {} (TP rank {}) loading local checkpoint shards on {}", dist.get_rank(), self.tp_rank, load_device)
        # The base loader forces TP through CPU for rank-0 distribution.
        # Local sharding below keeps the runner-selected load device instead.
        use_tp = self.use_tp
        self.use_tp = False
        try:
            weight_dict = super()._load_ckpt(unified_dtype, sensitive_layer)
        finally:
            self.use_tp = use_tp
        self._validate_checkpoint_devices(weight_dict, load_device)
        return weight_dict

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        if not self.config.get("tensor_parallel", False):
            return super()._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
        if os.path.splitext(file_path)[-1] != ".safetensors":
            raise ValueError(f"Qwen-Image-2.1 TP checkpoint loading requires safetensors; got {file_path}")
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        load_device = self._checkpoint_load_device()
        with safe_open(file_path, framework="pt", device="cpu") as source:
            return {
                key: self._load_local_tensor(source, key, load_device, unified_dtype, sensitive_layer, quantized=False)
                for key in source.keys()
                if not any(remove_key in key for remove_key in remove_keys) and (preserve_keys is None or any(preserve_key in key for preserve_key in preserve_keys))
            }

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        if not self.use_tp:
            return super()._load_quant_ckpt(unified_dtype, sensitive_layer)
        checkpoint_path = self.config["dit_quantized_ckpt"]
        files = sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))) if os.path.isdir(checkpoint_path) else [checkpoint_path]
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        load_device = self._checkpoint_load_device()
        logger.info("Qwen-Image-2.1 rank {} (TP rank {}) loading local quantized checkpoint shards on {}", dist.get_rank(), self.tp_rank, load_device)
        weight_dict = {}
        for file_path in files:
            with safe_open(file_path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if any(remove_key in key for remove_key in remove_keys):
                        continue
                    weight_dict[key] = self._load_local_tensor(source, key, load_device, unified_dtype, sensitive_layer, quantized=True)
        self._validate_checkpoint_devices(weight_dict, load_device)
        return weight_dict

    def _init_infer_class(self):
        self.pre_infer_class = QwenImage21PreInfer
        self.transformer_infer_class = QwenImage21OffloadTransformerInfer if self.cpu_offload else QwenImage21TransformerInfer
        self.post_infer_class = QwenImage21PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class()
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class()
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def _seq_parallel_pre_process(self, state):
        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        target_length = state.hidden_states.shape[0]
        if target_length % world_size:
            raise ValueError(f"Qwen-Image-2.1 target token count ({target_length}) must be divisible by seq_p_size ({world_size}); choose another output size or SP size")
        local_length = target_length // world_size
        begin = rank * local_length
        end = begin + local_length
        state.hidden_states = state.hidden_states[begin:end].contiguous()
        if state.rotary_positions is None:
            state.rotary = state.rotary[begin:end].contiguous()
        else:
            # FlashInfer keeps the full lookup table and indexes the rank-local
            # target positions from it.
            state.rotary_positions = state.rotary_positions[begin:end].contiguous()
        return state

    def _seq_parallel_post_process(self, hidden, state):
        target_length = state.layout.target_len
        world_size = dist.get_world_size(self.seq_p_group)
        expected_local_length = target_length // world_size
        if hidden.shape[0] != expected_local_length:
            raise RuntimeError(f"Qwen-Image-2.1 local target length changed from {expected_local_length} to {hidden.shape[0]}")
        gathered = [torch.empty_like(hidden) for _ in range(world_size)]
        dist.all_gather(gathered, hidden.contiguous(), group=self.seq_p_group)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def prefill_condition_kv(self, inputs):
        """Create and fill condition caches owned by this model for one request."""
        self.clear_condition_kv()
        cache_config = {
            "num_layers": self.config["num_layers"],
            "num_heads": self.config["num_attention_heads"] // self.tp_size,
            "dim": self.config["num_attention_heads"] * self.config["attention_head_dim"] // self.tp_size,
        }
        cache_device = inputs["cond"]["prompt_embeds"].device
        self.kv_cache_manager = KVCacheManager(cache_config, device=cache_device)
        try:
            for name in ("cond", "uncond"):
                if name in inputs:
                    branch = inputs[name]
                    cache = self.kv_cache_manager.create_self_attn_kv_cache(name, branch["layout"].prefix_len, kv_cache_scheme="static", step_kv_cache=False)
                    state = self.pre_infer.infer_condition(self.pre_weight, branch["prompt_embeds"], inputs.get("image_latents"), branch["layout"])
                    self.transformer_infer.prefill(self.transformer_weights, state, cache)
        except Exception:
            self.clear_condition_kv()
            raise

    def clear_condition_kv(self):
        if self.kv_cache_manager is not None:
            for cache in self.kv_cache_manager.self_attn_kv_caches.values():
                cache.reset()
        self.kv_cache_manager = None

    def _infer_cond_uncond(self, inputs, infer_condition=True):
        name = "cond" if infer_condition else "uncond"
        branch = inputs[name]
        cache = self.kv_cache_manager.get_self_attn_kv_cache(name)
        state = self.pre_infer.infer_target(self.pre_weight, self.scheduler.latents[0], branch["layout"])
        if self.config["seq_parallel"]:
            state = self._seq_parallel_pre_process(state)
        hidden = self.transformer_infer.infer(self.transformer_weights, state, cache)
        if self.config["seq_parallel"]:
            hidden = self._seq_parallel_post_process(hidden, state)
        noise = self.post_infer.infer(self.post_weight, hidden, state)
        return noise.unsqueeze(0)

    @torch.no_grad()
    def infer(self, inputs):
        if self.kv_cache_manager is None:
            raise RuntimeError("Condition KV must be prefilled before denoising")
        if self.config["enable_cfg"]:
            positive = self._infer_cond_uncond(inputs, True)
            negative = self._infer_cond_uncond(inputs, False)
            self.scheduler.noise_pred = negative + self.scheduler.sample_guide_scale * (positive - negative)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs)
