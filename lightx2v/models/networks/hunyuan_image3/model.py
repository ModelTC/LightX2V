import os
import re

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3TaylorCache
from lightx2v.models.networks.hunyuan_image3.infer.module_io import HunyuanImage3SequenceParallelState
from lightx2v.models.networks.hunyuan_image3.infer.post_infer import HunyuanImage3PostInfer
from lightx2v.models.networks.hunyuan_image3.infer.pre_infer import HunyuanImage3PreInfer
from lightx2v.models.networks.hunyuan_image3.infer.transformer_infer import HunyuanImage3TransformerInfer
from lightx2v.models.networks.hunyuan_image3.weights.post_weights import HunyuanImage3PostWeights
from lightx2v.models.networks.hunyuan_image3.weights.pre_weights import HunyuanImage3PreWeights
from lightx2v.models.networks.hunyuan_image3.weights.transformer_weights import HunyuanImage3TransformerWeights
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


def _normalize_device_name(device):
    if isinstance(device, int):
        return f"cuda:{device}"
    device = str(device)
    if device.isdigit():
        return f"cuda:{device}"
    if device == "cuda":
        return f"cuda:{torch.cuda.current_device()}"
    return device


def _resolve_sequence_parallel_pipeline_lane(config, devices):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HunyuanImage3 sequence-parallel pipeline split requires torch.distributed initialization.")

    # In pure SP, global rank/world are identical to seq rank/world.  In a
    # CFG+SP mesh, however, using seq rank would make both CFG rows select the
    # same physical pipeline devices.  Allocate one disjoint lane per global
    # process instead; collectives themselves still use the seq_p group.
    parallel_world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if len(devices) % parallel_world_size:
        raise ValueError(f"HunyuanImage3 sequence parallel requires pipeline device count ({len(devices)}) to be divisible by cfg_p_size * seq_p_size ({parallel_world_size}).")

    parallel = config.get("parallel") or {}
    layout = str(parallel.get("pipeline_layout", config.get("hunyuan_image3_pipeline_layout", "interleaved"))).strip().lower()
    if layout != "interleaved":
        raise ValueError(f"HunyuanImage3 sequence parallel uses Wan-style rank/device initialization and currently requires parallel.pipeline_layout='interleaved'; got {layout!r}.")

    lane = devices[global_rank::parallel_world_size]
    expected_device = f"cuda:{torch.cuda.current_device()}"
    if lane[0] != expected_device:
        raise RuntimeError(f"HunyuanImage3 interleaved pipeline lane must start on the CUDA device selected by init_parallel_env: lane={lane}, current_device={expected_device}.")
    return lane


def resolve_pipeline_devices(config, fallback_device):
    parallel = config.get("parallel") or {}
    configured = parallel.get("pipeline_devices") or config.get("pipeline_parallel_devices") or config.get("hunyuan_image3_pipeline_devices")
    if configured:
        if isinstance(configured, str):
            devices = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            devices = list(configured)
        devices = [_normalize_device_name(device) for device in devices]
        if config.get("seq_parallel", False):
            return _resolve_sequence_parallel_pipeline_lane(config, devices)
        return devices

    pipeline_parallel = parallel.get("pipeline_parallel", config.get("pipeline_parallel", True))
    if pipeline_parallel and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            if config.get("seq_parallel", False):
                devices = [f"cuda:{idx}" for idx in range(device_count)]
                return _resolve_sequence_parallel_pipeline_lane(config, devices)
            if config.get("cfg_parallel", False):
                cfg_p_size = int(config.get("parallel", {}).get("cfg_p_size", 1))
                if cfg_p_size > 1:
                    if not dist.is_available() or not dist.is_initialized():
                        raise RuntimeError("HunyuanImage3 cfg_parallel pipeline split requires an initialized distributed process group.")
                    if device_count % cfg_p_size != 0:
                        raise ValueError(f"HunyuanImage3 cfg_parallel requires visible CUDA device count ({device_count}) to be divisible by cfg_p_size ({cfg_p_size}).")
                    cfg_p_group = config["device_mesh"].get_group(mesh_dim="cfg_p")
                    cfg_p_rank = dist.get_rank(cfg_p_group)
                    devices_per_cfg_rank = device_count // cfg_p_size
                    start = cfg_p_rank * devices_per_cfg_rank
                    stop = start + devices_per_cfg_rank
                    return [f"cuda:{idx}" for idx in range(start, stop)]
            return [f"cuda:{idx}" for idx in range(device_count)]

    return [_normalize_device_name(fallback_device)]


def resolve_pipeline_device_for_key(key, config, devices):
    devices = [_normalize_device_name(device) for device in devices]
    if len(devices) == 1:
        return devices[0]

    layer_match = re.match(r"model\.layers\.(\d+)\.", key)
    if layer_match is not None:
        layer_idx = int(layer_match.group(1))
        num_layers = int(config.get("num_layers") or config["num_hidden_layers"])
        stage_idx = min(layer_idx * len(devices) // num_layers, len(devices) - 1)
        return devices[stage_idx]

    if key.startswith(("model.ln_f.", "lm_head.", "final_layer.", "time_embed_2.")):
        return devices[-1]

    return devices[0]


class HunyuanImage3Model(BaseTransformerModel):
    pre_weight_class = HunyuanImage3PreWeights
    transformer_weight_class = HunyuanImage3TransformerWeights
    post_weight_class = HunyuanImage3PostWeights

    def __init__(self, model_path, config, device, lora_path=None, lora_strength=1.0):
        super().__init__(model_path, config, device, "hunyuan_image3", lora_path, lora_strength)
        self._init_tensor_parallel()
        self.pipeline_devices = resolve_pipeline_devices(config, device)
        self.pipeline_parallel = len(set(self.pipeline_devices)) > 1
        self.sequence_parallel_attn_type = self._resolve_sequence_parallel_attn_type()
        self._sp_gather_buffers = {}
        self._validate_tensor_parallel_config()
        self._validate_sequence_parallel_config()
        if self.lazy_load:
            self.remove_keys.extend(["model.layers."])
        self.preserved_keys = [
            "model.",
            "lm_head.",
            "patch_embed.",
            "final_layer.",
            "time_embed.",
            "time_embed_2.",
            "timestep_emb.",
            "guidance_emb.",
            "timestep_r_emb.",
        ]
        self.sensitive_layer = {
            "norm",
            "layernorm",
            "time_embed",
            "timestep_emb",
            "guidance_emb",
            "timestep_r_emb",
            "gate.wg",
        }
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

        if self.config.get("seq_parallel", False) and self.tensor_parallel:
            logger.info(
                "HunyuanImage3 tensor + sequence parallel initialized: "
                f"tp_rank={self.tp_rank}, "
                f"tp_size={self.tp_size}, "
                f"sp_rank={dist.get_rank(self.seq_p_group)}, "
                f"sp_size={dist.get_world_size(self.seq_p_group)}, "
                f"attention={self.sequence_parallel_attn_type}, "
                f"device={self.pipeline_devices[0]}"
            )
        elif self.config.get("seq_parallel", False):
            logger.info(
                "HunyuanImage3 sequence parallel initialized: "
                f"rank={dist.get_rank(self.seq_p_group)}, "
                f"size={dist.get_world_size(self.seq_p_group)}, "
                f"attention={self.sequence_parallel_attn_type}, "
                f"pipeline_devices={self.pipeline_devices}"
            )
        elif self.tensor_parallel:
            logger.info(f"HunyuanImage3 tensor parallel initialized: rank={self.tp_rank}, size={self.tp_size}, device={self.pipeline_devices[0]}")

    def _init_tensor_parallel(self):
        self.tensor_parallel = bool(self.config.get("tensor_parallel", False))
        if self.tensor_parallel:
            self.tp_group = self.config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_rank = dist.get_rank(self.tp_group)
            self.tp_size = dist.get_world_size(self.tp_group)
        else:
            self.tp_group = None
            self.tp_rank = 0
            self.tp_size = 1

    @staticmethod
    def _iter_config_ints(value):
        if isinstance(value, list):
            return [int(item) for item in value]
        if value is None:
            return []
        return [int(value)]

    def _validate_tensor_parallel_config(self):
        if not self.tensor_parallel:
            return
        if self.pipeline_parallel:
            raise ValueError("HunyuanImage3 tensor parallel cannot be combined with pipeline parallel; set parallel.pipeline_parallel=false.")
        if self.config.get("cpu_offload", False):
            raise NotImplementedError("HunyuanImage3 tensor parallel does not support cpu_offload.")
        if self.config.get("lazy_load", False):
            raise NotImplementedError("HunyuanImage3 tensor parallel does not support lazy_load.")
        if self.config.get("dit_quantized", False):
            raise NotImplementedError("HunyuanImage3 tensor parallel currently supports the unquantized checkpoint only.")
        if self.config.get("load_from_rank0", False):
            raise NotImplementedError("HunyuanImage3 tensor parallel loads and shards safetensors locally on each rank; set load_from_rank0=false.")
        if self.lora_path is not None:
            raise NotImplementedError("HunyuanImage3 tensor parallel does not support LoRA weight loading yet.")
        moe_impl = str(self.config.get("moe_impl", "eager")).strip().lower()
        if moe_impl not in ("eager", "flashinfer"):
            raise NotImplementedError("HunyuanImage3 tensor parallel supports moe_impl='eager' or 'flashinfer'.")

        divisibility_checks = {
            "num_attention_heads": [int(self.config.get("num_attention_heads") or self.config["num_heads"])],
            "num_key_value_heads": [int(self.config.get("num_key_value_heads") or self.config.get("num_attention_heads") or self.config["num_heads"])],
            "intermediate_size": self._iter_config_ints(self.config.get("intermediate_size")),
            "moe_intermediate_size": self._iter_config_ints(self.config.get("moe_intermediate_size")),
            "vocab_size": self._iter_config_ints(self.config.get("vocab_size")),
        }
        shared_experts = self._iter_config_ints(self.config.get("num_shared_expert"))
        moe_intermediate = self._iter_config_ints(self.config.get("moe_intermediate_size"))
        if shared_experts and moe_intermediate:
            if len(shared_experts) == 1:
                shared_experts *= len(moe_intermediate)
            if len(moe_intermediate) == 1:
                moe_intermediate *= len(shared_experts)
            divisibility_checks["shared_mlp_intermediate_size"] = [experts * intermediate for experts, intermediate in zip(shared_experts, moe_intermediate)]

        for name, values in divisibility_checks.items():
            invalid = sorted({value for value in values if value % self.tp_size})
            if invalid:
                raise ValueError(f"HunyuanImage3 TP size {self.tp_size} must divide every {name}; invalid values: {invalid}.")

    @staticmethod
    def _tp_split_type(key):
        if not key.startswith("model.layers.") and not key.startswith("lm_head."):
            return None
        if ".self_attn.qkv_proj." in key:
            return "qkv_col"
        if ".self_attn.o_proj." in key:
            return "row"
        if ".gate_and_up_proj." in key:
            return "gate_up_col"
        if ".down_proj." in key:
            return "row"
        if key.startswith("lm_head."):
            return "col"
        return None

    def _select_tensor_parallel_shard(self, key, tensor):
        split_type = self._tp_split_type(key)
        if split_type is None:
            return tensor

        if split_type == "row":
            # Row-parallel biases are replicated and added after the reduction.
            if tensor.ndim == 1:
                return tensor
            if tensor.ndim != 2 or tensor.shape[1] % self.tp_size:
                raise ValueError(f"Cannot row-shard HunyuanImage3 tensor {key} with shape {tuple(tensor.shape)} across TP size {self.tp_size}.")
            return torch.chunk(tensor, self.tp_size, dim=1)[self.tp_rank].contiguous()

        if split_type == "gate_up_col":
            if tensor.shape[0] % 2:
                raise ValueError(f"HunyuanImage3 fused gate/up tensor {key} has an odd output dimension: {tuple(tensor.shape)}.")
            gate, up = tensor.chunk(2, dim=0)
            if gate.shape[0] % self.tp_size:
                raise ValueError(f"Cannot shard HunyuanImage3 fused gate/up tensor {key} with shape {tuple(tensor.shape)} across TP size {self.tp_size}.")
            gate_shard = torch.chunk(gate, self.tp_size, dim=0)[self.tp_rank]
            up_shard = torch.chunk(up, self.tp_size, dim=0)[self.tp_rank]
            return torch.cat((gate_shard, up_shard), dim=0).contiguous()

        if tensor.shape[0] % self.tp_size:
            raise ValueError(f"Cannot column-shard HunyuanImage3 tensor {key} with shape {tuple(tensor.shape)} across TP size {self.tp_size}.")
        return torch.chunk(tensor, self.tp_size, dim=0)[self.tp_rank].contiguous()

    def _resolve_sequence_parallel_attn_type(self):
        if not self.config.get("seq_parallel", False):
            return None
        parallel = self.config.get("parallel") or {}
        attn_type = str(parallel.get("seq_p_attn_type", "kv_all_gather")).strip().lower().replace("-", "_")
        aliases = {
            "kv_allgather": "kv_all_gather",
            "kv_gather": "kv_all_gather",
            "ulysses_sp": "ulysses",
        }
        attn_type = aliases.get(attn_type, attn_type)
        if attn_type not in ("kv_all_gather", "ulysses"):
            raise ValueError(f"HunyuanImage3 parallel.seq_p_attn_type must be one of: kv_all_gather, ulysses; got {attn_type!r}.")
        return attn_type

    def _validate_sequence_parallel_config(self):
        if not self.config.get("seq_parallel", False):
            return
        if self.seq_p_group is None:
            raise RuntimeError("HunyuanImage3 sequence parallel requires an initialized seq_p process group.")
        parallel = self.config.get("parallel") or {}
        cfg_mode = str(parallel.get("cfg_mode", self.config.get("hunyuan_cfg_mode", "batch"))).strip().lower()
        if self.config.get("cfg_parallel", False):
            if not self.config.get("enable_cfg", False):
                raise ValueError("HunyuanImage3 cfg_parallel requires enable_cfg=true.")
            if cfg_mode != "parallel":
                raise ValueError("HunyuanImage3 CFG+SP requires parallel.cfg_mode='parallel'.")
        elif self.config.get("enable_cfg", False) and cfg_mode != "serial":
            raise ValueError("HunyuanImage3 sequence parallel requires parallel.cfg_mode='serial' so every transformer forward has batch size 1.")
        if self.config.get("use_taylor_cache", False) and self.config.get("enable_kv_cache", False):
            raise ValueError("HunyuanImage3 sequence parallel does not support enabling Taylor cache and KV cache together.")
        if self.sequence_parallel_attn_type == "ulysses":
            world_size = dist.get_world_size(self.seq_p_group)
            global_q_heads = int(self.config.get("num_attention_heads") or self.config["num_heads"])
            global_kv_heads = int(self.config.get("num_key_value_heads") or global_q_heads)
            local_q_heads = global_q_heads // self.tp_size
            local_kv_heads = global_kv_heads // self.tp_size
            if local_q_heads % world_size or local_kv_heads % world_size:
                raise ValueError(
                    "HunyuanImage3 Ulysses requires seq_p_size to divide TP-local Q and KV heads: "
                    f"global_Q={global_q_heads}, global_KV={global_kv_heads}, tp_size={self.tp_size}, "
                    f"local_Q={local_q_heads}, local_KV={local_kv_heads}, seq_p_size={world_size}."
                )

    def _tensor_target_device(self, key):
        return resolve_pipeline_device_for_key(key, self.config, self.pipeline_devices)

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        ext = os.path.splitext(file_path)[-1]
        if ext in (".pt", ".pth", ".tar"):
            if self.tensor_parallel:
                raise NotImplementedError("HunyuanImage3 tensor parallel requires safetensors checkpoints.")
            return super()._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)

        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []
        preserve_keys = self.preserved_keys if hasattr(self, "preserved_keys") else None
        weight_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(remove_key in key for remove_key in remove_keys):
                    continue
                if preserve_keys is not None and not any(preserve_key in key for preserve_key in preserve_keys):
                    continue

                tensor = f.get_tensor(key)
                if self.tensor_parallel:
                    tensor = self._select_tensor_parallel_shard(key, tensor)
                if tensor.dtype.is_floating_point:
                    dtype = GET_DTYPE() if unified_dtype or all(s not in key for s in sensitive_layer) else GET_SENSITIVE_DTYPE()
                else:
                    dtype = tensor.dtype
                target_device = self._tensor_target_device(key)
                weight_dict[key] = tensor.to(device=target_device, dtype=dtype)

        if self.pipeline_parallel:
            logger.debug(f"HunyuanImage3 loaded {len(weight_dict)} tensors from {file_path} across {self.pipeline_devices}")
        return weight_dict

    def _init_infer_class(self):
        if self.config["feature_caching"] != "NoCaching":
            raise NotImplementedError("HunyuanImage3 native feature caching is not implemented yet.")
        self.pre_infer_class = HunyuanImage3PreInfer
        self.transformer_infer_class = HunyuanImage3TransformerInfer
        self.post_infer_class = HunyuanImage3PostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        self.reset_taylor_cache()
        if hasattr(self.transformer_infer, "offload_manager"):
            self._init_offload_manager()

    def reset_taylor_cache(self):
        self.taylor_cache = None
        self.taylor_counter = 0
        self.taylor_last_full_computation_step = 0

    def _should_taylor_full_compute(self, cache_dic):
        current_step = int(cache_dic["current_step"])
        taylor_counter = getattr(self, "taylor_counter", 0)
        return (
            current_step == 0
            or taylor_counter == int(cache_dic["cache_interval"]) - 1
            or (cache_dic["enable_first_enhance"] and current_step < int(cache_dic["first_enhance_steps"]))
            or (cache_dic["enable_tailing_enhance"] and current_step >= int(cache_dic["num_steps"]) - int(cache_dic["tailing_enhance_steps"]))
        )

    def _infer_transformer(self, pre_infer_out):
        hidden_states = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)
        if self.config["seq_parallel"]:
            hidden_states = self._seq_parallel_post_process(hidden_states, pre_infer_out.sequence_parallel_state)
        return hidden_states

    def _infer_transformer_with_taylor_cache(self, pre_infer_out, cache_dic):
        if not hasattr(self, "taylor_counter"):
            self.reset_taylor_cache()
        if self.taylor_cache is None:
            self.taylor_cache = HunyuanImage3TaylorCache(cache_dic["max_order"])

        current_step = int(cache_dic["current_step"])
        if self._should_taylor_full_compute(cache_dic):
            self.taylor_counter = 0
            hidden_states = self._infer_transformer(pre_infer_out)
            if not (cache_dic["enable_first_enhance"] and current_step < int(cache_dic["first_enhance_steps"]) - 1):
                self.taylor_cache.derivatives_computation(
                    hidden_states,
                    distance=current_step - self.taylor_last_full_computation_step,
                    low_freqs_order=cache_dic["low_freqs_order"],
                    high_freqs_order=cache_dic["high_freqs_order"],
                )
            self.taylor_last_full_computation_step = current_step
            self.taylor_cache.last_past_key_values = pre_infer_out.past_key_values
        else:
            self.taylor_counter += 1
            hidden_states = self.taylor_cache.taylor_formula(distance=self.taylor_counter)

        if current_step == int(cache_dic["num_steps"]) - 1:
            self.taylor_cache.clear_derivatives()
        return hidden_states

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        if hasattr(self, "scheduler"):
            self.scheduler.infer_condition = infer_condition
        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)
        # in default setting seq_parallel is False
        if self.config["seq_parallel"]:
            pre_infer_out = self._seq_parallel_pre_process(pre_infer_out)
        if inputs.get("cache_dic") is not None:
            hidden_states = self._infer_transformer_with_taylor_cache(pre_infer_out, inputs["cache_dic"])
        else:
            hidden_states = self._infer_transformer(pre_infer_out)
        # for seq_parallel, the _seq_parallel_post_process function is run in _infer_transformer
        return self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        if pre_infer_out.hidden_states.shape[0] != 1:
            raise ValueError(
                "HunyuanImage3 sequence parallel expects batch size 1 per transformer forward; use parallel.cfg_mode='serial' when cfg_p_size=1 (including TP+SP), or 'parallel' for CFG+SP."
            )

        world_size = dist.get_world_size(self.seq_p_group)
        rank = dist.get_rank(self.seq_p_group)
        original_seq_len = int(pre_infer_out.hidden_states.shape[1])
        padding_size = (-original_seq_len) % world_size
        padded_seq_len = original_seq_len + padding_size
        local_seq_len = padded_seq_len // world_size
        local_start = rank * local_seq_len
        valid_local_seq_len = max(0, min(local_seq_len, original_seq_len - local_start))

        global_position_ids = pre_infer_out.position_ids
        global_attention_mask = pre_infer_out.attention_mask if self.sequence_parallel_attn_type == "ulysses" else None

        pre_infer_out.hidden_states = self._pad_and_slice_sequence(pre_infer_out.hidden_states, 1, padding_size, local_start, local_seq_len, value=0)
        pre_infer_out.position_ids = self._pad_and_slice_sequence(pre_infer_out.position_ids, 1, padding_size, local_start, local_seq_len, value=0)
        if pre_infer_out.custom_pos_emb is not None:
            cos, sin = pre_infer_out.custom_pos_emb
            pre_infer_out.custom_pos_emb = (
                self._pad_and_slice_sequence(cos, 1, padding_size, local_start, local_seq_len, value=1),
                self._pad_and_slice_sequence(sin, 1, padding_size, local_start, local_seq_len, value=0),
            )
        if pre_infer_out.attention_mask is not None:
            local_attention_mask = self._pad_and_slice_sequence(
                pre_infer_out.attention_mask,
                -2,
                padding_size,
                local_start,
                local_seq_len,
                value=False,
            )
            if valid_local_seq_len < local_seq_len:
                local_attention_mask = local_attention_mask.clone()
                local_attention_mask[..., valid_local_seq_len:, 0] = True
            pre_infer_out.attention_mask = local_attention_mask

        pre_infer_out.sequence_parallel_state = HunyuanImage3SequenceParallelState(
            attn_type=self.sequence_parallel_attn_type,
            original_seq_len=original_seq_len,
            padded_seq_len=padded_seq_len,
            local_seq_len=local_seq_len,
            local_start=local_start,
            valid_local_seq_len=valid_local_seq_len,
            global_position_ids=global_position_ids,
            global_attention_mask=global_attention_mask,
        )
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x, sequence_parallel_state):
        if sequence_parallel_state is None:
            raise RuntimeError("HunyuanImage3 sequence parallel post-process is missing sequence metadata.")
        world_size = dist.get_world_size(self.seq_p_group)
        local = x.transpose(0, 1).contiguous()
        output_shape = (local.shape[0] * world_size, *local.shape[1:])
        key = ("hidden", local.device, local.dtype, output_shape)
        gathered = self._sp_gather_buffers.get(key)
        if gathered is None or gathered.shape != output_shape:
            gathered = torch.empty(output_shape, device=local.device, dtype=local.dtype)
            self._sp_gather_buffers[key] = gathered
        dist.all_gather_into_tensor(gathered, local, group=self.seq_p_group)
        return gathered[: sequence_parallel_state.original_seq_len].transpose(0, 1).contiguous()

    @staticmethod
    def _pad_and_slice_sequence(tensor, sequence_dim, padding_size, start, length, value):
        if tensor is None:
            return None
        sequence_dim %= tensor.ndim
        if padding_size:
            pad_shape = list(tensor.shape)
            pad_shape[sequence_dim] = padding_size
            padding = tensor.new_full(pad_shape, value)
            tensor = torch.cat((tensor, padding), dim=sequence_dim)
        return tensor.narrow(sequence_dim, start, length).contiguous()

    def _guidance_scale(self):
        return float(self.config.get("sample_guide_scale", self.config.get("diff_guidance_scale", 1.0)))

    def _set_cfg_scheduler_predictions(self, noise_pred_cond, noise_pred_uncond, noise_pred_guided):
        if not hasattr(self, "scheduler"):
            return
        self.scheduler.noise_pred_cond = noise_pred_cond
        self.scheduler.noise_pred_uncond = noise_pred_uncond
        self.scheduler.noise_pred_guided = noise_pred_guided
        self.scheduler.noise_pred = noise_pred_guided

    def combine_cfg_predictions(self, noise_pred_cond, noise_pred_uncond):
        return noise_pred_uncond + self._guidance_scale() * (noise_pred_cond - noise_pred_uncond)

    @torch.no_grad()
    def infer_branch(self, inputs, infer_condition=True):
        output = self._infer_cond_uncond(inputs, infer_condition=infer_condition)
        if hasattr(self, "scheduler") and "diffusion_prediction" in output:
            self.scheduler.noise_pred = output["diffusion_prediction"]
        return output

    @torch.no_grad()
    def infer_cfg_serial(self, cond_inputs, uncond_inputs):
        cond_output = self._infer_cond_uncond(cond_inputs, infer_condition=True)
        uncond_output = self._infer_cond_uncond(uncond_inputs, infer_condition=False)
        noise_pred_cond = cond_output["diffusion_prediction"]
        noise_pred_uncond = uncond_output["diffusion_prediction"]
        noise_pred_guided = self.combine_cfg_predictions(noise_pred_cond, noise_pred_uncond)
        output = dict(cond_output)
        output["diffusion_prediction"] = noise_pred_guided
        self._set_cfg_scheduler_predictions(noise_pred_cond, noise_pred_uncond, noise_pred_guided)
        return output

    @torch.no_grad()
    def infer(self, inputs):
        cfg_parallel_branch = bool(inputs.get("_cfg_parallel_branch", False))
        infer_condition = True
        cfg_p_group = None
        # 当当前 forward 属于 CFG parallel 时，找到当前进程对应的 CFG 通信组，并判断它负责 conditional 还是 unconditional 分支。
        if cfg_parallel_branch:
            if not self.config.get("cfg_parallel", False):
                raise RuntimeError("HunyuanImage3 received a cfg-parallel branch input, but config['cfg_parallel'] is not enabled.")
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
            infer_condition = dist.get_rank(cfg_p_group) == 0

        output = self._infer_cond_uncond(inputs, infer_condition=infer_condition)

        if cfg_parallel_branch and "diffusion_prediction" in output:
            # Keep the CFG collective on the device that produced the
            # diffusion prediction (the last device of each pipeline lane).
            # This matches the proven pure-CFG path and avoids an extra
            # cross-device copy immediately before the NCCL collective.
            noise_pred = output["diffusion_prediction"].contiguous()
            if noise_pred.device.type == "cuda":
                torch.cuda.set_device(noise_pred.device)
            noise_pred_list = [torch.empty_like(noise_pred) for _ in range(2)]
            dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
            noise_pred_cond = noise_pred_list[0]
            noise_pred_uncond = noise_pred_list[1]
            noise_pred_guided = self.combine_cfg_predictions(noise_pred_cond, noise_pred_uncond)
            output["diffusion_prediction"] = noise_pred_guided
            self._set_cfg_scheduler_predictions(noise_pred_cond, noise_pred_uncond, noise_pred_guided)
        elif hasattr(self, "scheduler") and "diffusion_prediction" in output:
            self.scheduler.noise_pred = output["diffusion_prediction"]
        return output
