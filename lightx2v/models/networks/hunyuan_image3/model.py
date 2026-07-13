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
        return "cuda:0"
    return device


def _resolve_sequence_parallel_pipeline_lane(config, devices):
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HunyuanImage3 sequence-parallel pipeline split requires torch.distributed initialization.")

    seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p")
    seq_p_size = dist.get_world_size(seq_p_group)
    seq_p_rank = dist.get_rank(seq_p_group)
    if len(devices) % seq_p_size:
        raise ValueError(
            f"HunyuanImage3 sequence parallel requires pipeline device count ({len(devices)}) "
            f"to be divisible by seq_p_size ({seq_p_size})."
        )

    layout = str(config.get("hunyuan_image3_pipeline_layout", "interleaved")).strip().lower()
    if layout != "interleaved":
        raise ValueError(
            "HunyuanImage3 sequence parallel uses Wan-style rank/device initialization and currently requires "
            f"hunyuan_image3_pipeline_layout='interleaved'; got {layout!r}."
        )

    lane = devices[seq_p_rank::seq_p_size]
    expected_device = f"cuda:{torch.cuda.current_device()}"
    if lane[0] != expected_device:
        raise RuntimeError(
            "HunyuanImage3 interleaved pipeline lane must start on the CUDA device selected by "
            f"init_parallel_env: lane={lane}, current_device={expected_device}."
        )
    return lane


def resolve_pipeline_devices(config, fallback_device):
    configured = config.get("pipeline_parallel_devices") or config.get("hunyuan_image3_pipeline_devices")
    if configured:
        if isinstance(configured, str):
            devices = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            devices = list(configured)
        devices = [_normalize_device_name(device) for device in devices]
        if config.get("seq_parallel", False):
            return _resolve_sequence_parallel_pipeline_lane(config, devices)
        return devices

    if config.get("pipeline_parallel", False) and torch.cuda.is_available():
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
                        raise ValueError(
                            f"HunyuanImage3 cfg_parallel requires visible CUDA device count ({device_count}) "
                            f"to be divisible by cfg_p_size ({cfg_p_size})."
                        )
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
        num_layers = int(config["num_layers"])
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
        self.pipeline_devices = resolve_pipeline_devices(config, device)
        self.pipeline_parallel = len(set(self.pipeline_devices)) > 1
        self.sequence_parallel_attn_type = self._resolve_sequence_parallel_attn_type()
        self._sp_gather_buffers = {}
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

        if self.config.get("seq_parallel", False):
            logger.info(
                "HunyuanImage3 sequence parallel initialized: "
                f"rank={dist.get_rank(self.seq_p_group)}, "
                f"size={dist.get_world_size(self.seq_p_group)}, "
                f"attention={self.sequence_parallel_attn_type}, "
                f"pipeline_devices={self.pipeline_devices}"
            )

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
            raise ValueError(
                "HunyuanImage3 parallel.seq_p_attn_type must be one of: "
                f"kv_all_gather, ulysses; got {attn_type!r}."
            )
        return attn_type

    def _validate_sequence_parallel_config(self):
        if not self.config.get("seq_parallel", False):
            return
        if self.seq_p_group is None:
            raise RuntimeError("HunyuanImage3 sequence parallel requires an initialized seq_p process group.")
        if self.config.get("cfg_parallel", False):
            raise ValueError("HunyuanImage3 sequence parallel requires parallel.cfg_p_size=1.")
        cfg_mode = str(self.config.get("hunyuan_cfg_mode", "batch")).strip().lower()
        if self.config.get("enable_cfg", False) and cfg_mode != "serial":
            raise ValueError(
                "HunyuanImage3 sequence parallel requires hunyuan_cfg_mode='serial' so every transformer forward has batch size 1."
            )
        if self.config.get("use_taylor_cache", False) and self.config.get("enable_kv_cache", False):
            raise ValueError("HunyuanImage3 sequence parallel does not support enabling Taylor cache and KV cache together.")
        if self.sequence_parallel_attn_type == "ulysses":
            world_size = dist.get_world_size(self.seq_p_group)
            q_heads = int(self.config["num_attention_heads"])
            kv_heads = int(self.config.get("num_key_value_heads") or q_heads)
            if q_heads % world_size or kv_heads % world_size:
                raise ValueError(
                    "HunyuanImage3 Ulysses requires seq_p_size to divide both Q and KV heads: "
                    f"Q={q_heads}, KV={kv_heads}, seq_p_size={world_size}."
                )

    def _tensor_target_device(self, key):
        return resolve_pipeline_device_for_key(key, self.config, self.pipeline_devices)

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        ext = os.path.splitext(file_path)[-1]
        if ext in (".pt", ".pth", ".tar"):
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
            or (
                cache_dic["enable_tailing_enhance"]
                and current_step >= int(cache_dic["num_steps"]) - int(cache_dic["tailing_enhance_steps"])
            )
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
        return self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        if pre_infer_out.hidden_states.shape[0] != 1:
            raise ValueError(
                "HunyuanImage3 sequence parallel expects batch size 1 per transformer forward; "
                "set hunyuan_cfg_mode='serial'."
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

        pre_infer_out.hidden_states = self._pad_and_slice_sequence(
            pre_infer_out.hidden_states, 1, padding_size, local_start, local_seq_len, value=0
        )
        pre_infer_out.position_ids = self._pad_and_slice_sequence(
            pre_infer_out.position_ids, 1, padding_size, local_start, local_seq_len, value=0
        )
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
        if cfg_parallel_branch:
            if not self.config.get("cfg_parallel", False):
                raise RuntimeError("HunyuanImage3 received a cfg-parallel branch input, but config['cfg_parallel'] is not enabled.")
            cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
            assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
            infer_condition = dist.get_rank(cfg_p_group) == 0

        output = self._infer_cond_uncond(inputs, infer_condition=infer_condition)

        if cfg_parallel_branch and "diffusion_prediction" in output:
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
