import json
import os
import re

import torch
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hunyuan_image3.infer.kv_cache import HunyuanImage3TaylorCache
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


def resolve_pipeline_devices(config, fallback_device):
    configured = config.get("pipeline_parallel_devices") or config.get("hunyuan_image3_pipeline_devices")
    if configured:
        if isinstance(configured, str):
            devices = [item.strip() for item in configured.split(",") if item.strip()]
        else:
            devices = list(configured)
        return [_normalize_device_name(device) for device in devices]

    if config.get("pipeline_parallel", False) and torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
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
        self._validate_requested_model_variant()
        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _tensor_target_device(self, key):
        return resolve_pipeline_device_for_key(key, self.config, self.pipeline_devices)

    def _iter_checkpoint_keys(self):
        checkpoint_path = self.config.get("dit_original_ckpt") or self.model_path
        if os.path.isdir(checkpoint_path):
            index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    index = json.load(f)
                weight_map = index.get("weight_map")
                if isinstance(weight_map, dict):
                    yield from weight_map.keys()
                    return
            file_paths = [os.path.join(checkpoint_path, name) for name in sorted(os.listdir(checkpoint_path)) if name.endswith(".safetensors")]
        else:
            file_paths = [checkpoint_path]

        for file_path in file_paths:
            if not str(file_path).endswith(".safetensors") or not os.path.exists(file_path):
                continue
            with safe_open(file_path, framework="pt", device="cpu") as f:
                yield from f.keys()

    def _checkpoint_has_prefix(self, prefix):
        return any(key.startswith(prefix) for key in self._iter_checkpoint_keys())

    def _validate_requested_model_variant(self):
        if self.config.get("use_meanflow", False) and not self._checkpoint_has_prefix("timestep_r_emb."):
            raise ValueError(
                "HunyuanImage3 use_meanflow=True requires checkpoint weights under 'timestep_r_emb.'. "
                f"The checkpoint at {self.model_path!r} does not contain these weights. "
                "Disable use_meanflow for the standard HunyuanImage-3-Instruct checkpoint, "
                "or switch model_path to a meanflow-distilled HunyuanImage3 checkpoint."
            )

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
        feature_caching = self.config.get("feature_caching", "NoCaching")
        if feature_caching in ("TaylorSeer", "Taylor", "TaylorCache"):
            self.config["use_taylor_cache"] = True
        elif feature_caching != "NoCaching":
            raise NotImplementedError(f"HunyuanImage3 feature_caching={feature_caching!r} is not supported; use TaylorSeer or NoCaching.")
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
            hidden_states = self._seq_parallel_post_process(hidden_states)
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
        raise NotImplementedError("HunyuanImage3 native sequence parallel is not implemented yet.")

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        raise NotImplementedError("HunyuanImage3 native sequence parallel is not implemented yet.")

    @torch.no_grad()
    def infer(self, inputs):
        output = self._infer_cond_uncond(inputs, infer_condition=True)
        if hasattr(self, "scheduler") and "diffusion_prediction" in output:
            self.scheduler.noise_pred = output["diffusion_prediction"]
        return output
