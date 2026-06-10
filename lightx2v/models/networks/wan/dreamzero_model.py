import gc
import json
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.wan.infer.dreamzero.pre_infer import DreamZeroPreInfer
from lightx2v.models.networks.wan.infer.dreamzero.transformer_infer import DreamZeroTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.model import WanModel
from lightx2v.models.networks.wan.weights.dreamzero.pre_weights import DreamZeroPreWeights
from lightx2v.models.networks.wan.weights.dreamzero.transformer_weights import DreamZeroTransformerWeights
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE


class DreamZeroModel(WanModel):
    pre_weight_class = DreamZeroPreWeights
    transformer_weight_class = DreamZeroTransformerWeights

    def _init_infer_class(self):
        self.pre_infer_class = DreamZeroPreInfer
        self.post_infer_class = WanPostInfer
        self.transformer_infer_class = DreamZeroTransformerInfer

    @staticmethod
    def cfg_cache_name(cache_name, infer_condition):
        return f"{cache_name}_{'cond' if infer_condition else 'uncond'}"

    def clear_cache(self, cache_name=None):
        self.transformer_infer.clear_cache(cache_name)

    @staticmethod
    def _strip_dreamzero_prefix(key):
        for prefix in ("action_head.model.", "model.diffusion_model.", "diffusion_model."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    @staticmethod
    def _keep_dreamzero_key(key):
        if key.startswith("action_head.model."):
            return True
        if key.startswith(("blocks.", "patch_embedding.", "text_embedding.", "time_embedding.", "time_projection.", "img_emb.", "head.", "action_encoder.", "state_encoder.", "action_decoder.")):
            return True
        return False

    def _load_safetensor_subset(self, file_path, keys, unified_dtype, sensitive_layer):
        if self.device.type != "cpu" and dist.is_initialized():
            device = dist.get_rank()
        else:
            device = str(self.device)
        tensors = {}
        with safe_open(file_path, framework="pt", device=device) as f:
            for key in keys:
                stripped_key = self._strip_dreamzero_prefix(key)
                tensor = f.get_tensor(key)
                if tensor.dtype.is_floating_point:
                    dtype = GET_DTYPE() if unified_dtype or all(s not in stripped_key for s in sensitive_layer) else GET_SENSITIVE_DTYPE()
                    tensor = tensor.to(dtype)
                tensors[stripped_key] = tensor
        return tensors

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = self.config.get("dit_original_ckpt") or self.model_path
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json") if os.path.isdir(ckpt_path) else ""
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
            file_to_keys = {}
            for key, filename in index["weight_map"].items():
                if not self._keep_dreamzero_key(key):
                    continue
                file_to_keys.setdefault(os.path.join(ckpt_path, filename), []).append(key)

            weight_dict = {}
            for file_path, keys in sorted(file_to_keys.items()):
                logger.info("Loading DreamZero DiT weights from {} ({} tensors)", file_path, len(keys))
                weight_dict.update(self._load_safetensor_subset(file_path, keys, unified_dtype, sensitive_layer))
            return weight_dict

        weight_dict = super()._load_ckpt(unified_dtype, sensitive_layer)
        return {self._strip_dreamzero_prefix(key): tensor for key, tensor in weight_dict.items() if self._keep_dreamzero_key(key)}

    def _load_dummy_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = self.config.get("dit_original_ckpt") or self.model_path
        index_path = os.path.join(ckpt_path, "model.safetensors.index.json") if os.path.isdir(ckpt_path) else ""
        if not os.path.exists(index_path):
            return super()._load_dummy_ckpt(unified_dtype, sensitive_layer)

        dummy_device = str(self.device)
        with open(index_path, "r") as f:
            index = json.load(f)
        file_to_keys = {}
        for key, filename in index["weight_map"].items():
            if self._keep_dreamzero_key(key):
                file_to_keys.setdefault(os.path.join(ckpt_path, filename), []).append(key)

        weight_dict = {}
        for file_path, keys in file_to_keys.items():
            tensors_meta = self._read_safetensors_metadata(file_path)
            for key in keys:
                meta = tensors_meta[key]
                stripped_key = self._strip_dreamzero_prefix(key)
                dtype = GET_DTYPE() if unified_dtype or all(s not in stripped_key for s in sensitive_layer) else GET_SENSITIVE_DTYPE()
                weight_dict[stripped_key] = torch.randn(meta["shape"], dtype=dtype, device=dummy_device)
        return weight_dict

    @torch.no_grad()
    def _infer_once(self, inputs, infer_condition=True, update_cache=False, cache_name="pos"):
        model_inputs = dict(inputs)
        if not infer_condition:
            model_inputs["context"] = model_inputs["negative_context"]
        model_inputs["update_cache"] = update_cache
        model_inputs["cache_name"] = cache_name
        pre_infer_out = self.pre_infer.infer(self.pre_weight, model_inputs)
        video_noise_pred, action_noise_pred = self.transformer_infer.infer(self.transformer_weights, pre_infer_out)
        if self.clean_cuda_cache:
            del pre_infer_out
            torch.cuda.empty_cache()
            gc.collect()
        return video_noise_pred, action_noise_pred

    @torch.no_grad()
    def infer(self, inputs):
        enable_cfg = bool(inputs.get("enable_cfg", self.config.get("enable_cfg", False)))
        guide_scale = float(inputs.get("guide_scale", self.config.get("sample_guide_scale", 1.0)))
        update_cache = bool(inputs.get("update_cache", False))
        cache_name = inputs.get("cache_name", "pos")

        if enable_cfg:
            cond_video, cond_action = self._infer_once(
                inputs,
                infer_condition=True,
                update_cache=update_cache,
                cache_name=self.cfg_cache_name(cache_name, True),
            )
            uncond_video, _uncond_action = self._infer_once(
                inputs,
                infer_condition=False,
                update_cache=update_cache,
                cache_name=self.cfg_cache_name(cache_name, False),
            )
            video_noise_pred = uncond_video + guide_scale * (cond_video - uncond_video)
            action_noise_pred = cond_action
        else:
            video_noise_pred, action_noise_pred = self._infer_once(
                inputs,
                infer_condition=True,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        return {
            "video": video_noise_pred,
            "action": action_noise_pred,
        }
