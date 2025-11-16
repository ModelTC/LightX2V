import gc
import glob
import os

import torch
import torch.distributed as dist
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.hunyuan_video.infer.post_infer import HunyuanVideo15PostInfer
from lightx2v.models.networks.hunyuan_video.infer.pre_infer import HunyuanVideo15PreInfer
from lightx2v.models.networks.hunyuan_video.infer.transformer_infer import HunyuanVideo15TransformerInfer
from lightx2v.models.networks.hunyuan_video.weights.post_weights import HunyuanVideo15PostWeights
from lightx2v.models.networks.hunyuan_video.weights.pre_weights import HunyuanVideo15PreWeights
from lightx2v.models.networks.hunyuan_video.weights.transformer_weights import HunyuanVideo15TransformerWeights
from lightx2v.utils.custom_compiler import CompiledMethodsMixin
from lightx2v.utils.envs import *


class HunyuanVideo15Model(CompiledMethodsMixin):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.device = device
        self.cpu_offload = self.config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")

        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def _init_infer_class(self):
        self.pre_infer_class = HunyuanVideo15PreInfer
        self.post_infer_class = HunyuanVideo15PostInfer
        self.transformer_infer_class = HunyuanVideo15TransformerInfer

    def _init_weights(self):
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        sensitive_layer = {}
        weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
        self.original_weight_dict = weight_dict
        self.pre_weight = HunyuanVideo15PreWeights(self.config)
        self.transformer_weight = HunyuanVideo15TransformerWeights(self.config)
        self.post_weight = HunyuanVideo15PostWeights(self.config)
        # print(f"original_weight_dict: {self.original_weight_dict}")
        # for k in self.original_weight_dict.keys():
        #     print(k, self.original_weight_dict[k].shape)
        self._apply_weights()

    def _apply_weights(self, weight_dict=None):
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()
        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weight.load(self.original_weight_dict)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if self.config.get("dit_original_ckpt", None):
            safetensors_path = self.config["dit_original_ckpt"]
        else:
            safetensors_path = self.config["transformer_model_path"]

        if os.path.isdir(safetensors_path):
            safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            safetensors_files = [safetensors_path]

        weight_dict = {}
        for file_path in safetensors_files:
            if self.config.get("adapter_model_path", None) is not None:
                if self.config["adapter_model_path"] == file_path:
                    continue
            logger.info(f"Loading weights from {file_path}")
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.device.type == "cuda" and dist.is_initialized():
            device = torch.device("cuda:{}".format(dist.get_rank()))
        else:
            device = self.device

        with safe_open(file_path, framework="pt", device=str(device)) as f:
            return {
                key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE()))
                for key in f.keys()
                if not any(remove_key in key for remove_key in remove_keys)
            }

    @torch.no_grad()
    def infer(self, inputs):
        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)
                else:
                    noise_pred = self._infer_cond_uncond(inputs, infer_condition=False)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]  # cfg_p_rank == 0
                noise_pred_uncond = noise_pred_list[1]  # cfg_p_rank == 1
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(inputs, infer_condition=True)
                noise_pred_uncond = self._infer_cond_uncond(inputs, infer_condition=False)

            self.scheduler.noise_pred = noise_pred_uncond + self.scheduler.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            # ==================== No CFG ====================
            self.scheduler.noise_pred = self._infer_cond_uncond(inputs, infer_condition=True)

    @torch.no_grad()
    def _infer_cond_uncond(self, inputs, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(self.pre_weight, inputs)

        x = self.transformer_infer.infer(self.transformer_weight, pre_infer_out)

        noise_pred = self.post_infer.infer(x, pre_infer_out)[0]

        return noise_pred
