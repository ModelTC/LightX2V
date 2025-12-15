import gc
import glob
import json
import os

import torch
from safetensors import safe_open

from lightx2v.utils.envs import *
from lightx2v.utils.utils import *

from .infer.offload.transformer_infer import QwenImageOffloadTransformerInfer
from .infer.post_infer import QwenImagePostInfer
from .infer.pre_infer import QwenImagePreInfer
from .infer.transformer_infer import QwenImageTransformerInfer
from .weights.post_weights import QwenImagePostWeights
from .weights.pre_weights import QwenImagePreWeights
from .weights.transformer_weights import QwenImageTransformerWeights


class QwenImageTransformerModel:
    pre_weight_class = QwenImagePreWeights
    transformer_weight_class = QwenImageTransformerWeights
    post_weight_class = QwenImagePostWeights

    def __init__(self, config):
        self.config = config
        self.model_path = os.path.join(config["model_path"], "transformer")
        self.cpu_offload = config.get("cpu_offload", False)
        self.offload_granularity = self.config.get("offload_granularity", "block")
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)

        with open(os.path.join(config["model_path"], "transformer", "config.json"), "r") as f:
            transformer_config = json.load(f)
            self.in_channels = transformer_config["in_channels"]
        self.attention_kwargs = {}

        self.dit_quantized = self.config.get("dit_quantized", False)

        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def _init_infer_class(self):
        if self.config["feature_caching"] == "NoCaching":
            self.transformer_infer_class = QwenImageTransformerInfer if not self.cpu_offload else QwenImageOffloadTransformerInfer
        else:
            assert NotImplementedError
        self.pre_infer_class = QwenImagePreInfer
        self.post_infer_class = QwenImagePostInfer

    def _init_weights(self, weight_dict=None):
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        # Some layers run with float32 to achieve high accuracy
        sensitive_layer = {}

        if weight_dict is None:
            is_weight_loader = self._should_load_weights()
            if is_weight_loader:
                if not self.dit_quantized:
                    # Load original weights
                    weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
                else:
                    # Load quantized weights
                    if not self.config.get("lazy_load", False):
                        weight_dict = self._load_quant_ckpt(unified_dtype, sensitive_layer)
                    else:
                        weight_dict = self._load_quant_split_ckpt(unified_dtype, sensitive_layer)

            if self.config.get("device_mesh") is not None and self.config.get("load_from_rank0", False):
                weight_dict = self._load_weights_from_rank0(weight_dict, is_weight_loader)

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        if not self._should_init_empty_model():
            self._apply_weights()

    def _apply_weights(self, weight_dict=None):
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()
        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk."""
        if self.config.get("device_mesh") is None:
            # Single GPU mode
            return True
        elif dist.is_initialized():
            if self.config.get("load_from_rank0", False):
                # Multi-GPU mode, only rank 0 loads
                if dist.get_rank() == 0:
                    logger.info(f"Loading weights from {self.model_path}")
                    return True
            else:
                return True
        return False

    def _should_init_empty_model(self):
        if self.config.get("lora_configs") and self.config["lora_configs"]:
            return True
        return False

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.device.type != "cpu" and dist.is_initialized():
            device = dist.get_rank()
        else:
            device = str(self.device)

        with safe_open(file_path, framework="pt", device=device) as f:
            return {
                key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE()))
                for key in f.keys()
                if not any(remove_key in key for remove_key in remove_keys)
            }

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        if self.config.get("dit_original_ckpt", None):
            safetensors_path = self.config["dit_original_ckpt"]
        else:
            safetensors_path = self.model_path

        if os.path.isdir(safetensors_path):
            safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            safetensors_files = [safetensors_path]

        weight_dict = {}
        for file_path in safetensors_files:
            logger.info(f"Loading weights from {file_path}")
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)

        return weight_dict

    def _load_quant_ckpt(self, unified_dtype, sensitive_layer):
        remove_keys = self.remove_keys if hasattr(self, "remove_keys") else []

        if self.config.get("dit_quantized_ckpt", None):
            safetensors_path = self.config["dit_quantized_ckpt"]
        else:
            safetensors_path = self.model_path

        if os.path.isdir(safetensors_path):
            safetensors_files = glob.glob(os.path.join(safetensors_path, "*.safetensors"))
        else:
            safetensors_files = [safetensors_path]
            safetensors_path = os.path.dirname(safetensors_path)

        weight_dict = {}
        for safetensor_path in safetensors_files:
            with safe_open(safetensor_path, framework="pt") as f:
                logger.info(f"Loading weights from {safetensor_path}")
                for k in f.keys():
                    if any(remove_key in k for remove_key in remove_keys):
                        continue
                    if f.get_tensor(k).dtype in [
                        torch.float16,
                        torch.bfloat16,
                        torch.float,
                    ]:
                        if unified_dtype or all(s not in k for s in sensitive_layer):
                            weight_dict[k] = f.get_tensor(k).to(GET_DTYPE()).to(self.device)
                        else:
                            weight_dict[k] = f.get_tensor(k).to(GET_SENSITIVE_DTYPE()).to(self.device)
                    else:
                        weight_dict[k] = f.get_tensor(k).to(self.device)

        if self.config.get("dit_quant_scheme", "Default") == "nvfp4":
            calib_path = os.path.join(safetensors_path, "calib.pt")
            logger.info(f"[CALIB] Loaded calibration data from: {calib_path}")
            calib_data = torch.load(calib_path, map_location="cpu")
            for k, v in calib_data["absmax"].items():
                weight_dict[k.replace(".weight", ".input_absmax")] = v.to(self.device)

        return weight_dict

    def _load_quant_split_ckpt(self, unified_dtype, sensitive_layer):  # Need rewrite
        lazy_load_model_path = self.dit_quantized_ckpt
        logger.info(f"Loading splited quant model from {lazy_load_model_path}")
        pre_post_weight_dict = {}

        safetensor_path = os.path.join(lazy_load_model_path, "non_block.safetensors")
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if f.get_tensor(k).dtype in [
                    torch.float16,
                    torch.bfloat16,
                    torch.float,
                ]:
                    if unified_dtype or all(s not in k for s in sensitive_layer):
                        pre_post_weight_dict[k] = f.get_tensor(k).to(GET_DTYPE()).to(self.device)
                    else:
                        pre_post_weight_dict[k] = f.get_tensor(k).to(GET_SENSITIVE_DTYPE()).to(self.device)
                else:
                    pre_post_weight_dict[k] = f.get_tensor(k).to(self.device)

        return pre_post_weight_dict

    def _load_weights_from_rank0(self, weight_dict, is_weight_loader):
        logger.info("Loading distributed weights")
        global_src_rank = 0
        target_device = "cpu" if self.cpu_offload else "cuda"

        if is_weight_loader:
            meta_dict = {}
            for key, tensor in weight_dict.items():
                meta_dict[key] = {"shape": tensor.shape, "dtype": tensor.dtype}

            obj_list = [meta_dict]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]
        else:
            obj_list = [None]
            dist.broadcast_object_list(obj_list, src=global_src_rank)
            synced_meta_dict = obj_list[0]

        distributed_weight_dict = {}
        for key, meta in synced_meta_dict.items():
            distributed_weight_dict[key] = torch.empty(meta["shape"], dtype=meta["dtype"], device=target_device)

        if target_device == "cuda":
            dist.barrier(device_ids=[torch.cuda.current_device()])

        for key in sorted(synced_meta_dict.keys()):
            if is_weight_loader:
                distributed_weight_dict[key].copy_(weight_dict[key], non_blocking=True)

            if target_device == "cpu":
                if is_weight_loader:
                    gpu_tensor = distributed_weight_dict[key].cuda()
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()
                else:
                    gpu_tensor = torch.empty_like(distributed_weight_dict[key], device="cuda")
                    dist.broadcast(gpu_tensor, src=global_src_rank)
                    distributed_weight_dict[key].copy_(gpu_tensor.cpu(), non_blocking=True)
                    del gpu_tensor
                    torch.cuda.empty_cache()

                if distributed_weight_dict[key].is_pinned():
                    distributed_weight_dict[key].copy_(distributed_weight_dict[key], non_blocking=True)
            else:
                dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        if target_device == "cuda":
            torch.cuda.synchronize()
        else:
            for tensor in distributed_weight_dict.values():
                if tensor.is_pinned():
                    tensor.copy_(tensor, non_blocking=False)

        logger.info(f"Weights distributed across {dist.get_world_size()} devices on {target_device}")

        return distributed_weight_dict

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)
        if hasattr(self.transformer_infer, "offload_manager"):
            self.transformer_infer.offload_manager.init_cuda_buffer(self.transformer_weights.offload_block_cuda_buffers, self.transformer_weights.offload_phase_cuda_buffers)

    def to_cpu(self):
        self.pre_weight.to_cpu()
        self.transformer_weights.to_cpu()
        self.post_weight.to_cpu()

    def to_cuda(self):
        self.pre_weight.to_cuda()
        self.transformer_weights.to_cuda()
        self.post_weight.to_cuda()

    @torch.no_grad()
    def infer(self, inputs):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        latents = self.scheduler.latents
        if self.config["task"] == "i2i":
            image_latents = torch.cat([item["image_latents"] for item in inputs["image_encoder_output"]], dim=1)
            latents_input = torch.cat([latents, image_latents], dim=1)
        else:
            latents_input = latents

        if self.config["enable_cfg"]:
            if self.config["cfg_parallel"]:
                # ==================== CFG Parallel Processing ====================
                pass
            else:
                # ==================== CFG Processing ====================
                noise_pred_cond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
                if self.config["task"] == "i2i":
                    noise_pred_cond = noise_pred_cond[:, : latents.size(1)]

                noise_pred_uncond = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["negative_prompt_embeds"], infer_condition=False)
                if self.config["task"] == "i2i":
                    noise_pred_uncond = noise_pred_uncond[:, : latents.size(1)]

            comb_pred = noise_pred_uncond + self.config["true_cfg_scale"] * (noise_pred_cond - noise_pred_uncond)
            noise_pred_cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            self.scheduler.noise_pred = comb_pred * (noise_pred_cond_norm / noise_norm)
        else:
            # ==================== No CFG Processing ====================
            noise_pred = self._infer_cond_uncond(latents_input, inputs["text_encoder_output"]["prompt_embeds"], infer_condition=True)
            if self.config["task"] == "i2i":
                noise_pred = noise_pred[:, : latents.size(1)]
            self.scheduler.noise_pred = noise_pred

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        hidden_states, encoder_hidden_states, pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
        )

        encoder_hidden_states, hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            hidden_states=hidden_states.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states.unsqueeze(0),
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out[0])
        return noise_pred
