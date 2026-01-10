import gc
import glob
import os

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn import functional as F

from lightx2v.utils.envs import *
from lightx2v.utils.utils import logger
from lightx2v_platform.base.global_var import AI_DEVICE

from .infer.post_infer import LongCatImagePostInfer
from .infer.pre_infer import LongCatImagePreInfer
from .infer.transformer_infer import LongCatImageTransformerInfer
from .weights.post_weights import LongCatImagePostWeights
from .weights.pre_weights import LongCatImagePreWeights
from .weights.transformer_weights import LongCatImageTransformerWeights


class LongCatImageTransformerModel:
    """Transformer model for LongCat Image.

    Handles weight loading and inference for the LongCat architecture
    (10 double-stream blocks + 20 single-stream blocks).
    """

    pre_weight_class = LongCatImagePreWeights
    transformer_weight_class = LongCatImageTransformerWeights
    post_weight_class = LongCatImagePostWeights

    def __init__(self, config):
        self.config = config
        self.model_path = os.path.join(config["model_path"], "transformer")
        self.cpu_offload = config.get("cpu_offload", False)
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)

        # Use transformer_in_channels to avoid conflict with VAE's in_channels
        self.in_channels = self.config.get("transformer_in_channels", self.config.get("in_channels", 64))
        self.attention_kwargs = {}
        self.remove_keys = []

        if self.config.get("seq_parallel", False):
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

        self._init_infer_class()
        self._init_weights()
        self._init_infer()

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def _init_infer_class(self):
        self.transformer_infer_class = LongCatImageTransformerInfer
        self.pre_infer_class = LongCatImagePreInfer
        self.post_infer_class = LongCatImagePostInfer

    def _init_weights(self, weight_dict=None):
        unified_dtype = GET_DTYPE() == GET_SENSITIVE_DTYPE()
        sensitive_layer = {}

        if weight_dict is None:
            is_weight_loader = self._should_load_weights()
            if is_weight_loader:
                weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)

            if self.config.get("device_mesh") is not None and self.config.get("load_from_rank0", False):
                weight_dict = self._load_weights_from_rank0(weight_dict, is_weight_loader)

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)

        self._apply_weights()

    def _apply_weights(self, weight_dict=None):
        if weight_dict is not None:
            self.original_weight_dict = weight_dict
            del weight_dict
            gc.collect()

        # Load weights into containers
        self.pre_weight.load_from_state_dict(self.original_weight_dict)
        self.transformer_weights.load_from_state_dict(self.original_weight_dict)
        self.post_weight.load_from_state_dict(self.original_weight_dict)

        del self.original_weight_dict
        torch.cuda.empty_cache()
        gc.collect()

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk."""
        if self.config.get("device_mesh") is None:
            return True
        elif dist.is_initialized():
            if self.config.get("load_from_rank0", False):
                if dist.get_rank() == 0:
                    logger.info(f"Loading weights from {self.model_path}")
                    return True
            else:
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
            dist.broadcast(distributed_weight_dict[key], src=global_src_rank)

        if target_device == "cuda":
            torch.cuda.synchronize()

        logger.info(f"Weights distributed across {dist.get_world_size()} devices on {target_device}")
        return distributed_weight_dict

    def _init_infer(self):
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.pre_infer = self.pre_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

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
            self.to_cuda()

        latents = self.scheduler.latents

        if self.config.get("enable_cfg", True):
            # ==================== CFG Processing ====================
            noise_pred_cond = self._infer_cond_uncond(
                latents,
                inputs["text_encoder_output"]["prompt_embeds"],
                infer_condition=True
            )
            noise_pred_uncond = self._infer_cond_uncond(
                latents,
                inputs["text_encoder_output"]["negative_prompt_embeds"],
                infer_condition=False
            )

            # Apply CFG with optional renormalization
            noise_pred = self.scheduler.apply_cfg(noise_pred_cond, noise_pred_uncond)
            self.scheduler.noise_pred = noise_pred
        else:
            # ==================== No CFG Processing ====================
            noise_pred = self._infer_cond_uncond(
                latents,
                inputs["text_encoder_output"]["prompt_embeds"],
                infer_condition=True
            )
            self.scheduler.noise_pred = noise_pred

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, prompt_embeds, infer_condition=True):
        self.scheduler.infer_condition = infer_condition

        pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            encoder_hidden_states=prompt_embeds,
        )

        hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(
            self.post_weight,
            hidden_states,
            pre_infer_out.temb
        )

        return noise_pred
