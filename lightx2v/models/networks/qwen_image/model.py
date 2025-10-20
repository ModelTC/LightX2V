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
        self.device = torch.device("cpu") if self.cpu_offload else torch.device("cuda")

        with open(os.path.join(config["model_path"], "transformer", "config.json"), "r") as f:
            transformer_config = json.load(f)
            self.in_channels = transformer_config["in_channels"]
        self.attention_kwargs = {}

        self.dit_quantized = self.config["mm_config"].get("mm_type", "Default") != "Default"
        self.weight_auto_quant = self.config["mm_config"].get("weight_auto_quant", False)

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
                if not self.dit_quantized or self.weight_auto_quant:
                    # Load original weights
                    weight_dict = self._load_ckpt(unified_dtype, sensitive_layer)
                else:
                    # Load quantized weights
                    assert NotImplementedError

            if self.config.get("device_mesh") is not None:
                weight_dict = self._load_weights_distribute(weight_dict, is_weight_loader)

            self.original_weight_dict = weight_dict
        else:
            self.original_weight_dict = weight_dict

        # Initialize weight containers
        self.pre_weight = self.pre_weight_class(self.config)
        self.transformer_weights = self.transformer_weight_class(self.config)
        self.post_weight = self.post_weight_class(self.config)
        # Load weights into containers
        self.pre_weight.load(self.original_weight_dict)
        self.transformer_weights.load(self.original_weight_dict)
        self.post_weight.load(self.original_weight_dict)

    def _should_load_weights(self):
        """Determine if current rank should load weights from disk."""
        if self.config.get("device_mesh") is None:
            # Single GPU mode
            return True
        elif dist.is_initialized():
            # Multi-GPU mode, only rank 0 loads
            if dist.get_rank() == 0:
                logger.info(f"Loading weights from {self.model_path}")
                return True
        return False

    def _load_safetensor_to_dict(self, file_path, unified_dtype, sensitive_layer):
        with safe_open(file_path, framework="pt", device=str(self.device)) as f:
            return {key: (f.get_tensor(key).to(GET_DTYPE()) if unified_dtype or all(s not in key for s in sensitive_layer) else f.get_tensor(key).to(GET_SENSITIVE_DTYPE())) for key in f.keys()}

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        safetensors_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
        weight_dict = {}
        for file_path in safetensors_files:
            file_weights = self._load_safetensor_to_dict(file_path, unified_dtype, sensitive_layer)
            weight_dict.update(file_weights)
        return weight_dict

    def _load_weights_distribute(self, weight_dict, is_weight_loader):
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

        t = self.scheduler.timesteps[self.scheduler.step_index]
        latents = self.scheduler.latents
        if self.config["task"] == "i2i":
            image_latents = inputs["image_encoder_output"]["image_latents"]
            latents_input = torch.cat([latents, image_latents], dim=1)
        else:
            latents_input = latents

        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        img_shapes = inputs["img_shapes"]

        prompt_embeds = inputs["text_encoder_output"]["prompt_embeds"]
        prompt_embeds_mask = inputs["text_encoder_output"]["prompt_embeds_mask"]

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None

        hidden_states, encoder_hidden_states, _, pre_infer_out = self.pre_infer.infer(
            weights=self.pre_weight,
            hidden_states=latents_input,
            timestep=timestep / 1000,
            guidance=self.scheduler.guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=self.attention_kwargs,
        )

        encoder_hidden_states, hidden_states = self.transformer_infer.infer(
            block_weights=self.transformer_weights,
            hidden_states=hidden_states.unsqueeze(0),
            encoder_hidden_states=encoder_hidden_states.unsqueeze(0),
            pre_infer_out=pre_infer_out,
        )

        noise_pred = self.post_infer.infer(self.post_weight, hidden_states, pre_infer_out[0])

        if self.config["do_true_cfg"]:
            neg_prompt_embeds = inputs["text_encoder_output"]["negative_prompt_embeds"]
            neg_prompt_embeds_mask = inputs["text_encoder_output"]["negative_prompt_embeds_mask"]

            negative_txt_seq_lens = neg_prompt_embeds_mask.sum(dim=1).tolist() if neg_prompt_embeds_mask is not None else None

            neg_hidden_states, neg_encoder_hidden_states, _, neg_pre_infer_out = self.pre_infer.infer(
                weights=self.pre_weight,
                hidden_states=latents_input,
                timestep=timestep / 1000,
                guidance=self.scheduler.guidance,
                encoder_hidden_states_mask=neg_prompt_embeds_mask,
                encoder_hidden_states=neg_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=negative_txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
            )

            neg_encoder_hidden_states, neg_hidden_states = self.transformer_infer.infer(
                block_weights=self.transformer_weights,
                hidden_states=neg_hidden_states.unsqueeze(0),
                encoder_hidden_states=neg_encoder_hidden_states.unsqueeze(0),
                pre_infer_out=neg_pre_infer_out,
            )

            neg_noise_pred = self.post_infer.infer(self.post_weight, neg_hidden_states, neg_pre_infer_out[0])

        if self.config["task"] == "i2i":
            noise_pred = noise_pred[:, : latents.size(1)]

        if self.config["do_true_cfg"]:
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
            comb_pred = neg_noise_pred + self.config["true_cfg_scale"] * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        noise_pred = noise_pred[:, : latents.size(1)]
        self.scheduler.noise_pred = noise_pred
