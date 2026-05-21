"""Hunyuan3D DiT model wrapper for LightX2V."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.hunyuan3d.infer.post_infer import Hunyuan3DPostInfer
from lightx2v.models.networks.hunyuan3d.infer.pre_infer import Hunyuan3DPreInfer
from lightx2v.models.networks.hunyuan3d.infer.transformer_infer import Hunyuan3DTransformerInfer
from lightx2v.models.networks.hunyuan3d.utils.checkpoint import (
    load_checkpoint_dict,
    load_pipeline_config,
    resolve_ckpt_paths,
    resolve_model_dir,
)
from lightx2v.models.networks.hunyuan3d.weights.post_weights import Hunyuan3DPostWeights
from lightx2v.models.networks.hunyuan3d.weights.pre_weights import Hunyuan3DPreWeights
from lightx2v.models.networks.hunyuan3d.weights.transformer_weights import Hunyuan3DTransformerWeights
from lightx2v.utils.custom_compiler import compiled_method
from lightx2v.utils.envs import GET_DTYPE, GET_SENSITIVE_DTYPE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v_platform.base.global_var import AI_DEVICE


class Hunyuan3DDiTModel(BaseTransformerModel):
    """LightX2V DiT for Hunyuan3D shape generation (pre / transformer / post split)."""

    pre_weight_class = Hunyuan3DPreWeights
    transformer_weight_class = Hunyuan3DTransformerWeights
    post_weight_class = Hunyuan3DPostWeights

    def __init__(self, model_path, config, device, weight_dict=None, model_type="hunyuan3d", lora_path=None, lora_strength=1.0):
        self.config = config
        self._merge_dit_config(model_path, config)
        super().__init__(model_path, config, device, model_type, lora_path, lora_strength)
        self.in_channels = self.config.get("in_channels", 64)
        self._init_infer_class()
        self._init_weights(weight_dict=weight_dict)
        self._init_infer()
        if not self.cpu_offload:
            self.to_cuda()

    def _resolve_dit_dtype(self):
        return self._resolve_dit_dtype_from_config(self.config)

    @staticmethod
    def _merge_dit_config(model_path, config):
        if config.get("hidden_size") and config.get("depth"):
            return
        config_path = os.path.join(model_path, "config.yaml")
        if not os.path.isfile(config_path):
            model_dir = resolve_model_dir(config["model_path"], config.get("subfolder", "hunyuan3d-dit-v2-1"))
            config_path = os.path.join(model_dir, "config.yaml")
        pipeline_cfg = load_pipeline_config(config_path)
        for key, value in pipeline_cfg["model"]["params"].items():
            config.setdefault(key, value)
        config.setdefault("dit_quant_scheme", "Default")
        config.setdefault("ln_norm_type", "torch")
        config.setdefault("rms_norm_type", "torch")
        config.setdefault("attn_type", "flash_attn3")

    def _init_infer_class(self):
        self.pre_infer_class = Hunyuan3DPreInfer
        self.transformer_infer_class = Hunyuan3DTransformerInfer
        self.post_infer_class = Hunyuan3DPostInfer

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    def _load_ckpt(self, unified_dtype, sensitive_layer):
        ckpt_path = self.config.get("dit_original_ckpt")
        if not ckpt_path:
            model_dir = self.model_path
            if not os.path.isfile(os.path.join(model_dir, "config.yaml")):
                model_dir = resolve_model_dir(self.config["model_path"], self.config.get("subfolder", "hunyuan3d-dit-v2-1"))
            _, ckpt_path = resolve_ckpt_paths(
                model_dir,
                use_safetensors=bool(self.config.get("use_safetensors", False)),
                variant=self.config.get("variant", "fp16"),
            )
        use_safetensors = bool(self.config.get("use_safetensors", False)) or str(ckpt_path).endswith(".safetensors")
        ckpt = load_checkpoint_dict(ckpt_path, use_safetensors=use_safetensors)
        return self._build_weight_dict(ckpt["model"], unified_dtype, sensitive_layer, config=self.config)

    @staticmethod
    def _build_weight_dict(state_dict, unified_dtype, sensitive_layer, config=None):
        if config is not None:
            dtype = Hunyuan3DDiTModel._resolve_dit_dtype_from_config(config)
        else:
            dtype = GET_DTYPE() if unified_dtype else GET_SENSITIVE_DTYPE()
        weight_dict = {}
        for key, tensor in state_dict.items():
            if config is None and not (unified_dtype or all(s not in key for s in sensitive_layer)):
                dtype = GET_SENSITIVE_DTYPE()
            weight_dict[key] = tensor.to(dtype=dtype)
        return weight_dict

    @property
    def guidance_embed(self):
        return self.config.get("guidance_cond_proj_dim") is not None

    @staticmethod
    def _resolve_dit_dtype_from_config(config):
        dtype_name = str(config.get("dtype", "fp16")).lower()
        if dtype_name in {"fp16", "float16"}:
            return torch.float16
        if dtype_name in {"bf16", "bfloat16"}:
            return torch.bfloat16
        return GET_DTYPE()

    @property
    def dtype(self):
        return self._resolve_dit_dtype()

    @torch.no_grad()
    def _infer_forward(self, latent_model_input, timestep, cond, guidance=None):
        cond_main = cond["main"]
        with ProfilingContext4DebugL1("pre_infer"):
            pre_out = self.pre_infer.infer(
                self.pre_weight,
                latent_model_input,
                cond_main,
                timestep,
                guidance_cond=guidance,
            )
        with ProfilingContext4DebugL1("transformer_infer"):
            hidden_states = self.transformer_infer.infer(self.transformer_weights, pre_out)
        with ProfilingContext4DebugL1("post_infer"):
            noise_pred = self.post_infer.infer(self.post_weight, hidden_states)
        return noise_pred

    @torch.no_grad()
    def _infer_cond_uncond(self, latents_input, cond, guidance=None):
        latent_model_input = latents_input.to(dtype=self.dtype)
        t = self.scheduler.current_timestep
        timestep = t.expand(latent_model_input.shape[0]).to(self.dtype)
        timestep = timestep / self.scheduler.num_train_timesteps
        return self._infer_forward(latent_model_input, timestep, cond, guidance=guidance)

    @torch.no_grad()
    def _seq_parallel_pre_process(self, pre_infer_out):
        return pre_infer_out

    @torch.no_grad()
    def _seq_parallel_post_process(self, x):
        return x

    @classmethod
    def from_pretrained(cls, config, ckpt: dict[str, dict[str, torch.Tensor]] | None = None):
        model_dir = resolve_model_dir(config["model_path"], config.get("subfolder", "hunyuan3d-dit-v2-1"))
        device = config.get("device", AI_DEVICE)
        weight_dict = None
        if ckpt is not None:
            weight_dict = cls._build_weight_dict(ckpt["model"], GET_DTYPE() == GET_SENSITIVE_DTYPE(), {}, config=config)
        return cls(model_dir, config, device, weight_dict=weight_dict)

    @compiled_method()
    @torch.no_grad()
    def infer(self, inputs: dict):
        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == 0:
                self.to_cuda()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cuda()
                self.post_weight.to_cuda()

        latents = self.scheduler.latents
        cond = inputs["cond"]
        uncond = inputs.get("uncond")
        guidance = inputs.get("guidance")
        guidance_scale = inputs.get("guidance_scale", 5.0)
        do_cfg = inputs.get("do_classifier_free_guidance", False)

        if do_cfg:
            if self.config.get("cfg_parallel"):
                cfg_p_group = self.config["device_mesh"].get_group(mesh_dim="cfg_p")
                assert dist.get_world_size(cfg_p_group) == 2, "cfg_p_world_size must be equal to 2"
                cfg_p_rank = dist.get_rank(cfg_p_group)

                if cfg_p_rank == 0:
                    noise_pred = self._infer_cond_uncond(latents, cond, guidance=guidance)
                else:
                    noise_pred = self._infer_cond_uncond(latents, uncond, guidance=guidance)

                noise_pred_list = [torch.zeros_like(noise_pred) for _ in range(2)]
                dist.all_gather(noise_pred_list, noise_pred, group=cfg_p_group)
                noise_pred_cond = noise_pred_list[0]
                noise_pred_uncond = noise_pred_list[1]
            else:
                noise_pred_cond = self._infer_cond_uncond(latents, cond, guidance=guidance)
                noise_pred_uncond = self._infer_cond_uncond(latents, uncond, guidance=guidance)

            self.scheduler.noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            self.scheduler.noise_pred = self._infer_cond_uncond(latents, cond, guidance=guidance)

        if self.cpu_offload:
            if self.offload_granularity == "model" and self.scheduler.step_index == self.scheduler.infer_steps - 1:
                self.to_cpu()
            elif self.offload_granularity != "model":
                self.pre_weight.to_cpu()
                self.post_weight.to_cpu()
