import os

import torch
from loguru import logger

from lightx2v.models.networks.helios.transformer_helios import HeliosTransformer3DModel
from lightx2v.utils.envs import GET_DTYPE


class HeliosModel:
    def __init__(self, model_path, config, device):
        self.config = config
        self.device = device
        transformer_path = config.get("transformer_model_path") or model_path
        self.transformer = HeliosTransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=None if os.path.basename(transformer_path) == "transformer" else "transformer",
            torch_dtype=GET_DTYPE(),
        ).to(device)
        self.scheduler = None
        self._set_attention_backend()

    def _set_attention_backend(self):
        attn_type = self.config.get("attn_type")
        if not attn_type:
            return
        try:
            if attn_type == "flash_attn3":
                self.transformer.set_attention_backend("_flash_3_hub")
            elif attn_type == "flash_attn2":
                self.transformer.set_attention_backend("flash_hub")
            elif attn_type == "torch_sdpa":
                self.transformer.set_attention_backend("sdpa")
        except Exception as exc:
            logger.warning(f"Failed to set Helios attention backend {attn_type}: {exc}")

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @property
    def dtype(self):
        return self.transformer.dtype

    def infer_noise(
        self,
        latents,
        timestep,
        encoder_hidden_states,
        history_inputs,
        attention_kwargs=None,
    ):
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            attention_kwargs=attention_kwargs,
            return_dict=False,
            **history_inputs,
        )[0]

    def infer_cfg(
        self,
        latents,
        timestep,
        prompt_embeds,
        negative_prompt_embeds,
        history_inputs,
        guidance_scale,
        attention_kwargs=None,
        is_cfg_zero_star=False,
        use_zero_init=False,
        zero_steps=1,
        stage_idx=0,
        step_idx=0,
    ):
        with self.transformer.cache_context("cond"):
            noise_pred = self.infer_noise(latents, timestep, prompt_embeds, history_inputs, attention_kwargs)

        if guidance_scale <= 1.0 or negative_prompt_embeds is None:
            return noise_pred

        with self.transformer.cache_context("uncond"):
            noise_uncond = self.infer_noise(latents, timestep, negative_prompt_embeds, history_inputs, attention_kwargs)

        if is_cfg_zero_star:
            positive_flat = noise_pred.view(noise_pred.shape[0], -1).float()
            negative_flat = noise_uncond.view(noise_uncond.shape[0], -1).float()
            dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
            squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
            alpha = (dot_product / squared_norm).view(noise_pred.shape[0], *([1] * (noise_pred.ndim - 1))).to(noise_pred.dtype)
            if stage_idx == 0 and step_idx <= zero_steps and use_zero_init:
                return noise_pred * 0.0
            return noise_uncond * alpha + guidance_scale * (noise_pred - noise_uncond * alpha)

        return noise_uncond + guidance_scale * (noise_pred - noise_uncond)
