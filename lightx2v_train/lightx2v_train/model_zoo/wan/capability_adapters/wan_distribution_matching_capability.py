"""Distribution-matching capabilities for Wan-family video models."""

import torch

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _cached_condition,
    _negative_prompt,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.utils.constants import (
    LINGBOT_VIDEO_NEGATIVE_PROMPT,
    WAN_NEGATIVE_PROMPT,
)
from lightx2v_train.utils.generation_shapes import resolve_generation_shape


class WanDistributionMatchingCapability(GenericDistributionMatchingCapability):
    """Distribution-matching operations for Wan-family video models."""

    def encode_training_cache(self, batch):
        cache = super().encode_training_cache(batch)
        inputs = batch.get("inputs", {})
        if inputs.get("video") is not None or inputs.get("latents") is not None:
            cache["inputs"]["latents"] = self.model.encode_to_cache_latent(batch)
        return cache

    @property
    def default_negative_prompt(self):
        return WAN_NEGATIVE_PROMPT

    @property
    def default_lora_target_modules(self):
        return ("q", "k", "v", "o", "ffn.0", "ffn.2")

    @property
    def generation_shape_dimensions(self) -> int:
        return 3

    def latent_shape(
        self,
        batch,
        generation_shapes,
        broadcast,
    ):
        prompt = batch["conditioning"].get("prompt", "")
        _require_single_prompt(prompt)
        num_frames, height, width = resolve_generation_shape(
            generation_shapes,
            batch,
            expected_dimensions=self.generation_shape_dimensions,
            broadcast=broadcast,
        )
        latent_frames = (num_frames - 1) // self.model.vae_scale_factor_temporal + 1
        return (
            1,
            self.model._latent_channels(),
            latent_frames,
            height // self.model.vae_scale_factor_spatial,
            width // self.model.vae_scale_factor_spatial,
        )

    def encode_conditions(
        self,
        batch,
        negative_prompt,
        guidance_scale,
        broadcast,
    ):
        conditioning = batch["conditioning"]
        cached_condition = _cached_condition(batch, self.model)
        if cached_condition is None:
            return super().encode_conditions(
                batch,
                negative_prompt,
                guidance_scale,
                broadcast,
            )
        with torch.no_grad():
            condition = self.prepare_cached_condition(cached_condition)
            if guidance_scale > 1:
                if "negative" in conditioning:
                    cached_negative = _cached_condition(
                        batch,
                        self.model,
                        role="negative",
                    )
                    negative = self.prepare_cached_condition(cached_negative)
                else:
                    scalar = _require_single_prompt(conditioning.get("prompt", ""))
                    prompts = _negative_prompt(
                        conditioning,
                        negative_prompt,
                        scalar=scalar,
                    )
                    negative = self.model.encode_prompt_condition(prompts)
            else:
                negative = None
        return (
            broadcast(condition),
            broadcast(negative) if negative is not None else None,
        )

    def prepare_cached_condition(self, condition):
        if torch.is_tensor(condition):
            prompt_embed = condition
        elif isinstance(condition, dict) and "prompt_embed" in condition:
            prompt_embed = condition["prompt_embed"]
        else:
            raise KeyError("Wan DMD cached condition expects a prompt_embed tensor.")
        prompt_embed = prompt_embed.to(
            device=self.device,
            dtype=self.model.running_dtype,
        )
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        _require_singleton_tensor(prompt_embed, "Wan cached prompt embedding")
        return {"prompt_embed": prompt_embed}


class LingBotDistributionMatchingCapability(WanDistributionMatchingCapability):
    @property
    def default_negative_prompt(self):
        return LINGBOT_VIDEO_NEGATIVE_PROMPT

    @property
    def default_lora_target_modules(self):
        return None

    def prepare_cached_condition(self, condition):
        return self.model.prepare_text_condition(condition)
