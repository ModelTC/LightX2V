"""Distribution-matching capabilities for Wan-family video models."""

import math

import torch
from torch.autograd import forward_ad

from lightx2v_train.model_zoo.capability_adapters.common import (
    GenericDistributionMatchingCapability,
    _cached_condition,
    _negative_prompt,
    _require_single_prompt,
    _require_singleton_tensor,
)
from lightx2v_train.runtime.distributed import get_sequence_parallel_world_size
from lightx2v_train.trainers.dmd.math import dmd_loss_with_stats, expand_sigma, official_pdmd_critic_loss, official_pdmd_loss_with_stats
from lightx2v_train.utils.constants import (
    LINGBOT_VIDEO_NEGATIVE_PROMPT,
    WAN_NEGATIVE_PROMPT,
)
from lightx2v_train.utils.generation_shapes import resolve_generation_shape


class WanDistributionMatchingCapability(GenericDistributionMatchingCapability):
    """Distribution-matching operations for Wan-family video models."""

    def __init__(self, model):
        super().__init__(model)
        options = model.config["model"].get("capabilities", {}).get("distribution_matching", {})
        self.official_pdmd = bool(options.get("official_pdmd", False))
        self.projected_dmd = bool(options.get("projected_dmd", self.official_pdmd))
        if self.projected_dmd and not self.official_pdmd:
            raise ValueError("The early projected_dmd recipe has been removed; use official_pdmd=true for the released PDMD recipe.")
        self.official_pdmd_weight = float(options.get("video_loss_weight", 0.8))
        self.official_pdmd_normalizer_floor = float(options.get("official_pdmd_normalizer_floor", 1e-5))
        self.official_pdmd_projection_epsilon = float(options.get("official_pdmd_projection_epsilon", 1e-8))
        self.official_pdmd_loss_clamp = float(options.get("official_pdmd_loss_clamp", 5.0))
        if self.official_pdmd:
            values = (self.official_pdmd_weight, self.official_pdmd_projection_epsilon, self.official_pdmd_loss_clamp)
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("Wan official PDMD weight, projection epsilon and loss clamp must be finite and non-negative.")
            if not math.isfinite(self.official_pdmd_normalizer_floor) or self.official_pdmd_normalizer_floor <= 0:
                raise ValueError("Wan official PDMD normalizer floor must be finite and positive.")
            if getattr(model, "use_causal_transformer", False):
                raise ValueError("Wan official PDMD supports the bidirectional T2V transformer.")
        self._last_dmd_metrics = {}

    def dmd_loss(self, latents, fake_x0, teacher_x0):
        if self.official_pdmd:
            loss, normalizer, direction_rms, nonfinite = official_pdmd_loss_with_stats(
                latents,
                fake_x0,
                teacher_x0,
                projected=self.projected_dmd,
                normalizer_floor=self.official_pdmd_normalizer_floor,
                projection_epsilon=self.official_pdmd_projection_epsilon,
                clamp_max=self.official_pdmd_loss_clamp,
            )
            self._last_dmd_metrics = {"dmd_video": loss.detach(), "dmd_video_normalizer": normalizer, "dmd_video_direction_rms": direction_rms, "dmd_nonfinite_update": nonfinite}
            return self.official_pdmd_weight * loss
        loss, normalizer, direction_rms = dmd_loss_with_stats(
            latents,
            fake_x0,
            teacher_x0,
        )
        self._last_dmd_metrics = {
            "dmd_normalizer": normalizer,
            "dmd_direction_rms": direction_rms,
        }
        return loss

    def dmd_metrics(self):
        return self._last_dmd_metrics

    def add_noise(self, scheduler, latents, noise, sigma):
        if scheduler is None and self.official_pdmd:
            scale = expand_sigma(torch.as_tensor(sigma, device=latents.device, dtype=torch.float32), latents.ndim)
            return (1 - scale) * latents.float() + scale * noise.float()
        return super().add_noise(scheduler, latents, noise, sigma)

    def regression_loss(self, prediction, target):
        if self.official_pdmd:
            return self.official_pdmd_weight * official_pdmd_critic_loss(prediction, target, clamp_max=self.official_pdmd_loss_clamp)
        return super().regression_loss(prediction, target)

    def _validate_turbo_model(self):
        if getattr(self.model, "use_causal_transformer", False) or get_sequence_parallel_world_size() != 1:
            raise ValueError("Wan Turbo currently supports non-causal T2V with sequence_parallel.size=1.")
        if self.model.running_dtype != torch.float32:
            raise ValueError("Wan Turbo currently requires running_dtype=fp32 and no autocast.")

    def predict_velocity_at_sigmas(self, latents, sigma, condition):
        """Turbo-only cleanward RF velocity at one physical noise sigma.

        Wan natively predicts noise-minus-clean. Negate it at this explicit
        boundary; the existing DMD interface retains its original convention.
        """
        self._validate_turbo_model()
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32)
        if sigma.numel() != 1:
            raise ValueError("Wan Turbo requires one physical sigma per singleton sample.")
        with torch.autocast(device_type=self.device.type, enabled=False):
            return -self.predict_velocity(latents.float(), sigma.reshape(1), condition)

    @torch.no_grad()
    def extract_turbo_latents(self, sample, latent_shape):
        """Read cached Wan latents or encode the raw video online, never write."""
        self._validate_turbo_model()
        with torch.autocast(device_type=self.device.type, enabled=False):
            latents = self.model.encode_to_latent(sample).to(device=self.device, dtype=torch.float32)
        if tuple(latents.shape) != tuple(latent_shape):
            raise ValueError(f"Wan Turbo clean latent shape {tuple(latents.shape)} differs from expected {tuple(latent_shape)}.")
        if not bool(torch.isfinite(latents).all()):
            raise ValueError("Wan Turbo clean latents contain nonfinite values.")
        return latents

    @torch.no_grad()
    def predict_velocity_jvp(self, latents, sigma, condition, latent_tangent, sigma_tangent):
        """Detached exact input/time JVP, using FP32 attention throughout."""
        from .wan_turbo_jvp import exact_wan_jvp_attention

        self._validate_turbo_model()
        if latents.shape != latent_tangent.shape:
            raise ValueError("Wan Turbo latent tangent must match the primal shape.")
        sigma = torch.as_tensor(sigma, device=self.device, dtype=torch.float32).reshape(-1)
        sigma_tangent = torch.as_tensor(sigma_tangent, device=self.device, dtype=torch.float32).reshape(-1)
        if sigma.numel() != 1 or sigma_tangent.numel() != 1:
            raise ValueError("Wan Turbo sigma and sigma tangent must be scalars.")
        with exact_wan_jvp_attention(), forward_ad.dual_level():
            state = forward_ad.make_dual(latents.detach().float(), latent_tangent.detach().float())
            time = forward_ad.make_dual(sigma.detach(), sigma_tangent.detach())
            prediction = self.predict_velocity_at_sigmas(state, time, condition)
            primal, tangent = forward_ad.unpack_dual(prediction)
            if tangent is None:
                raise RuntimeError("Wan input/time JVP was lost; no finite-difference fallback is allowed.")
            return primal.detach(), tangent.detach()

    def extra_checkpoint_metadata(self):
        metadata = {"wan_distribution_matching": {"projected_dmd": self.projected_dmd}}
        if self.official_pdmd:
            metadata["wan_official_pdmd_loss"] = {
                "schema_version": 1,
                "video_loss_weight": self.official_pdmd_weight,
                "normalizer_floor": self.official_pdmd_normalizer_floor,
                "projection_epsilon": self.official_pdmd_projection_epsilon,
                "loss_clamp": self.official_pdmd_loss_clamp,
                "mse_scale": 1.0,
                "nonfinite_update": "zero",
                "velocity_convention": "noise_minus_clean",
            }
        return metadata

    def legacy_extra_checkpoint_metadata(self):
        return {"wan_distribution_matching": {"projected_dmd": False}}

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
