"""Official PDMD's single-image objective and cached-text conditioning."""

import math

import torch

from lightx2v_train.model_zoo.capability_adapters.common import GenericDistributionMatchingCapability
from lightx2v_train.model_zoo.capability_adapters.latent_geometry import SpatialLatentGeometry
from lightx2v_train.trainers.dmd.math import expand_sigma, official_pdmd_critic_loss, official_pdmd_loss_with_stats


class MiniMaxH3ImageDistributionMatchingCapability(GenericDistributionMatchingCapability):
    def __init__(self, model):
        super().__init__(model, latent_geometry=SpatialLatentGeometry(channels_path="latent_channels", spatial_scale_path="vae_scale_factor", temporal_size=1))
        options = model.config["model"].get("capabilities", {}).get("distribution_matching", {})
        self.official_pdmd = bool(options.get("official_pdmd", False))
        self.projected_dmd = bool(options.get("projected_dmd", True))
        if model.config.get("training", {}).get("method") != "pdd" and (not self.official_pdmd or not self.projected_dmd):
            raise ValueError("H3-Image training currently requires official_pdmd=true and projected_dmd=true.")
        self.weight = float(options.get("image_loss_weight", 1.0))
        self.normalizer_floor = float(options.get("official_pdmd_normalizer_floor", 1e-5))
        self.projection_epsilon = float(options.get("official_pdmd_projection_epsilon", 1e-8))
        self.loss_clamp = float(options.get("official_pdmd_loss_clamp", 5.0))
        if any(not math.isfinite(value) or value <= 0 for value in (self.weight, self.normalizer_floor, self.projection_epsilon, self.loss_clamp)):
            raise ValueError("H3-Image PDMD loss settings must be finite and positive.")
        self._last_metrics = {}

    @property
    def default_lora_target_modules(self):
        return ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "adaln_proj.linear", "norm_out.linear")

    def encode_conditions(self, batch, negative_prompt, guidance_scale, broadcast):
        if float(guidance_scale) != 1:
            raise ValueError("The released H3-Image model uses guidance_scale=1 without an unconditional branch.")
        condition, negative = super().encode_conditions(batch, negative_prompt, guidance_scale, broadcast)
        return self.model.prepare_text_condition(condition), negative

    def add_noise(self, scheduler, latents, noise, sigma):
        del scheduler
        scale = expand_sigma(torch.as_tensor(sigma, device=latents.device, dtype=torch.float32), latents.ndim)
        return (1 - scale) * latents.float() + scale * noise.float()

    def dmd_loss(self, latents, fake_x0, teacher_x0):
        loss, normalizer, direction_rms, nonfinite = official_pdmd_loss_with_stats(
            latents, fake_x0, teacher_x0, projected=True, normalizer_floor=self.normalizer_floor, projection_epsilon=self.projection_epsilon, clamp_max=self.loss_clamp
        )
        self._last_metrics = {"dmd_image": loss.detach(), "dmd_normalizer": normalizer, "dmd_direction_rms": direction_rms, "dmd_nonfinite_update": nonfinite}
        return self.weight * loss

    def dmd_metrics(self):
        return self._last_metrics

    def regression_loss(self, prediction, target):
        return self.weight * official_pdmd_critic_loss(prediction, target, clamp_max=self.loss_clamp)

    def predict_dmad_features(self, latents, sigma, condition, feature_block=49):
        return self.model.predict_dmad_features(latents, sigma, condition, feature_block=feature_block)

    def extract_turbo_latents(self, sample, latent_shape):
        """Load a cached, normalized GAN target without allocating the VAE."""
        metadata = sample["meta"]
        if metadata.get("vae_normalized") is not True:
            raise ValueError("H3-Image GAN real latents must be marked vae_normalized=true.")
        payload = torch.load(metadata["image_latent_path"], map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("H3-Image GAN real cache must contain latents and their sample identity.")
        if payload.get("vae_normalized", True) is not True:
            raise ValueError("H3-Image GAN payload is not in normalized model latent space.")
        for name in ("real_sample_key", "real_cache_fingerprint"):
            if not metadata.get(name) or payload.get(name) != metadata[name]:
                raise ValueError(f"H3-Image GAN real latent {name} does not match its manifest.")
        values = payload.get("latents")
        if not torch.is_tensor(values) or values.dtype != torch.float32:
            raise ValueError("H3-Image GAN real latents must be an FP32 tensor.")
        if values.ndim == 4:
            values = values.unsqueeze(0)
        if tuple(values.shape) != tuple(latent_shape) or not bool(torch.isfinite(values).all()):
            raise ValueError(f"H3-Image GAN real latent shape/values do not match {tuple(latent_shape)}.")
        return values.to(device=self.device)

    def extra_checkpoint_metadata(self):
        return {
            "h3_image_pdmd": {
                "schema_version": 1,
                **self.model.pdd_schedule.metadata(),
                "velocity_convention": "noise_minus_clean",
                "transformer_timestep": "1-sigma",
                "image_loss_weight": self.weight,
                "normalizer_floor": self.normalizer_floor,
                "projection_epsilon": self.projection_epsilon,
                "loss_clamp": self.loss_clamp,
            }
        }
