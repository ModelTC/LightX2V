"""Thin DMD trainer specialization for MiniMax-H3 T2AV."""

import torch
from loguru import logger

from lightx2v_train.runtime.sequence_parallel import broadcast_sequence_parallel_value
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .trainer import DmdTrainer


@TRAINER_REGISTER("minimax_h3_t2av_dmd")
class MiniMaxH3T2AVDmdTrainer(DmdTrainer):
    """Configure generic DMD for H3's supported training feature set.

    Audio/video tensor math and the clean-ward velocity convention live in
    ``MiniMaxH3DistillationCapability``. This class only validates algorithm
    options and supplies H3's continuous, unshifted score-sigma sampling.
    """

    trainer_name = "minimax_h3_t2av_dmd"
    default_negative_prompt = ""
    supports_cdm = False
    supports_ida = False
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        super().__init__(config)
        model_name = self.model_config.get("name")
        if model_name != "minimax_h3_t2av":
            raise ValueError(f"minimax_h3_t2av_dmd requires model.name=minimax_h3_t2av, got {model_name!r}.")
        if self.guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 is guidance-distilled and has no unconditional branch; set training.teacher.guidance_scale to 1.0.")
        if self.warp_denoising_step:
            raise ValueError("MiniMax-H3 applies separate video/audio flow shifts in its model capability; set training.dmd.warp_denoising_step to false.")

        h3_config = self.training_config.get("minimax_h3", {})
        if not isinstance(h3_config, dict):
            raise TypeError("training.minimax_h3 must be a mapping.")
        dtype_name = str(h3_config.get("latent_dtype", "fp32")).lower()
        dtype_aliases = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_name not in dtype_aliases:
            supported = ", ".join(sorted(dtype_aliases))
            raise ValueError(f"Unsupported training.minimax_h3.latent_dtype={dtype_name!r}; expected one of: {supported}.")
        self.latent_dtype = dtype_aliases[dtype_name]
        # Generic DMD uses running_dtype for rollout states and score sigmas.
        # H3 defaults to fp32 latents even when prompt embeddings are bf16.
        self.running_dtype = self.latent_dtype

        self.renoise_sigma_min = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        self.renoise_sigma_max = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        if not 0.0 <= self.renoise_sigma_min < self.renoise_sigma_max <= 1.0:
            raise ValueError(f"MiniMax-H3 renoise sigma range must satisfy 0 <= min < max <= 1, got [{self.renoise_sigma_min}, {self.renoise_sigma_max}].")
        logger.info(
            "[train] MiniMax-H3 generic DMD latent_dtype={} renoise_sigma=[{}, {}]",
            dtype_name,
            self.renoise_sigma_min,
            self.renoise_sigma_max,
        )

    def _sample_score_sigma(
        self,
        denoised_timestep_from,
        denoised_timestep_to,
        device,
        dtype,
    ):
        del denoised_timestep_from, denoised_timestep_to
        sigma = torch.empty((1,), device=device, dtype=torch.float32).uniform_(
            self.renoise_sigma_min,
            self.renoise_sigma_max,
        )
        return broadcast_sequence_parallel_value(sigma).to(dtype=dtype)
