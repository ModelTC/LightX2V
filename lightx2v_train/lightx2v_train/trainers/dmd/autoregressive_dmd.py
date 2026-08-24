from lightx2v_train.model_capabilities import (
    AutoregressiveDistributionMatchingCapability,
    AutoregressiveRolloutContext,
)
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .trainer import DmdTrainer


@TRAINER_REGISTER("autoregressive_dmd")
class AutoregressiveDmdTrainer(DmdTrainer):
    trainer_name = "autoregressive_dmd"
    required_capabilities = (
        *DmdTrainer.required_capabilities,
        AutoregressiveDistributionMatchingCapability,
    )
    supports_diversity_loss = False
    supports_real_data_fake = False

    def __init__(self, config):
        super().__init__(config)
        self.num_frame_per_chunk = int(
            self.dmd_config.get(
                "num_frame_per_chunk",
                self.model_config.get("num_frame_per_chunk", 3),
            )
        )
        self.same_step_across_blocks = bool(self.dmd_config.get("same_step_across_blocks", True))
        self.context_noise = float(self.dmd_config.get("context_noise", 0.0))
        self.sequence_parallel_cache = bool(self.dmd_config.get("sp_cache", False))

    def set_model(self, model):
        super().set_model(model)
        self.autoregressive = model.capabilities.require(AutoregressiveDistributionMatchingCapability)

    def run_back_simulation(
        self,
        condition,
        latent_shape,
        grad_enabled,
        xt=None,
    ):
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        self.scheduler.set_timesteps(
            self.num_inference_steps,
            sigmas=[float(sigma) for sigma in self.denoising_sigmas.detach().cpu()],
            latent_hw=self.student.latent_hw(latent_shape),
            device=self.student.device,
        )
        generated, exit_indices = self.autoregressive.rollout(
            condition,
            latent_shape,
            xt,
            AutoregressiveRolloutContext(
                denoising_steps=self.denoising_steps,
                denoising_scheduler=self.denoising_scheduler,
                trajectory_scheduler=self.scheduler,
                running_dtype=self.latent_dtype,
                frames_per_chunk=self.num_frame_per_chunk,
                same_step_across_blocks=self.same_step_across_blocks,
                context_noise=self.context_noise,
                sequence_parallel_cache=self.sequence_parallel_cache,
                grad_enabled=grad_enabled,
            ),
        )
        return generated, *self._denoised_timestep_window(exit_indices)

    def _denoised_timestep_window(self, exit_indices):
        if not self.same_step_across_blocks:
            return None, None
        return super()._denoised_timestep_window(int(exit_indices[0]))
