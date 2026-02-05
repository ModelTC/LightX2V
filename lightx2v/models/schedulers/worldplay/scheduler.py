import torch
from loguru import logger

from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WorldPlayDistillScheduler(HunyuanVideo15Scheduler):
    """
    Flow-match scheduler for WorldPlay distill model.

    Extends HunyuanVideo15Scheduler with:
    - Support for few-step inference (4 steps by default)
    - Autoregressive chunk-based generation
    - Action and camera pose conditioning support
    """

    def __init__(self, config):
        super().__init__(config)
        # Distill-specific parameters - use exact timesteps from HY-WorldPlay
        # These are the specific timesteps for 4-step distill inference
        self.distill_timesteps = [1000.0, 960.0, 888.8889, 727.2728, 0.0]
        self.infer_steps = len(self.distill_timesteps) - 1  # 4 steps (5 timesteps including final 0)

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

        # AR generation parameters
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)
        self.model_type = config.get("model_type", "ar")

        # Camera/action conditioning
        self.viewmats = None
        self.Ks = None
        self.action = None

        # Per-token vec flag (set by pre_infer when action conditioning is active)
        self.vec_is_per_token = False

    def set_timesteps(self, num_inference_steps, device, shift):
        """Compute distill timestep schedule.

        For distill model, we use the exact timesteps from HY-WorldPlay:
        [1000.0, 960.0, 888.8889, 727.2728, 0.0]

        Sigmas are simply timesteps / 1000 (no shift applied for distill).
        """
        # Use exact timesteps from HY-WorldPlay for distill model
        self.timesteps = torch.tensor(self.distill_timesteps, dtype=torch.float32, device=device)

        # Compute sigmas - for distill model, NO shift is applied
        # sigmas = timesteps / 1000.0 directly
        self.sigmas = (self.timesteps / self.num_train_timesteps).to("cpu")

        logger.info(f"[WorldPlayDistillScheduler] Timesteps: {self.timesteps.tolist()}")
        logger.info(f"[WorldPlayDistillScheduler] Sigmas: {self.sigmas.tolist()}")

    def step_post(self):
        """Euler step for flow matching with distill schedule.

        Matches HY-WorldPlay implementation exactly:
        dt = sigma_next - sigma
        prev_sample = sample + model_output * dt
        """
        flow_pred = self.noise_pred.to(torch.float32)

        # Get current and next sigma
        sigma = self.sigmas[self.step_index].item()
        sigma_next = self.sigmas[self.step_index + 1].item()

        # Euler step: prev_sample = sample + (sigma_next - sigma) * model_output
        dt = sigma_next - sigma

        prev_sample = self.latents.to(torch.float32) + dt * flow_pred

        self.latents = prev_sample.to(self.latents.dtype)

    def prepare(self, seed, latent_shape, image_encoder_output=None, pose_output=None):
        """
        Initialize latents and timesteps with optional pose conditioning.

        Args:
            seed: Random seed for latent initialization
            latent_shape: Shape of latent tensor [C, T, H, W]
            image_encoder_output: Dict with siglip_output, siglip_mask, cond_latents
            pose_output: Dict with viewmats, Ks, action tensors (optional)
        """
        self.prepare_latents(seed, latent_shape, dtype=torch.bfloat16)
        self.set_timesteps(self.infer_steps, device=AI_DEVICE, shift=self.sample_shift)
        self.multitask_mask = self.get_task_mask(self.config["task"], latent_shape[-3])

        cond_latents = image_encoder_output.get("cond_latents") if image_encoder_output else None
        self.cond_latents_concat, self.mask_concat = self._prepare_cond_latents_and_mask(self.config["task"], cond_latents, self.latents, self.multitask_mask, self.reorg_token)
        self.cos_sin = self.prepare_cos_sin((latent_shape[1], latent_shape[2], latent_shape[3]))

        # Store pose conditioning if provided
        if pose_output is not None:
            self.viewmats = pose_output.get("viewmats")
            self.Ks = pose_output.get("Ks")
            self.action = pose_output.get("action")

    def get_chunk_timesteps(self, chunk_idx):
        """Get timesteps for a specific chunk in AR generation."""
        return self.timesteps

    def get_chunk_sigmas(self, chunk_idx):
        """Get sigmas for a specific chunk in AR generation."""
        return self.sigmas

    def clear(self):
        """Cleanup scheduler state."""
        super().clear()
        self.viewmats = None
        self.Ks = None
        self.action = None
        self.vec_is_per_token = False
