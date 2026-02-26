import torch
from loguru import logger

from lightx2v.models.schedulers.hunyuan_video.scheduler import HunyuanVideo15Scheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WorldPlayARScheduler(HunyuanVideo15Scheduler):
    """
    Flow-match scheduler for WorldPlay AR (Autoregressive) model.

    Key differences from WorldPlayDistillScheduler:
    - Supports autoregressive chunk-based generation
    - Manages memory window for long video generation
    - No distill-specific timestep schedule

    Extends HunyuanVideo15Scheduler with:
    - Support for chunk-based AR generation
    - Action and camera pose conditioning support
    - Memory window selection for efficient generation
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

        # AR generation parameters
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)
        self.model_type = config.get("model_type", "ar")

        # Memory window for AR generation
        self.memory_window_size = config.get("memory_window_size", 8)
        self.select_window_out_flag = config.get("select_window_out_flag", True)

        # Camera/action conditioning
        self.viewmats = None
        self.Ks = None
        self.action = None

        # Per-token vec flag
        self.vec_is_per_token = False

        # Chunk tracking
        self.chunk_idx = 0
        self.total_chunks = 1

        # Generated chunks storage
        self.generated_chunks = []

    def set_timesteps(self, num_inference_steps, device, shift):
        """Compute timestep schedule for AR model.

        Uses standard flow-matching schedule with shift.
        Follows the same pattern as HunyuanVideo15Scheduler.
        """
        # Create sigmas from 1 to 0
        sigmas = torch.linspace(1, 0, num_inference_steps + 1, dtype=torch.float32)

        # Apply timestep shift (same as parent sd3_time_shift)
        if shift != 1.0:
            sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        # timesteps should exclude the final sigma (0)
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(dtype=torch.float32, device=device)

        logger.info(f"[WorldPlayARScheduler] Timesteps (first 5): {self.timesteps[:5].tolist()}")
        logger.info(f"[WorldPlayARScheduler] Sigmas (first 5): {self.sigmas[:5].tolist()}")

    def step_post(self):
        """Euler step for flow matching.

        Same as parent HunyuanVideo15Scheduler.
        dt = sigma_next - sigma
        prev_sample = sample + model_output * dt
        """
        model_output = self.noise_pred.to(torch.float32)
        sample = self.latents.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        self.latents = sample + model_output * dt

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

        # Calculate total chunks for AR generation
        total_frames = latent_shape[1]
        self.total_chunks = (total_frames + self.chunk_latent_frames - 1) // self.chunk_latent_frames
        self.chunk_idx = 0
        self.generated_chunks = []

    def prepare_chunk(self, chunk_idx):
        """
        Prepare scheduler state for a specific chunk.

        Args:
            chunk_idx: Index of the chunk to prepare

        Returns:
            Tuple of (chunk_latents, chunk_viewmats, chunk_Ks, chunk_action)
        """
        self.chunk_idx = chunk_idx

        # Calculate frame range for this chunk
        start_frame = chunk_idx * self.chunk_latent_frames
        end_frame = min(start_frame + self.chunk_latent_frames, self.latents.shape[2])

        # Extract chunk latents
        chunk_latents = self.latents[:, :, start_frame:end_frame, :, :]

        # Extract chunk pose data if available
        chunk_viewmats = None
        chunk_Ks = None
        chunk_action = None

        if self.viewmats is not None:
            chunk_viewmats = self.viewmats[:, start_frame:end_frame]
        if self.Ks is not None:
            chunk_Ks = self.Ks[:, start_frame:end_frame]
        if self.action is not None:
            chunk_action = self.action[:, start_frame:end_frame]

        return chunk_latents, chunk_viewmats, chunk_Ks, chunk_action

    def update_chunk_latents(self, chunk_idx, chunk_latents):
        """
        Update latents with generated chunk.

        Args:
            chunk_idx: Index of the chunk
            chunk_latents: Generated latent tensor for this chunk
        """
        start_frame = chunk_idx * self.chunk_latent_frames
        end_frame = min(start_frame + self.chunk_latent_frames, self.latents.shape[2])

        self.latents[:, :, start_frame:end_frame, :, :] = chunk_latents
        self.generated_chunks.append(chunk_idx)

    def get_memory_window(self, chunk_idx):
        """
        Get memory window indices for current chunk.

        For AR generation, we need to attend to previous chunks
        within the memory window.

        Args:
            chunk_idx: Current chunk index

        Returns:
            Tuple of (start_chunk_idx, end_chunk_idx) for memory window
        """
        if not self.select_window_out_flag:
            return 0, chunk_idx

        # Select memory window
        start_chunk = max(0, chunk_idx - self.memory_window_size + 1)
        end_chunk = chunk_idx

        return start_chunk, end_chunk

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
        self.chunk_idx = 0
        self.total_chunks = 1
        self.generated_chunks = []
