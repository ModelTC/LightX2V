"""
WorldPlay AR Training Pipeline for autoregressive video generation.

This pipeline supports:
- Chunk-based training with memory window
- Camera pose (ProPE) conditioning
- Action conditioning
- I2V masking for conditional generation
- KV cache simulation during training
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class WorldPlayARTrainingPipeline:
    """
    Training pipeline for WorldPlay AR model.

    Implements chunk-based training with memory window selection
    for efficient autoregressive video generation training.

    Args:
        model: WorldPlay AR model
        config: Training configuration
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        device: Training device
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.device = device

        # Training parameters
        self.chunk_latent_num = config.get("chunk_latent_num", 4)
        self.memory_window_size = config.get("memory_window_size", 8)
        self.select_window_out_flag = config.get("select_window_out_flag", True)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # Loss weights
        self.flow_loss_weight = config.get("flow_loss_weight", 1.0)

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Logging
        self.log_interval = config.get("log_interval", 100)
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Dictionary containing:
                - video: Video tensor [B, T, C, H, W]
                - w2c: World-to-camera matrices [B, T, 4, 4]
                - intrinsic: Camera intrinsics [B, T, 3, 3]
                - action: Action labels [B, T]
                - i2v_mask: I2V conditioning mask [B, T]
                - caption: List of text captions

        Returns:
            Dictionary of loss values
        """
        self.model.train()

        # Move batch to device
        video = batch["video"].to(self.device)
        w2c = batch["w2c"].to(self.device)
        intrinsic = batch["intrinsic"].to(self.device)
        action = batch["action"].to(self.device)
        i2v_mask = batch["i2v_mask"].to(self.device)

        B, T, C, H, W = video.shape

        # Sample timestep
        timestep = self._sample_timestep(B)

        # Add noise to video
        noise = torch.randn_like(video)
        noisy_video = self._add_noise(video, noise, timestep)

        # Prepare chunk training
        total_loss = 0.0
        num_chunks = (T + self.chunk_latent_num - 1) // self.chunk_latent_num

        for chunk_idx in range(num_chunks):
            # Get chunk boundaries
            start_frame = chunk_idx * self.chunk_latent_num
            end_frame = min(start_frame + self.chunk_latent_num, T)

            # Get memory window
            mem_start, mem_end = self._get_memory_window(chunk_idx, T)

            # Extract chunk data
            chunk_video = noisy_video[:, start_frame:end_frame]
            chunk_w2c = w2c[:, start_frame:end_frame]
            chunk_intrinsic = intrinsic[:, start_frame:end_frame]
            chunk_action = action[:, start_frame:end_frame]
            chunk_mask = i2v_mask[:, start_frame:end_frame]

            # Extract memory context
            if mem_start < start_frame:
                mem_video = video[:, mem_start:start_frame]
                mem_w2c = w2c[:, mem_start:start_frame]
                mem_intrinsic = intrinsic[:, mem_start:start_frame]
            else:
                mem_video = None
                mem_w2c = None
                mem_intrinsic = None

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_flow = self._forward_chunk(
                    chunk_video=chunk_video,
                    chunk_w2c=chunk_w2c,
                    chunk_intrinsic=chunk_intrinsic,
                    chunk_action=chunk_action,
                    chunk_mask=chunk_mask,
                    mem_video=mem_video,
                    mem_w2c=mem_w2c,
                    mem_intrinsic=mem_intrinsic,
                    timestep=timestep,
                    captions=batch["caption"],
                )

                # Compute loss
                target_flow = noise[:, start_frame:end_frame] - video[:, start_frame:end_frame]
                loss = self._compute_loss(pred_flow, target_flow, chunk_mask)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Optimizer step
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.global_step += 1

        return {
            "loss": total_loss * self.gradient_accumulation_steps,
            "flow_loss": total_loss * self.gradient_accumulation_steps,
        }

    def _sample_timestep(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        timesteps = torch.rand(batch_size, device=self.device)
        return timesteps

    def _add_noise(self, video: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Add noise to video using flow matching schedule."""
        # Expand timestep for broadcasting
        t = timestep.view(-1, 1, 1, 1, 1)

        # Linear interpolation (flow matching)
        noisy_video = (1 - t) * video + t * noise

        return noisy_video

    def _get_memory_window(self, chunk_idx: int, total_frames: int) -> tuple:
        """Get memory window boundaries for chunk."""
        if not self.select_window_out_flag:
            return 0, chunk_idx * self.chunk_latent_num

        # Calculate window boundaries
        chunk_start = chunk_idx * self.chunk_latent_num
        window_frames = self.memory_window_size * self.chunk_latent_num

        mem_start = max(0, chunk_start - window_frames)
        mem_end = chunk_start

        return mem_start, mem_end

    def _forward_chunk(
        self,
        chunk_video: torch.Tensor,
        chunk_w2c: torch.Tensor,
        chunk_intrinsic: torch.Tensor,
        chunk_action: torch.Tensor,
        chunk_mask: torch.Tensor,
        mem_video: Optional[torch.Tensor],
        mem_w2c: Optional[torch.Tensor],
        mem_intrinsic: Optional[torch.Tensor],
        timestep: torch.Tensor,
        captions: list,
    ) -> torch.Tensor:
        """
        Forward pass for a single chunk.

        This simulates the AR generation process during training.
        """
        # Prepare inputs
        inputs = {
            "video": chunk_video,
            "w2c": chunk_w2c,
            "intrinsic": chunk_intrinsic,
            "action": chunk_action,
            "mask": chunk_mask,
            "timestep": timestep,
            "captions": captions,
        }

        # Add memory context if available
        if mem_video is not None:
            inputs["mem_video"] = mem_video
            inputs["mem_w2c"] = mem_w2c
            inputs["mem_intrinsic"] = mem_intrinsic

        # Forward through model
        output = self.model.forward_training(inputs)

        return output

    def _compute_loss(self, pred_flow: torch.Tensor, target_flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute flow matching loss with I2V masking."""
        # Expand mask for broadcasting
        mask = mask.view(mask.shape[0], mask.shape[1], 1, 1, 1)

        # MSE loss with masking
        loss = F.mse_loss(pred_flow * mask, target_flow * mask, reduction="sum")
        loss = loss / (mask.sum() * pred_flow.shape[2] * pred_flow.shape[3] * pred_flow.shape[4])

        return loss * self.flow_loss_weight

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path}")


def create_ar_training_pipeline(
    model: nn.Module,
    config: Dict[str, Any],
    device: str = "cuda",
) -> WorldPlayARTrainingPipeline:
    """
    Create AR training pipeline with optimizer and scheduler.

    Args:
        model: WorldPlay AR model
        config: Training configuration
        device: Training device

    Returns:
        Configured training pipeline
    """
    # Create optimizer
    lr = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Create learning rate scheduler
    warmup_steps = config.get("warmup_steps", 1000)
    total_steps = config.get("total_steps", 100000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create pipeline
    pipeline = WorldPlayARTrainingPipeline(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    return pipeline
