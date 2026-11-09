"""Replicated FP32 video classifier for the shared H3 fake-score backbone.

DMD2's convolutional image classifier does not fit H3's packed video tokens.
This explicit adaptation retains four ordered pooled token bins before an MLP.
Only target video tokens enter the head; audio and references condition the DiT.
"""

import copy
import math
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn


class H3GanHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=1024, temporal_bins=4):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim), ("temporal_bins", temporal_bins)):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"H3 GAN {name} must be a positive integer.")
        self.feature_dim = int(feature_dim)
        self.temporal_bins = int(temporal_bins)
        self.classifier = nn.Sequential(nn.Linear(self.feature_dim * self.temporal_bins, int(hidden_dim)), nn.SiLU(), nn.Linear(int(hidden_dim), 1))

    def forward(self, video_features):
        if video_features.ndim != 3 or video_features.shape[1] == 0 or video_features.shape[2] != self.feature_dim:
            raise ValueError("H3 GAN expects nonempty target video features [B, tokens, feature_dim].")
        # Parameters and arithmetic stay FP32, including inside caller autocast.
        with torch.autocast(device_type=video_features.device.type, enabled=False):
            length = video_features.shape[1]
            # Reduce directly with FP32 accumulation. Casting/normalizing the
            # entire [B, ~37k, 5376] sequence would retain several extra GB per
            # classifier graph. Only these small ordered-bin means become FP32.
            pooled = torch.stack(
                [
                    video_features[:, index * length // self.temporal_bins : ((index + 1) * length + self.temporal_bins - 1) // self.temporal_bins].mean(1, dtype=torch.float32)
                    for index in range(self.temporal_bins)
                ],
                dim=1,
            )
            pooled = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-5)
            return self.classifier(pooled.flatten(1)).squeeze(-1)


def setup_dmd2_gan_head(trainer):
    """Create the head after fake FSDP wrapping, before checkpoint restoration."""
    trainer.dmd2_gan_head = None
    trainer.dmd2_gan_optimizer = None
    trainer.dmd2_gan_lr_scheduler = None
    trainer.dmd2_gan_trainable_params = []
    options = trainer.dmd_config.get("gan", {})
    if not options.get("enabled", False):
        return
    if not trainer.dmd_config.get("official_pdmd", False):
        raise ValueError("H3 GAN requires the official PDMD training loop.")
    backbone = trainer.fake.denoiser()
    block = options.get("feature_block", 49)
    if isinstance(block, bool) or int(block) != block or not 0 <= block < len(backbone.transformer_blocks):
        raise ValueError("H3 GAN feature_block must identify a fake-backbone transformer block.")
    width = int(backbone.config.hidden_size)
    # Do not perturb model/data RNG, and initialize the same replicated head on
    # every rank. Broadcast also protects against process-local RNG differences.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(trainer.config.get("seed", 42)) + 20903)
        head = H3GanHead(width, options.get("head_hidden_dim", 1024), options.get("head_temporal_bins", 4))
    trainer.dmd2_gan_head = head.to(device=trainer.student.device, dtype=torch.float32).eval()
    if dist.is_initialized():
        for value in head.state_dict().values():
            dist.broadcast(value, src=0)
    trainer.dmd2_gan_trainable_params = list(head.parameters())
    # The runtime resolves omitted fake hyperparameters from student defaults.
    # Read those resolved values rather than reapplying AdamW's own defaults.
    fake_defaults = trainer.fake_optimizer.defaults
    optimizer_options = {
        "learning_rate": fake_defaults["lr"],
        "adam_beta1": fake_defaults["betas"][0],
        "adam_beta2": fake_defaults["betas"][1],
        "weight_decay": fake_defaults["weight_decay"],
        "adam_epsilon": fake_defaults["eps"],
    }
    trainer.dmd2_gan_optimizer = trainer._build_optimizer(trainer.dmd2_gan_trainable_params, optimizer_options)
    trainer.dmd2_gan_lr_scheduler = trainer._build_lr_scheduler(
        trainer.dmd2_gan_optimizer,
        num_warmup_steps=0,
        num_training_steps=max(1, trainer.max_train_iters * trainer.fake_update_ratio),
    )
    trainer.dmd2_gan_metadata = {
        "version": 1,
        "head": "fp32_target_video_ordered_pool_mlp",
        "feature_dim": width,
        "options": copy.deepcopy(options),
    }


@contextmanager
def frozen_dmd2_gan_critic(trainer):
    """Retain input gradients while freezing fake/head through G backward.

    The caller must keep backward inside this context: activation checkpointing
    recomputes the fake forward and must observe the same frozen parameters.
    """
    parameters = [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params]
    enabled = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    trainer.fake.set_training(False)
    try:
        yield
    finally:
        for parameter, previous in zip(parameters, enabled):
            parameter.requires_grad_(previous)


def sync_dmd2_gan_head_gradients(trainer):
    """Average replicated head gradients once after microbatch accumulation."""
    if dist.is_initialized():
        for parameter in trainer.dmd2_gan_trainable_params:
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(dist.get_world_size())


def clip_dmd2_gan_grad_norm(trainer, max_norm):
    """Sync and clip fake DTensors plus replicated head as one parameter set.

    The two optimizers have identical hyperparameters and update cadence. This
    common coefficient preserves shared-optimizer clipping without mixing
    ordinary tensors and DTensors inside PyTorch's foreach norm operation.
    """
    max_norm = float(max_norm)
    if max_norm < 0 or math.isnan(max_norm):
        raise ValueError("H3 GAN max_grad_norm must be nonnegative.")
    sync_dmd2_gan_head_gradients(trainer)
    fake_norm = torch.nn.utils.clip_grad_norm_(trainer.fake_trainable_params, float("inf"))
    if hasattr(fake_norm, "full_tensor"):
        fake_norm = fake_norm.full_tensor()
    head_norm = torch.nn.utils.clip_grad_norm_(trainer.dmd2_gan_trainable_params, float("inf"))
    norm = torch.linalg.vector_norm(torch.stack((fake_norm.float(), head_norm.to(fake_norm.device).float())))
    # Do not modify gradients on a nonfinite norm; the training loop rejects it
    # collectively before either optimizer is stepped.
    if bool(torch.isfinite(norm)):
        coefficient = min(1.0, max_norm / (float(norm) + 1e-6))
        for parameter in [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params]:
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
    return norm
