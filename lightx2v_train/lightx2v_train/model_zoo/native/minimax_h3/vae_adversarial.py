"""Adversarial components for MiniMax-H3 video VAE distillation."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


PatchLogits: TypeAlias = tuple[torch.Tensor, ...]


def _checkpoint_contexts(*modules):
    spectral_modules = [
        layer for module in modules for layer in module.modules() if hasattr(layer, "weight_u")
    ]
    forward_buffers = []

    @contextmanager
    def forward_context():
        yield
        forward_buffers.extend(
            (layer.weight_u.clone(), layer.weight_v.clone()) for layer in spectral_modules
        )

    @contextmanager
    def recompute_context():
        states = [(layer.training, layer.weight_u, layer.weight_v) for layer in spectral_modules]
        try:
            # Recompute with the same normalized weights, without another power iteration.
            for layer, (weight_u, weight_v) in zip(spectral_modules, forward_buffers, strict=True):
                layer.training = False
                layer.weight_u = weight_u
                layer.weight_v = weight_v
            yield
        finally:
            for layer, (training, weight_u, weight_v) in zip(spectral_modules, states, strict=True):
                layer.training = training
                layer.weight_u = weight_u
                layer.weight_v = weight_v

    return forward_context(), recompute_context()


def _patch_stage(hidden_states, block, head):
    hidden_states = block(hidden_states)
    return hidden_states, head(hidden_states)


def _run_patch_stage(hidden_states, block, head, gradient_checkpointing):
    if gradient_checkpointing and torch.is_grad_enabled():
        return checkpoint(
            partial(_patch_stage, block=block, head=head),
            hidden_states,
            use_reentrant=False,
            context_fn=partial(_checkpoint_contexts, block, head),
            preserve_rng_state=False,
        )
    return _patch_stage(hidden_states, block, head)


@dataclass(frozen=True, slots=True)
class DiscriminatorLoss:
    total: torch.Tensor
    real: torch.Tensor
    fake: torch.Tensor


@dataclass(frozen=True, slots=True)
class CorrectionTarget:
    target: torch.Tensor
    correction: torch.Tensor
    feature_distance: torch.Tensor
    raw_correction_rms: torch.Tensor
    branch_feature_distances: tuple[torch.Tensor, ...]


def _convolution(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    *,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    bias: bool = True,
    spectral_normalization: bool,
) -> nn.Conv3d:
    convolution = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    if spectral_normalization:
        convolution = nn.utils.spectral_norm(convolution)
    return convolution


class _PatchBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_stride: int,
        group_norm_groups: int,
        spectral_normalization: bool,
    ):
        super().__init__()
        groups = min(group_norm_groups, out_channels)
        while out_channels % groups:
            groups -= 1
        self.convolution = _convolution(
            in_channels,
            out_channels,
            (3, 4, 4),
            stride=(temporal_stride, 2, 2),
            padding=(1, 1, 1),
            bias=False,
            spectral_normalization=spectral_normalization,
        )
        self.normalization = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        return self.activation(hidden_states)


class LatentConditionedVideoPatchDiscriminator(nn.Module):
    """A lightweight multi-layer PatchGAN over space and short time windows.

    Video and latent tensors follow ``[B, C, T, H, W]``. The caller owns the
    video value convention and must use the same convention for real and fake.
    """

    def __init__(
        self,
        *,
        video_channels: int = 3,
        latent_channels: int = 24,
        condition_channels: int = 8,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        temporal_strides: tuple[int, ...] = (1, 1, 2, 2),
        group_norm_groups: int = 16,
        spectral_normalization: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if len(channel_multipliers) != len(temporal_strides):
            raise ValueError("channel_multipliers and temporal_strides must have equal length.")

        self.latent_projection = _convolution(
            latent_channels,
            condition_channels,
            1,
            spectral_normalization=spectral_normalization,
        )
        blocks = []
        heads = []
        in_channels = video_channels + condition_channels
        for multiplier, temporal_stride in zip(channel_multipliers, temporal_strides, strict=True):
            out_channels = base_channels * multiplier
            blocks.append(
                _PatchBlock(
                    in_channels,
                    out_channels,
                    temporal_stride=temporal_stride,
                    group_norm_groups=group_norm_groups,
                    spectral_normalization=spectral_normalization,
                )
            )
            heads.append(
                _convolution(
                    out_channels,
                    1,
                    1,
                    spectral_normalization=spectral_normalization,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, video: torch.Tensor, latent: torch.Tensor) -> PatchLogits:
        if video.ndim != 5 or latent.ndim != 5:
            raise ValueError("video and latent must be five-dimensional BCTHW tensors.")
        if video.shape[0] != latent.shape[0]:
            raise ValueError("video and latent batch sizes must match.")

        condition = self.latent_projection(latent)
        condition = F.interpolate(
            condition,
            size=video.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        hidden_states = torch.cat((video, condition.to(video.dtype)), dim=1)
        logits = []
        for block, head in zip(self.blocks, self.heads, strict=True):
            hidden_states, value = _run_patch_stage(
                hidden_states, block, head, self.gradient_checkpointing
            )
            logits.append(value)
        return tuple(logits)


def _convolution_2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    bias: bool = True,
    spectral_normalization: bool,
) -> nn.Conv2d:
    convolution = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    if spectral_normalization:
        convolution = nn.utils.spectral_norm(convolution)
    return convolution


class _FramePatchBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        spatial_stride: int,
        group_norm_groups: int,
        spectral_normalization: bool,
    ):
        super().__init__()
        groups = min(group_norm_groups, out_channels)
        while out_channels % groups:
            groups -= 1
        kernel_size = 4 if spatial_stride == 2 else 3
        self.convolution = _convolution_2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=spatial_stride,
            padding=1,
            bias=False,
            spectral_normalization=spectral_normalization,
        )
        self.normalization = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        return self.activation(hidden_states)


class LatentConditionedFramePatchDiscriminator(nn.Module):
    """A spatial PatchGAN that treats video frames as independent images."""

    def __init__(
        self,
        *,
        video_channels: int = 3,
        latent_channels: int = 24,
        condition_channels: int = 16,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 8, 8),
        spatial_strides: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        group_norm_groups: int = 32,
        spectral_normalization: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        if len(channel_multipliers) != len(spatial_strides):
            raise ValueError("channel_multipliers and spatial_strides must have equal length.")

        self.latent_projection = _convolution(
            latent_channels,
            condition_channels,
            1,
            spectral_normalization=spectral_normalization,
        )
        blocks = []
        heads = []
        in_channels = video_channels + condition_channels
        for multiplier, spatial_stride in zip(channel_multipliers, spatial_strides, strict=True):
            out_channels = base_channels * multiplier
            blocks.append(
                _FramePatchBlock(
                    in_channels,
                    out_channels,
                    spatial_stride=spatial_stride,
                    group_norm_groups=group_norm_groups,
                    spectral_normalization=spectral_normalization,
                )
            )
            heads.append(
                _convolution_2d(
                    out_channels,
                    1,
                    1,
                    spectral_normalization=spectral_normalization,
                )
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, video: torch.Tensor, latent: torch.Tensor) -> PatchLogits:
        if video.ndim != 5 or latent.ndim != 5:
            raise ValueError("video and latent must be five-dimensional BCTHW tensors.")
        if video.shape[0] != latent.shape[0]:
            raise ValueError("video and latent batch sizes must match.")

        condition = self.latent_projection(latent)
        condition = F.interpolate(
            condition,
            size=video.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        hidden_states = torch.cat((video, condition.to(video.dtype)), dim=1)
        batch, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            channels,
            height,
            width,
        )

        logits = []
        for block, head in zip(self.blocks, self.heads, strict=True):
            hidden_states, value = _run_patch_stage(
                hidden_states, block, head, self.gradient_checkpointing
            )
            logits.append(
                value.reshape(batch, frames, 1, value.shape[-2], value.shape[-1]).permute(0, 2, 1, 3, 4)
            )
        return tuple(logits)


def _lsgan_target_loss(logits: PatchLogits, target: float) -> torch.Tensor:
    losses = [(value.float() - target).square().mean() for value in logits]
    return torch.stack(losses).mean()


def discriminator_lsgan_loss(
    real_logits: PatchLogits,
    fake_logits: PatchLogits,
    *,
    real_target: float = 1.0,
    fake_target: float = -1.0,
) -> DiscriminatorLoss:
    """Return a symmetric multi-layer least-squares discriminator loss."""

    real = _lsgan_target_loss(real_logits, real_target)
    fake = _lsgan_target_loss(fake_logits, fake_target)
    return DiscriminatorLoss(total=0.5 * (real + fake), real=real, fake=fake)


def generator_lsgan_loss(fake_logits: PatchLogits, *, real_target: float = 1.0) -> torch.Tensor:
    """Return the direct generator-side least-squares adversarial loss."""

    return _lsgan_target_loss(fake_logits, real_target)


def _per_sample_feature_distance(real_logits: PatchLogits, fake_logits: PatchLogits) -> torch.Tensor:
    distances = [
        (fake.float() - real.float()).square().flatten(1).mean(dim=1)
        for real, fake in zip(real_logits, fake_logits, strict=True)
    ]
    return torch.stack(distances, dim=1).mean(dim=1)


def make_seraena_correction_target(
    discriminator: nn.Module,
    real_video: torch.Tensor,
    fake_video: torch.Tensor,
    latent: torch.Tensor,
    *,
    correction_scale: float = 0.05,
    normalized_clamp: float = 4.0,
    eps: float = 1e-6,
) -> CorrectionTarget:
    """Build a detached Seraena-style target from discriminator score gradients.

    Each sample's correction is RMS-normalized independently, clipped in the
    normalized domain, then multiplied by ``correction_scale``.
    """

    return make_ensemble_seraena_correction_target(
        ((discriminator, 1.0),),
        real_video,
        fake_video,
        latent,
        correction_scale=correction_scale,
        normalized_clamp=normalized_clamp,
        eps=eps,
    )


def make_ensemble_seraena_correction_target(
    discriminator_branches: tuple[tuple[nn.Module, float], ...],
    real_video: torch.Tensor,
    fake_video: torch.Tensor,
    latent: torch.Tensor,
    *,
    correction_scale: float = 0.05,
    normalized_clamp: float = 4.0,
    eps: float = 1e-6,
) -> CorrectionTarget:
    """Build one correction from the weighted feature distances of all branches."""

    total_weight = sum(weight for _, weight in discriminator_branches)
    if total_weight <= 0:
        raise ValueError("Discriminator branch weights must sum to a positive value.")

    real_video = real_video.detach()
    latent = latent.detach()
    fake_leaf = fake_video.detach().requires_grad_(True)

    gradient = torch.zeros_like(fake_leaf)
    branch_feature_distances = []
    for discriminator, weight in discriminator_branches:
        with torch.no_grad():
            real_logits = discriminator(real_video, latent)
        with torch.enable_grad():
            fake_logits = discriminator(fake_leaf, latent)
            distance = _per_sample_feature_distance(real_logits, fake_logits)
            # Release each branch's activations before constructing the next one.
            gradient.add_(torch.autograd.grad(distance.sum() * (weight / total_weight), fake_leaf)[0])
        branch_feature_distances.append(distance.detach())
    feature_distance = torch.stack(
        [
            distance * weight
            for distance, (_, weight) in zip(branch_feature_distances, discriminator_branches, strict=True)
        ]
    ).sum(dim=0) / total_weight

    raw_correction = -gradient
    raw_rms = raw_correction.float().square().flatten(1).mean(dim=1).sqrt()
    broadcast_shape = (raw_correction.shape[0],) + (1,) * (raw_correction.ndim - 1)
    correction = raw_correction.float() / raw_rms.clamp_min(eps).view(broadcast_shape)
    correction = correction.clamp(-normalized_clamp, normalized_clamp) * correction_scale
    correction = correction.to(fake_video.dtype).detach()
    target = (fake_video.detach() + correction).detach()
    return CorrectionTarget(
        target=target,
        correction=correction,
        feature_distance=feature_distance.detach(),
        raw_correction_rms=raw_rms.detach(),
        branch_feature_distances=tuple(distance.detach() for distance in branch_feature_distances),
    )


__all__ = [
    "CorrectionTarget",
    "DiscriminatorLoss",
    "LatentConditionedFramePatchDiscriminator",
    "LatentConditionedVideoPatchDiscriminator",
    "PatchLogits",
    "discriminator_lsgan_loss",
    "generator_lsgan_loss",
    "make_ensemble_seraena_correction_target",
    "make_seraena_correction_target",
]
