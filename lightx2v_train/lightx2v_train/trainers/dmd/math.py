import math

import torch
import torch.nn.functional as F


def expand_sigma(sigma, ndim):
    """Expand a batch sigma vector across non-batch tensor dimensions."""
    if sigma.ndim == 0:
        sigma = sigma.reshape(1)
    return sigma.reshape(
        sigma.shape[0],
        *([1] * (ndim - 1)),
    )


def euler_step(sample, velocity, sigma, sigma_next):
    """Advance a flow sample between two arbitrary sigma values."""
    sigma = expand_sigma(sigma, sample.ndim)
    sigma_next = expand_sigma(sigma_next, sample.ndim)
    return (sample + (sigma_next - sigma) * velocity).to(sample.dtype)


def velocity_to_x0(sample, velocity, sigma):
    """Project a flow velocity prediction from sigma to zero."""
    return euler_step(
        sample,
        velocity,
        sigma,
        torch.zeros_like(sigma),
    )


def do_cfg(cond_pred, uncond_pred, cfg_scale, cfg_norm):
    """Apply classifier-free guidance with the configured normalization."""
    pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
    if cfg_norm in (None, "none"):
        return pred
    if cfg_norm == "layer_norm":
        cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True)
        pred_norm = torch.norm(pred, dim=-1, keepdim=True)
        return pred * (cond_norm / torch.clamp(pred_norm, min=1e-12))
    if cfg_norm == "scalar":
        cond_norm = torch.norm(cond_pred)
        pred_norm = torch.norm(pred)
        return pred * min(
            1.0,
            (cond_norm / torch.clamp(pred_norm, min=1e-12)).item(),
        )
    raise ValueError(f"Unsupported cfg_norm: {cfg_norm}")


@torch.no_grad()
def project_dmd_direction(direction, residual):
    """Remove each sample's residual-parallel component (PDMD, Eq. 4)."""
    direction = direction.float()
    residual = residual.float()
    dims = tuple(range(1, direction.ndim))
    numerator = (direction * residual).sum(dim=dims, keepdim=True)
    denominator = residual.square().sum(dim=dims, keepdim=True)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return direction - (numerator / denominator) * residual


def dmd_loss_with_stats(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    norm_clip_min=None,
    *,
    normalize=True,
    normalization_epsilon=0.0,
    reduction="mean",
    projected=False,
):
    """Return a detached DMD surrogate, normalizer, and normalized direction RMS.

    The legacy mean surrogate uses half the MSE; sum uses the unhalved
    squared norm per sample, averaged over the batch.
    """
    normalization_epsilon = float(normalization_epsilon)
    if normalization_epsilon < 0:
        raise ValueError("DMD normalization epsilon must be non-negative.")
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported DMD element reduction: {reduction!r}")

    with torch.no_grad():
        grad = x_pred_fake_flow.float() - x_pred_teacher.float()
        if projected:
            residual = x_pred_fake_flow.float() - latents.float()
            grad = project_dmd_direction(grad, residual)
        dims = tuple(range(1, latents.ndim))
        normalizer = torch.abs(latents.float() - x_pred_teacher.float()).mean(dim=dims, keepdim=True)
        if norm_clip_min is not None:
            normalizer = normalizer.clamp(min=float(norm_clip_min))
        if normalize:
            grad = grad / (normalizer + normalization_epsilon)
        grad = torch.nan_to_num(grad)
        normalizer_mean = normalizer.mean()
        direction_rms = grad.square().mean().sqrt()

    latents = latents.float()
    target = (latents - grad).detach()
    if reduction == "mean":
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean")
    else:
        squared_error = F.mse_loss(latents, target, reduction="none")
        loss = squared_error.flatten(1).sum(dim=1).mean()
    return loss, normalizer_mean, direction_rms


def official_pdmd_loss_with_stats(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    *,
    projected=True,
    normalizer_floor=1e-5,
    projection_epsilon=1e-8,
    clamp_max=5.0,
):
    """Released PDMD endpoint surrogate, without changing the legacy objective.

    Inputs must have an explicit sample axis; H3's packed stereo audio already
    has shape [1, stereo_tokens, channels]. Projection and normalization reduce
    all non-sample axes independently for each modality. The final MSE (without
    the legacy 0.5 factor) is clamped before the caller applies modality weights.
    Returns loss, mean normalizer, update RMS, and a detached non-finite count.
    """
    if latents.ndim < 2 or not (latents.shape == x_pred_fake_flow.shape == x_pred_teacher.shape):
        raise ValueError("Official PDMD expects equal endpoint shapes [batch, ...].")
    normalizer_floor = float(normalizer_floor)
    projection_epsilon = float(projection_epsilon)
    if not math.isfinite(normalizer_floor) or normalizer_floor <= 0:
        raise ValueError("Official PDMD normalizer floor must be finite and positive.")
    if not math.isfinite(projection_epsilon) or projection_epsilon < 0:
        raise ValueError("Official PDMD projection epsilon must be finite and non-negative.")

    with torch.no_grad():
        student = latents.float()
        real_residual = student - x_pred_teacher.float()
        fake_residual = student - x_pred_fake_flow.float()
        # Keep the released subtraction order, including FP32 cancellation.
        direction = real_residual - fake_residual
        dims = tuple(range(1, student.ndim))
        if projected:
            rr = fake_residual.square().sum(dim=dims, keepdim=True)
            dr = (direction * fake_residual).sum(dim=dims, keepdim=True)
            direction = direction - dr / (rr + projection_epsilon) * fake_residual
        normalizer = real_residual.abs().mean(dim=dims, keepdim=True).clamp_min(normalizer_floor)
        update = direction / normalizer
        nonfinite_count = (~torch.isfinite(update)).sum()
        update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        normalizer_mean = normalizer.mean()
        direction_rms = update.square().mean().sqrt()

    target = (latents.float() - update).detach()
    loss = official_pdmd_critic_loss(latents, target, clamp_max=clamp_max)
    return loss, normalizer_mean, direction_rms, nonfinite_count


def official_pdmd_critic_loss(prediction, target, *, clamp_max=5.0):
    """Released unhalved FP32 MSE, capped before modality weighting.

    The caller supplies its backend's velocity convention. H3's native
    clean-ward target is the negative of the released noise-ward target, so
    negating both prediction and target yields the same scalar objective.
    Zero or None disables the cap, as in the released implementation.
    """
    if prediction.shape != target.shape:
        raise ValueError("Official PDMD prediction and target shapes differ.")
    if clamp_max is not None:
        clamp_max = float(clamp_max)
        if not math.isfinite(clamp_max) or clamp_max < 0:
            raise ValueError("Official PDMD loss clamp must be finite and non-negative.")
    loss = F.mse_loss(prediction.float(), target.detach().float())
    return loss.clamp(0.0, clamp_max) if clamp_max else loss


def dmd_loss(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    norm_clip_min=None,
    *,
    normalize=True,
    normalization_epsilon=0.0,
    reduction="mean",
    projected=False,
):
    """Compute the detached DMD regression objective."""
    loss, _, _ = dmd_loss_with_stats(
        latents,
        x_pred_fake_flow,
        x_pred_teacher,
        norm_clip_min,
        normalize=normalize,
        normalization_epsilon=normalization_epsilon,
        reduction=reduction,
        projected=projected,
    )
    return loss


def dmd_loss_pair(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    video_weight,
    audio_weight,
    *,
    normalize=True,
    normalization_epsilon=0.0,
    reduction="mean",
    projected=False,
):
    video_loss = dmd_loss(
        latents[0],
        x_pred_fake_flow[0],
        x_pred_teacher[0],
        normalize=normalize,
        normalization_epsilon=normalization_epsilon,
        reduction=reduction,
        projected=projected,
    )
    audio_loss = dmd_loss(
        latents[1],
        x_pred_fake_flow[1],
        x_pred_teacher[1],
        normalize=normalize,
        normalization_epsilon=normalization_epsilon,
        reduction=reduction,
        projected=projected,
    )
    return video_weight * video_loss + audio_weight * audio_loss


def weighted_mse_pair(
    pred,
    target,
    video_weight,
    audio_weight,
):
    video_loss = F.mse_loss(
        pred[0].float(),
        target[0].float(),
        reduction="mean",
    )
    audio_loss = F.mse_loss(
        pred[1].float(),
        target[1].float(),
        reduction="mean",
    )
    return video_weight * video_loss + audio_weight * audio_loss


def detach_pair(value):
    return value[0].detach(), value[1].detach()


def to_dtype_pair(value, dtype):
    return value[0].to(dtype=dtype), value[1].to(dtype=dtype)
