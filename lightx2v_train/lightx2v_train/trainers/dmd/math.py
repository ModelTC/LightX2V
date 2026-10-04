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
