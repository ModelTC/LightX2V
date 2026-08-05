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


def dmd_loss(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    norm_clip_min=None,
):
    """Compute the detached DMD regression objective."""
    with torch.no_grad():
        grad = x_pred_fake_flow - x_pred_teacher
        dims = tuple(range(1, latents.ndim))
        normalizer = torch.abs(latents - x_pred_teacher).mean(dim=dims, keepdim=True)
        if norm_clip_min is not None:
            normalizer = normalizer.clamp(min=float(norm_clip_min))
        grad = torch.nan_to_num(grad / normalizer)
    return 0.5 * F.mse_loss(
        latents.float(),
        (latents.float() - grad.float()).detach(),
        reduction="mean",
    )


def dmd_loss_pair(
    latents,
    x_pred_fake_flow,
    x_pred_teacher,
    video_weight,
    audio_weight,
):
    video_loss = dmd_loss(
        latents[0],
        x_pred_fake_flow[0],
        x_pred_teacher[0],
    )
    audio_loss = dmd_loss(
        latents[1],
        x_pred_fake_flow[1],
        x_pred_teacher[1],
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
