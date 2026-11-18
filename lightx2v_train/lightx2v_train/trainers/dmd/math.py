import torch
import torch.nn.functional as F


def pdmd_critic_loss(prediction, target, clamp_max=5.0):
    return F.mse_loss(prediction.float(), target.detach().float()).clamp(max=clamp_max)


def pdmd_loss_with_stats(latents, fake_x0, teacher_x0, *, normalizer_floor=1e-5, projection_epsilon=1e-8, clamp_max=5.0):
    """Compute the projected endpoint surrogate independently for each batch item."""
    if latents.ndim < 2 or not (latents.shape == fake_x0.shape == teacher_x0.shape):
        raise ValueError("PDMD requires equal endpoint shapes with a batch dimension.")
    dims = tuple(range(1, latents.ndim))
    with torch.no_grad():
        student = latents.float()
        teacher_residual = student - teacher_x0.float()
        fake_residual = student - fake_x0.float()
        delta = teacher_residual - fake_residual
        dot = (delta * fake_residual).sum(dim=dims, keepdim=True)
        norm_sq = fake_residual.square().sum(dim=dims, keepdim=True)
        delta = delta - dot / (norm_sq + projection_epsilon) * fake_residual
        normalizer = teacher_residual.abs().mean(dim=dims, keepdim=True).clamp_min(normalizer_floor)
        update = delta / normalizer
        nonfinite = (~torch.isfinite(update)).sum()
        update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
        target = student - update
    return pdmd_critic_loss(latents, target, clamp_max), normalizer.mean(), update.square().mean().sqrt(), nonfinite


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
    sigma = expand_sigma(sigma.float(), sample.ndim)
    sigma_next = expand_sigma(sigma_next.float(), sample.ndim)
    return (sample.float() + (sigma_next - sigma) * velocity.float()).to(sample.dtype)


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
    cond_pred, uncond_pred = cond_pred.float(), uncond_pred.float()
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
    latents = latents.float()
    x_pred_fake_flow = x_pred_fake_flow.float()
    x_pred_teacher = x_pred_teacher.float()
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
