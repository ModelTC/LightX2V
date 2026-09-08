# Adapted from TBSM (Apache-2.0), imgnet/methods/tbsm.py:
# https://github.com/sp12138/TBSM
# Changes: identity action features, padding-aware reductions, lambda=1/rho=0.
import torch


def action_scattering_loss(projectile, positive, negative, valid_mask=None):
    """Return self-normalized TBSM loss and detached raw loss for [B, T, D].

    Keep dataset-normalized action magnitudes; do not apply image-feature RMS
    normalization. Each nonempty chunk has equal weight, regardless of padding.
    Only the projectile receives gradients.
    """
    if projectile.ndim != 3 or positive.shape != projectile.shape or negative.shape != projectile.shape:
        raise ValueError("TBSM actions must have matching [B, T, D] shapes.")
    if valid_mask is None:
        valid_mask = torch.ones(projectile.shape[:2], device=projectile.device, dtype=torch.bool)
    elif valid_mask.shape != projectile.shape[:2]:
        raise ValueError("TBSM valid_mask must have shape [B, T].")
    mask = valid_mask.to(device=projectile.device, dtype=torch.bool).unsqueeze(-1).expand_as(projectile)
    x = projectile.float().masked_fill(~mask, 0.0)
    valid_count = mask.sum(dim=(1, 2))

    with torch.no_grad():

        def bearing(source):
            offset = (source.float() - x.detach()).masked_fill(~mask, 0.0).flatten(1)
            radius = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
            return (offset / (radius + 1e-6) * valid_count.float().sqrt().unsqueeze(-1)).view_as(x)

        target = x.detach() + bearing(positive) - bearing(negative)

    per_sample = (x - target).square().sum(dim=(1, 2)) / valid_count.clamp_min(1)
    nonempty = valid_count > 0
    raw_loss = (per_sample * nonempty).sum() / nonempty.sum().clamp_min(1)
    return raw_loss / raw_loss.detach().clamp_min(1e-8), raw_loss.detach()
