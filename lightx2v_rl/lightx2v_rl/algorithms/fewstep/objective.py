"""FewStepRL's clipped Wan policy objective with Geo-FSRL reweighting.

Adapted from FewStepRL ``wan_trainer.py`` at 3dd8e422 (Apache-2.0).
No reward scoring, reward aggregation or advantage estimation is included.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .bridge import _batch_sigma, _mean_nonbatch, _validate_sigmas, geo_reweight_scale
from .config import FewStepConfig


@dataclass(frozen=True)
class PolicyLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def compute_reweight_scales(
    sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    config: FewStepConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one policy and KL multiplier per transition.

    The Wan source applies a geometry factor only to Geo. CPS and Flow-GRPO
    receive the optional difficulty factor, with its square used for KL.
    """
    sigma = sigma.float().reshape(-1)
    next_sigma = next_sigma.to(device=sigma.device, dtype=torch.float32).reshape(-1)
    sigma, next_sigma = torch.broadcast_tensors(sigma, next_sigma)
    _validate_sigmas(sigma, next_sigma)
    scale = torch.ones_like(sigma)
    if config.loss_reweighting != "matched":
        return scale, scale
    if config.difficulty_reweighting == "linear_t2_minus_s2":
        scale = (sigma.square() - next_sigma.square()).clamp_min(0)
    if config.bridge == "geo":
        scale = scale * geo_reweight_scale(sigma, next_sigma) * config.geo_reweight_multiplier
    return scale, scale.square()


def _vector(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    value = value.to(device=reference.device, dtype=torch.float32)
    if value.numel() != reference.numel():
        raise ValueError(f"{name} must contain one value per transition ({reference.numel()}).")
    return value.reshape_as(reference)


def compute_policy_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
    config: FewStepConfig,
    transition_mean: torch.Tensor | None = None,
    reference_mean: torch.Tensor | None = None,
    transition_std: torch.Tensor | None = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> PolicyLossOutput:
    """Compute clipped policy loss plus the reference-transition penalty.

    Inputs contain one row per selected transition. Advantages are provided by
    the caller; this module never computes rewards or group statistics.
    Optional rollout importance weights affect only the policy term.

    CPS deliberately preserves FewStepRL's legacy KL surrogate: its variance
    denominator is fixed to 0.5, so the final penalty is mean squared mean
    difference / (2*0.5). It is *not* the Gaussian KL using CPS's actual std.
    Geo and Flow-GRPO use mean squared mean difference / (2*std**2).
    """
    if new_log_probs.ndim != 1 or new_log_probs.numel() == 0:
        raise ValueError("new_log_probs must be a nonempty vector with one value per transition.")
    new_log_probs = new_log_probs.float()
    old_log_probs = _vector(old_log_probs, new_log_probs, "old_log_probs").detach()
    advantages = (
        _vector(advantages, new_log_probs, "advantages")
        .detach()
        .clamp(
            -config.adv_clip_max,
            config.adv_clip_max,
        )
    )
    sigma = _batch_sigma(sigma, new_log_probs, "sigma")
    next_sigma = _batch_sigma(next_sigma, new_log_probs, "next_sigma")
    policy_scale, kl_scale = compute_reweight_scales(sigma, next_sigma, config)
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    policy_loss = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - config.clip_range, 1 + config.clip_range),
    )
    policy_terms = policy_loss
    if rollout_is_weights is not None:
        weights = _vector(rollout_is_weights, new_log_probs, "rollout_is_weights").detach()
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise ValueError("rollout_is_weights must be finite and nonnegative.")
        policy_terms = policy_terms * weights
    weighted_policy = policy_scale * policy_terms
    total = weighted_policy
    kl_loss = None
    if config.kl_beta > 0:
        if transition_mean is None or reference_mean is None:
            raise ValueError("Positive kl_beta requires transition_mean and reference_mean.")
        if transition_mean.shape != reference_mean.shape or transition_mean.shape[0] != new_log_probs.numel():
            raise ValueError("Transition and reference means must share a shape with one row per transition.")
        mean_delta = transition_mean.float() - reference_mean.detach().to(transition_mean.device, torch.float32)
        mean_square = _mean_nonbatch(mean_delta.square())
        if config.bridge == "cps":
            denominator = torch.full_like(new_log_probs, 0.5)
        else:
            if transition_std is None:
                raise ValueError("Positive kl_beta requires transition_std for Geo and Flow-GRPO.")
            std = transition_std.to(device=new_log_probs.device, dtype=torch.float32)
            if std.ndim == 0 or std.numel() == 1:
                std = std.reshape(1).expand_as(new_log_probs)
            if std.shape[0] != new_log_probs.numel():
                raise ValueError("transition_std must be scalar or have one row per transition.")
            if not torch.isfinite(std).all() or torch.any(std <= 0):
                raise ValueError("transition_std must be finite and positive.")
            denominator = _mean_nonbatch(std.square())
        kl_loss = mean_square / (2 * denominator)
        total = total + config.kl_beta * kl_loss * kl_scale
    loss = total.mean()
    metrics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.mean().detach(),
        "weighted_policy_loss": weighted_policy.mean().detach(),
        "approx_kl": (0.5 * log_ratio.square().mean()).detach(),
        "clip_frac": ((ratio - 1).abs() > config.clip_range).float().mean().detach(),
        "clip_frac_gt_one": (ratio - 1 > config.clip_range).float().mean().detach(),
        "clip_frac_lt_one": (1 - ratio > config.clip_range).float().mean().detach(),
        "policy_reweight_scale": policy_scale.mean().detach(),
        "kl_reweight_scale": kl_scale.mean().detach(),
    }
    if kl_loss is not None:
        metrics["kl_loss"] = kl_loss.mean().detach()
        metrics["weighted_kl_loss"] = (kl_loss * kl_scale).mean().detach()
    return PolicyLossOutput(loss=loss, metrics=metrics)
