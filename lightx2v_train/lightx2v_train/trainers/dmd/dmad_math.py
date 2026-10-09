"""DMAD objectives, noise routing and power EMA.

Adapted from Yzmblog/DMAD (Apache-2.0), commit
8067c05f74a8cfc818d6e21c2b49405b49ba9cbc, train/h3.
Reference: https://arxiv.org/abs/2610.02188.
Unlike the upstream rank-local router, statistics may be synchronized globally.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from .student_ema import StudentWeightEMA


@dataclass(frozen=True)
class DmadConfig:
    feature_block: int = 49
    gap_tau: float = 2.0
    gap_bands: int = 10
    gap_beta: float = 0.99
    gap_min_count: int = 10
    gap_ready_bands: int = 5
    gap_sync: bool = True
    lambda_real: float = 1.0
    lambda_teacher: float = 1.0
    renoise_sigma_min: float = 0.02
    renoise_sigma_max: float = 0.98
    ema_gammas: tuple = (6.94, 16.97)

    @classmethod
    def from_mapping(cls, value):
        value = dict(value or {})
        unknown = set(value) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown training.dmad options: {sorted(unknown)}")
        if "ema_gammas" in value:
            value["ema_gammas"] = tuple(float(x) for x in value["ema_gammas"])
        result = cls(**value)
        if not 0 <= result.renoise_sigma_min < result.renoise_sigma_max < 1:
            raise ValueError("DMAD requires 0 <= renoise_sigma_min < renoise_sigma_max < 1.")
        if result.feature_block < 0 or result.gap_tau <= 0 or not math.isfinite(result.gap_tau):
            raise ValueError("DMAD feature_block must be nonnegative and gap_tau finite and positive.")
        if not 0 <= result.gap_beta < 1 or result.gap_min_count < 1:
            raise ValueError("Invalid DMAD gap EMA settings.")
        if not 1 <= result.gap_ready_bands <= result.gap_bands:
            raise ValueError("Invalid DMAD gap band count.")
        if any(not math.isfinite(x) or x < 0 for x in (result.lambda_real, result.lambda_teacher)):
            raise ValueError("DMAD loss weights must be finite and nonnegative.")
        if result.lambda_real + result.lambda_teacher <= 0:
            raise ValueError("DMAD needs at least one active target.")
        if any(not math.isfinite(x) or x < 0 for x in result.ema_gammas):
            raise ValueError("DMAD EMA gamma must be finite and nonnegative.")
        return result

    def metadata(self):
        return asdict(self)


class DualHeadCritic(nn.Module):
    """FP32 real/teacher logits; equal audio/video weighting, target tokens only."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.real = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.teacher = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def _logits(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] == 0:
            raise ValueError("DMAD heads require nonempty [B, target_tokens, hidden] features.")
        tokens = tokens.float()
        tokens = tokens * torch.rsqrt(tokens.square().mean(-1, keepdim=True) + self.eps)
        return self.real(tokens).mean(1).squeeze(-1), self.teacher(tokens).mean(1).squeeze(-1)

    def forward(self, video_features, audio_features):
        vr, vt = self._logits(video_features)
        ar, at = self._logits(audio_features)
        return (vr + ar) / 2, (vt + at) / 2


def generator_loss(real_logits, teacher_logits, weight=1.0, lambda_real=1.0, lambda_teacher=1.0):
    # Linear logits, NOT non-saturating GAN softplus(-logit).
    weight = torch.as_tensor(weight, device=teacher_logits.device, dtype=teacher_logits.dtype).detach()
    return -(lambda_real * real_logits + lambda_teacher * weight * teacher_logits).mean()


def critic_loss(g_real, g_teacher, real_positive, teacher_positive):
    return (F.softplus(g_real) + F.softplus(g_teacher) + F.softplus(-real_positive) + F.softplus(-teacher_positive)).mean()


class GapRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("ema", torch.zeros(config.gap_bands, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(config.gap_bands, dtype=torch.long))

    def band(self, base_sigma):
        return min(self.config.gap_bands - 1, max(0, int(float(base_sigma) * self.config.gap_bands)))

    @torch.no_grad()
    def update(self, band, real_logits, teacher_real_logits):
        # Counts are observation rounds, not rank/sample counts. A global round
        # updates each observed band once, so beta remains world-size independent.
        stats = self.ema.new_zeros((2, self.config.gap_bands))
        gap = real_logits.detach().float().mean() - teacher_real_logits.detach().float().mean()
        stats[0, band], stats[1, band] = gap, 1
        if self.config.gap_sync and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats)
        seen = stats[1] > 0
        self.ema[seen] = self.config.gap_beta * self.ema[seen] + (1 - self.config.gap_beta) * stats[0, seen] / stats[1, seen]
        self.counts[seen] += 1

    @torch.no_grad()
    def weight(self, band):
        ready = self.counts >= self.config.gap_min_count
        if int(ready.sum()) < self.config.gap_ready_bands or not bool(ready[band]):
            return self.ema.new_tensor(1.0)
        corrected = self.ema / (1 - self.config.gap_beta ** self.counts.clamp_min(1).float())
        raw = torch.sigmoid((corrected[ready].median() - corrected) / self.config.gap_tau)
        return raw[band] / raw[ready].mean().clamp_min(1e-4)


def rollout_base_sigmas(steps, low, high, *, device, random_timesteps=True):
    """Training: descending running-min draws; inference: fixed linear grid."""
    if steps < 1:
        raise ValueError("DMAD rollout needs at least one evaluation.")
    if not random_timesteps:
        return torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float32)
    values = torch.ones(steps + 1, device=device, dtype=torch.float32)
    if steps > 1:
        values[1:-1] = torch.empty(steps - 1, device=device).uniform_(low, high).cummin(0).values
    values[-1] = 0
    return values


class PowerEMA(StudentWeightEMA):
    """Power-function EMA on trainable (possibly DTensor-sharded) weights."""

    def __init__(self, module, gamma):
        super().__init__(module, decay=0.0)
        self.gamma = float(gamma)

    def update(self):
        self.decay = (1 - 1 / (self.num_updates + 1)) ** (self.gamma + 1)
        super().update()

    def state_dict(self):
        return {**super().state_dict(), "gamma": self.gamma}

    def load_state_dict(self, state):
        if state["gamma"] != self.gamma:
            raise ValueError("DMAD power EMA gamma mismatch.")
        self.decay = float(state["decay"])
        super().load_state_dict(state)
