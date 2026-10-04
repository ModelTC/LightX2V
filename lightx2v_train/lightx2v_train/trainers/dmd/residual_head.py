"""Small, frozen-feature fake residual correction and held-out noise-bin gates.

The head predicts ``F - X`` in x0 units. It never consumes the clean sample
and cannot send a gradient into the fake features. The gate must be updated
only from independent held-out student batches, not head-fitting or student-
update samples; that separation is the responsibility of the trainer.
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class ResidualHeadConfig:
    enabled: bool = False
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    fit_steps: int = 5
    fit_grad_accum_steps: int = 1
    noise_bins: int = 5
    ema_decay: float = 0.9
    min_checks: int = 3
    gate_ramp: float = 0.25
    gate_mode: str = "full"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    min_relative_improvement: float = 0.0

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("residual_head.enabled must be a boolean.")
        for name in ("hidden_dim", "fit_steps", "fit_grad_accum_steps", "noise_bins", "min_checks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"residual_head.{name} must be a positive integer.")
        if self.gate_mode not in ("full", "calibrated"):
            raise ValueError("residual_head.gate_mode must be 'full' or 'calibrated'.")
        for name in ("learning_rate", "ema_decay", "gate_ramp", "max_grad_norm", "weight_decay", "min_relative_improvement"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"residual_head.{name} must be finite and numeric.")
        if self.learning_rate <= 0:
            raise ValueError("residual_head.learning_rate must be positive.")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("residual_head.ema_decay must lie in [0, 1).")
        if not 0 < self.gate_ramp <= 1:
            raise ValueError("residual_head.gate_ramp must lie in (0, 1].")
        if self.max_grad_norm <= 0:
            raise ValueError("residual_head.max_grad_norm must be positive.")
        if self.weight_decay < 0:
            raise ValueError("residual_head.weight_decay must be non-negative.")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("residual_head.min_relative_improvement must lie in [0, 1).")

    @classmethod
    def from_mapping(cls, mapping=None):
        if mapping is None:
            return cls()
        if not isinstance(mapping, Mapping):
            raise ValueError("residual_head configuration must be a mapping.")
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown residual_head configuration keys: {sorted(unknown)}")
        return cls(**mapping)

    def checkpoint_metadata(self):
        return asdict(self)


class TokenResidualHead(nn.Module):
    """A low-rank, sigma-conditioned x0 residual head on Wan fake tokens.

    Inputs are detached internally. The output projection is zero initialized
    so adding a new head leaves the original fake predictions unchanged.
    Computation remains FP32 even under the trainer's BF16 autocast context.
    """

    def __init__(self, feature_dim, latent_channels, patch_size, hidden_dim=64):
        super().__init__()
        for name, value in (("feature_dim", feature_dim), ("latent_channels", latent_channels), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.patch_size = tuple(patch_size)
        if len(self.patch_size) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.patch_size):
            raise ValueError("patch_size must contain three positive integers.")
        self.feature_dim = feature_dim
        self.latent_channels = latent_channels
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.sigma_projection = nn.Linear(1, hidden_dim, bias=False)
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(hidden_dim, latent_channels * math.prod(self.patch_size))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.float()

    def _latent_grid(self, latent_shape, batch_size):
        latent_shape = tuple(latent_shape)
        if len(latent_shape) != 5 or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in latent_shape):
            raise ValueError("latent_shape must be positive [B, C, T, H, W] dimensions.")
        if latent_shape[0] != batch_size or latent_shape[1] != self.latent_channels:
            raise ValueError("latent_shape batch/channel dimensions do not match the residual head.")
        if any(size % patch for size, patch in zip(latent_shape[2:], self.patch_size)):
            raise ValueError("latent_shape temporal/spatial dimensions must be divisible by patch_size.")
        return latent_shape, tuple(size // patch for size, patch in zip(latent_shape[2:], self.patch_size))

    def unpatchify(self, tokens, latent_shape):
        """Use Wan's exact ``fhwpqrc -> cfphqwr`` patch-to-latent layout."""
        if tokens.ndim != 3:
            raise ValueError("Residual tokens must have shape [B, L, C * patch_volume].")
        latent_shape, grid = self._latent_grid(latent_shape, tokens.shape[0])
        num_tokens = math.prod(grid)
        if tokens.shape[1] < num_tokens:
            raise ValueError("Residual token sequence is shorter than the latent patch grid.")
        if tokens.shape[2] != self.latent_channels * math.prod(self.patch_size):
            raise ValueError("Residual token width does not match channels * patch_volume.")
        # Wan may pad its token sequence; only the actual patch grid is used.
        output = tokens[:, :num_tokens].reshape(tokens.shape[0], *grid, *self.patch_size, self.latent_channels)
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return output.reshape(latent_shape)

    def forward(self, features, sigma, latent_shape):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Fake features must have shape [B, L, {self.feature_dim}].")
        self._latent_grid(latent_shape, features.shape[0])
        sigma = torch.as_tensor(sigma, device=features.device, dtype=torch.float32).detach()
        if sigma.numel() == 1:
            sigma = sigma.reshape(1).expand(features.shape[0])
        elif sigma.numel() == features.shape[0]:
            sigma = sigma.reshape(features.shape[0])
        else:
            raise ValueError("sigma must be scalar or contain one value per feature batch item.")
        if not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
            raise ValueError("sigma must contain finite values in [0, 1].")
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = self.feature_norm(features.detach().float())
            hidden = self.feature_projection(features) + self.sigma_projection(sigma[:, None, None])
            tokens = self.output_projection(self.activation(hidden))
            return self.unpatchify(tokens, latent_shape)


class NoiseBinGate(nn.Module):
    """Conservative EMA gate using independent held-out rounds per noise bin.

    ``counts`` counts outer update rounds with evidence in each bin, not pooled
    samples or DP ranks. Distributed sums are used for the round's MSE means,
    so all ranks make the same decision and min_checks is world-size invariant.
    Only qualifying bins ramp upward; insufficient or unfavorable EMA evidence
    resets them to zero, preserving ordinary DMD as the fallback.

    In calibrated mode, separate calibration data chooses a continuous scale;
    independent validation data only accepts/rejects that exact scale. EMA
    sufficient statistics allow evaluating a new calibration-selected scale
    without mixing MSE measurements of different scales. Validation statistics
    must never be used to choose a scale. Independence of the sampled batches
    (including from head fitting and the student update) is the caller's duty.
    """

    def __init__(self, config, device=None):
        super().__init__()
        if not isinstance(config, ResidualHeadConfig):
            raise ValueError("NoiseBinGate requires a ResidualHeadConfig.")
        self.config = config
        self.register_buffer("ema_delta", torch.zeros(config.noise_bins, device=device, dtype=torch.float32))
        self.register_buffer("ema_fake_mse", torch.zeros(config.noise_bins, device=device, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(config.noise_bins, device=device, dtype=torch.int64))
        self.register_buffer("lambdas", torch.zeros(config.noise_bins, device=device, dtype=torch.float32))
        # Keep full-mode state dictionaries compatible with pre-calibration
        # checkpoints. Calibrated checkpoints carry both independent streams.
        if config.gate_mode == "calibrated":
            for name in ("calibration_dot", "calibration_energy", "candidate_lambdas", "validation_dot", "validation_energy"):
                self.register_buffer(name, torch.zeros(config.noise_bins, device=device, dtype=torch.float32))
            self.register_buffer("calibration_counts", torch.zeros(config.noise_bins, device=device, dtype=torch.int64))

    def _sigma_tensor(self, sigma):
        sigma = torch.as_tensor(sigma, device=self.lambdas.device, dtype=torch.float32).detach()
        if not torch.isfinite(sigma).all() or ((sigma < 0) | (sigma > 1)).any():
            raise ValueError("Gate sigma must contain finite values in [0, 1].")
        return sigma

    def _bin_indices(self, sigma):
        return (self._sigma_tensor(sigma) * self.config.noise_bins).long().clamp(max=self.config.noise_bins - 1)

    @torch.no_grad()
    def lambda_for(self, sigma):
        return self.lambdas[self._bin_indices(sigma)].clone()

    @torch.no_grad()
    def update(self, checks):
        """Pool ``(sigma, fake_mse, corrected_mse)`` held-out observations.

        Each MSE can be scalar or per-example. Scalars broadcast against a
        sigma batch; a scalar sigma can similarly describe a whole MSE batch.
        The corrected MSE is for the full ``F - C`` prediction, never an
        adaptively chosen per-example correction direction.
        """
        if self.config.gate_mode != "full":
            raise ValueError("Calibrated gates require separate calibrate() and update_calibrated() calls.")
        stats = torch.zeros(3, self.config.noise_bins, device=self.lambdas.device, dtype=torch.float32)
        for sigma, fake_mse, corrected_mse in checks:
            sigma = self._sigma_tensor(sigma).reshape(-1)
            fake_mse = torch.as_tensor(fake_mse, device=stats.device, dtype=stats.dtype).detach().reshape(-1)
            corrected_mse = torch.as_tensor(corrected_mse, device=stats.device, dtype=stats.dtype).detach().reshape(-1)
            try:
                sigma, fake_mse, corrected_mse = torch.broadcast_tensors(sigma, fake_mse, corrected_mse)
            except RuntimeError as error:
                raise ValueError("Held-out sigma and MSE shapes must be scalar or matching per-example vectors.") from error
            if not torch.isfinite(fake_mse).all() or not torch.isfinite(corrected_mse).all():
                raise ValueError("Held-out MSE values must be finite.")
            if (fake_mse < 0).any() or (corrected_mse < 0).any():
                raise ValueError("Held-out MSE values must be non-negative.")
            bins = self._bin_indices(sigma)
            stats[0].scatter_add_(0, bins, fake_mse)
            stats[1].scatter_add_(0, bins, corrected_mse)
            stats[2].scatter_add_(0, bins, torch.ones_like(fake_mse))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        observed = stats[2] > 0
        mean_fake = stats[0] / stats[2].clamp_min(1)
        mean_delta = (stats[0] - stats[1]) / stats[2].clamp_min(1)
        first = observed & (self.counts == 0)
        subsequent = observed & ~first
        self.ema_fake_mse[first] = mean_fake[first]
        self.ema_delta[first] = mean_delta[first]
        decay = self.config.ema_decay
        self.ema_fake_mse[subsequent] = decay * self.ema_fake_mse[subsequent] + (1 - decay) * mean_fake[subsequent]
        self.ema_delta[subsequent] = decay * self.ema_delta[subsequent] + (1 - decay) * mean_delta[subsequent]
        self.counts.add_(observed.to(self.counts.dtype))
        reliable = (self.counts >= self.config.min_checks) & (self.ema_delta > self.config.min_relative_improvement * self.ema_fake_mse)
        if not self.config.enabled:
            reliable.zero_()
        self.lambdas[~reliable] = 0
        ramp = observed & reliable
        self.lambdas[ramp] = (self.lambdas[ramp] + self.config.gate_ramp).clamp(max=1)
        return self.metrics()

    def _pool_sufficient_statistics(self, checks, value_names, nonnegative):
        """Pool per-example mean inner products/energies, not tensor norms."""
        stats = torch.zeros(len(value_names) + 1, self.config.noise_bins, device=self.lambdas.device, dtype=torch.float32)
        for check in checks:
            if len(check) != len(value_names) + 1:
                raise ValueError(f"Gate checks require sigma and {', '.join(value_names)}.")
            sigma = self._sigma_tensor(check[0]).reshape(-1)
            values = [torch.as_tensor(value, device=stats.device, dtype=stats.dtype).detach().reshape(-1) for value in check[1:]]
            try:
                sigma, *values = torch.broadcast_tensors(sigma, *values)
            except RuntimeError as error:
                raise ValueError("Gate statistics and sigma must be scalar or matching per-example vectors.") from error
            for index, (name, value) in enumerate(zip(value_names, values)):
                if not torch.isfinite(value).all():
                    raise ValueError(f"Gate {name} must be finite.")
                if index in nonnegative and (value < 0).any():
                    raise ValueError(f"Gate {name} must be non-negative.")
            bins = self._bin_indices(sigma)
            for index, value in enumerate(values):
                stats[index].scatter_add_(0, bins, value)
            stats[-1].scatter_add_(0, bins, torch.ones_like(sigma))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    def _update_statistic_emas(self, stats, counts, buffers):
        observed = stats[-1] > 0
        first = observed & (counts == 0)
        subsequent = observed & ~first
        means = stats[:-1] / stats[-1].clamp_min(1)
        for mean, buffer in zip(means, buffers):
            buffer[first] = mean[first]
            buffer[subsequent] = self.config.ema_decay * buffer[subsequent] + (1 - self.config.ema_decay) * mean[subsequent]
        counts.add_(observed.to(counts.dtype))
        return observed

    @torch.no_grad()
    def calibrate(self, checks):
        """Choose per-bin scales ONLY from independent calibration queries.

        ``checks`` contains ``(sigma, mean((F-X)*C), mean(C**2))`` per-example
        observations. The returned snapshot is detached from mutable state;
        pass it unchanged to update_calibrated() using a fresh validation batch.
        Calibrating alone never opens a gate or changes validation evidence.
        """
        if self.config.gate_mode != "calibrated":
            raise ValueError("calibrate() requires gate_mode='calibrated'.")
        stats = self._pool_sufficient_statistics(checks, ("dot", "energy"), {1})
        observed = self._update_statistic_emas(stats, self.calibration_counts, (self.calibration_dot, self.calibration_energy))
        scales = torch.where(
            self.calibration_energy > 0,
            self.calibration_dot / self.calibration_energy.clamp_min(torch.finfo(self.calibration_energy.dtype).tiny),
            torch.zeros_like(self.calibration_energy),
        ).clamp(0, 1)
        self.candidate_lambdas[observed] = scales[observed]
        return self.candidate_lambdas.clone()

    @torch.no_grad()
    def update_calibrated(self, checks, candidate_lambdas):
        """Independently validate, never re-fit, calibration-selected scales.

        Validation checks are ``(sigma, mean((F-X)**2), mean((F-X)*C),
        mean(C**2))`` from NEW queries. The exact candidate's risk reduction is
        ``2 * lambda * E[dot] - lambda**2 * E[energy]``. Keeping these validation
        EMAs instead of scaled-MSE EMAs avoids recycling evidence for a stale
        lambda. No argmin, dot/energy ratio, or alternative scale selection is
        performed on validation data.

        A bin requires min_checks calibration AND validation rounds and fresh
        validation observations before changing its applied scale. Unobserved
        bins retain their previously accepted scale, not the new candidate.
        There is no ramp in this mode: only the exact tested scale is applied.
        """
        if self.config.gate_mode != "calibrated":
            raise ValueError("update_calibrated() requires gate_mode='calibrated'.")
        candidates = torch.as_tensor(candidate_lambdas, device=self.lambdas.device, dtype=torch.float32).detach()
        if candidates.shape != self.lambdas.shape or not torch.isfinite(candidates).all() or ((candidates < 0) | (candidates > 1)).any():
            raise ValueError("Candidate lambdas must be one finite [0, 1] scale per noise bin.")
        if not torch.equal(candidates, self.candidate_lambdas):
            raise ValueError("Candidate lambdas must match the unchanged latest calibration snapshot.")
        stats = self._pool_sufficient_statistics(checks, ("base_mse", "dot", "energy"), {0, 2})
        observed = self._update_statistic_emas(stats, self.counts, (self.ema_fake_mse, self.validation_dot, self.validation_energy))
        delta = 2 * candidates * self.validation_dot - candidates.square() * self.validation_energy
        self.ema_delta[observed] = delta[observed]
        reliable = (self.calibration_counts >= self.config.min_checks) & (self.counts >= self.config.min_checks) & (candidates > 0) & (delta > self.config.min_relative_improvement * self.ema_fake_mse)
        self.lambdas[observed] = torch.where(reliable[observed], candidates[observed], torch.zeros_like(candidates[observed]))
        if not self.config.enabled:
            self.lambdas.zero_()
        return self.metrics()

    @torch.no_grad()
    def metrics(self):
        metrics = {"lambda_mean": self.lambdas.mean().item(), "active_bins": (self.lambdas > 0).sum().item()}
        relative = self.ema_delta / self.ema_fake_mse.clamp_min(torch.finfo(self.ema_fake_mse.dtype).tiny)
        for index in range(self.config.noise_bins):
            metrics.update(
                {
                    f"lambda_bin_{index}": self.lambdas[index].item(),
                    f"checks_bin_{index}": self.counts[index].item(),
                    f"delta_bin_{index}": self.ema_delta[index].item(),
                    f"fake_mse_bin_{index}": self.ema_fake_mse[index].item(),
                    f"relative_improvement_bin_{index}": relative[index].item(),
                }
            )
            if self.config.gate_mode == "calibrated":
                metrics.update(
                    {
                        f"calibration_checks_bin_{index}": self.calibration_counts[index].item(),
                        f"calibration_dot_bin_{index}": self.calibration_dot[index].item(),
                        f"calibration_energy_bin_{index}": self.calibration_energy[index].item(),
                        f"candidate_lambda_bin_{index}": self.candidate_lambdas[index].item(),
                        f"validation_dot_bin_{index}": self.validation_dot[index].item(),
                        f"validation_energy_bin_{index}": self.validation_energy[index].item(),
                    }
                )
        return metrics
