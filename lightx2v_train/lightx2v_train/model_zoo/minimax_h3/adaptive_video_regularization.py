"""Losses from Adaptive Video Distillation (arXiv:2603.21864).

The paper adds two generator-side objectives to DMD:

* adaptive flow regression on clean video latents; and
* a truncated temporal-variance regularizer on generated video latents.

This module owns only configuration, scalar EMA state, and loss arithmetic.
Model-specific packing and forward passes remain in the H3 trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class AdaptiveRegressionConfig:
    enabled: bool = True
    weight: float = 2.0
    ema_decay: float = 0.95
    sigmoid_scale: float = 3.0


@dataclass(frozen=True)
class AudioRegressionConfig:
    enabled: bool = False
    weight: float = 0.0


@dataclass(frozen=True)
class TemporalRegularizationConfig:
    enabled: bool = True
    weight: float = 0.05
    epsilon: float = 1.0e-6
    loss_threshold: float = 0.6
    compute_dtype: str = "fp32"


class AdaptiveVideoRegularizer:
    """Configurable ADV losses with one regression-loss EMA per rollout step."""

    state_version = 1

    def __init__(self, config: Mapping | None, num_inference_steps: int):
        config = config or {}
        self.enabled = bool(config.get("enabled", False))
        regression = config.get("regression", {}) or {}
        audio_regression = config.get("audio_regression", {}) or {}
        temporal = config.get("temporal", {}) or {}
        self.regression = AdaptiveRegressionConfig(
            enabled=self.enabled and bool(regression.get("enabled", True)),
            weight=float(regression.get("weight", 2.0)),
            ema_decay=float(regression.get("ema_decay", 0.95)),
            sigmoid_scale=float(regression.get("sigmoid_scale", 3.0)),
        )
        self.audio_regression = AudioRegressionConfig(
            enabled=self.enabled and bool(audio_regression.get("enabled", False)),
            weight=float(audio_regression.get("weight", 0.0)),
        )
        self.temporal = TemporalRegularizationConfig(
            enabled=self.enabled and bool(temporal.get("enabled", True)),
            weight=float(temporal.get("weight", 0.05)),
            epsilon=float(temporal.get("epsilon", 1.0e-6)),
            loss_threshold=float(temporal.get("loss_threshold", 0.6)),
            compute_dtype=str(temporal.get("compute_dtype", "fp32")).lower(),
        )
        self.num_inference_steps = int(num_inference_steps)
        if self.num_inference_steps < 1:
            raise ValueError("ADV requires at least one student inference step.")
        self._validate()
        self.regression_ema = [None] * self.num_inference_steps
        self.regression_updates = [0] * self.num_inference_steps
        self._pending_regression_sum = [0.0] * self.num_inference_steps
        self._pending_regression_count = [0] * self.num_inference_steps

    def _validate(self):
        if self.enabled and not (self.regression.enabled or self.audio_regression.enabled or self.temporal.enabled):
            raise ValueError("training.dmd.adaptive_video_regularization.enabled=true requires regression.enabled, audio_regression.enabled, and/or temporal.enabled.")
        if self.regression.weight < 0:
            raise ValueError("ADV regression.weight must be non-negative.")
        if self.audio_regression.weight < 0:
            raise ValueError("ADV audio_regression.weight must be non-negative.")
        if not 0.0 <= self.regression.ema_decay < 1.0:
            raise ValueError("ADV regression.ema_decay must satisfy 0 <= value < 1.")
        if self.regression.sigmoid_scale <= 0:
            raise ValueError("ADV regression.sigmoid_scale must be positive.")
        if self.temporal.weight < 0:
            raise ValueError("ADV temporal.weight must be non-negative.")
        if self.temporal.epsilon <= 0:
            raise ValueError("ADV temporal.epsilon must be positive.")
        if self.temporal.compute_dtype not in {"fp32", "float32", "fp64", "float64"}:
            raise ValueError(f"ADV temporal.compute_dtype must be fp32 or fp64, got {self.temporal.compute_dtype!r}.")

    @property
    def regression_enabled(self) -> bool:
        return self.regression.enabled

    @property
    def temporal_enabled(self) -> bool:
        return self.temporal.enabled

    @property
    def audio_regression_enabled(self) -> bool:
        return self.audio_regression.enabled

    def regression_loss(
        self,
        raw_loss: torch.Tensor,
        step_index: int,
        global_mean_loss: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Weight a local regression loss and update the synchronized EMA.

        ``raw_loss`` is the current local sample/minibatch loss used by the
        adaptive weight in Eq. (7). ``global_mean_loss`` is detached and
        reduced by the trainer only to keep the EMA baseline identical across
        data-parallel ranks. Using the previous EMA for Eq. (7), then updating
        it, follows the paper rather than the public reference code's reversed
        update order.
        """

        if not self.regression_enabled:
            raise RuntimeError("ADV adaptive regression is disabled.")
        step_index = int(step_index)
        if not 0 <= step_index < self.num_inference_steps:
            raise IndexError(f"ADV regression step {step_index} is outside [0, {self.num_inference_steps}).")
        current = float(global_mean_loss)
        previous = self.regression_ema[step_index]
        if previous is None:
            # There is no historical mean yet. Every microbatch in this
            # optimizer step starts at the neutral weight; their synchronized
            # mean becomes the initial cache value when the step commits.
            adaptive_weight = torch.full(
                (),
                0.5,
                device=raw_loss.device,
                dtype=torch.float32,
            )
        else:
            # The paper downweights *the current sample* when it deviates from
            # the timestep's historical mean. Do not replace the local loss
            # with its cross-rank mean here: that would assign every DP sample
            # the same weight and remove adaptive outlier suppression.
            delta = raw_loss.detach().to(torch.float32) - previous
            adaptive_weight = 1.0 - torch.sigmoid(self.regression.sigmoid_scale * delta)
        weighted = self.regression.weight * adaptive_weight * raw_loss
        self._pending_regression_sum[step_index] += current
        self._pending_regression_count[step_index] += 1
        return weighted, adaptive_weight

    def commit_regression_ema(self) -> None:
        """Commit one EMA update per student optimizer step.

        Gradient accumulation is an implementation detail, so every
        microbatch in an optimizer step must use the same historical cache.
        We average synchronized microbatch losses per sampled timestep and
        update each touched cache entry once after the optimizer step.
        """

        if not self.regression_enabled:
            return
        decay = self.regression.ema_decay
        for step_index, count in enumerate(self._pending_regression_count):
            if count == 0:
                continue
            current = self._pending_regression_sum[step_index] / count
            previous = self.regression_ema[step_index]
            self.regression_ema[step_index] = current if previous is None else decay * previous + (1.0 - decay) * current
            self.regression_updates[step_index] += 1
            self._pending_regression_sum[step_index] = 0.0
            self._pending_regression_count[step_index] = 0

    def temporal_loss(
        self,
        generated_video_rows: torch.Tensor,
        num_latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted loss, raw loss, and mean temporal variance."""

        if not self.temporal_enabled:
            raise RuntimeError("ADV temporal regularization is disabled.")
        if generated_video_rows.ndim != 3:
            raise ValueError(f"ADV temporal loss expects packed video rows [B,rows,dim], got {tuple(generated_video_rows.shape)}.")
        frames = int(num_latent_frames)
        if frames < 2 or generated_video_rows.shape[1] % frames:
            raise ValueError(f"Cannot split {generated_video_rows.shape[1]} video rows over {frames} latent frames.")
        values = generated_video_rows.reshape(
            generated_video_rows.shape[0],
            frames,
            generated_video_rows.shape[1] // frames,
            generated_video_rows.shape[2],
        )
        dtype = torch.float64 if self.temporal.compute_dtype in {"fp64", "float64"} else torch.float32
        motion_metric = values.to(dtype).var(dim=1, unbiased=False).mean()
        raw_loss = -torch.log(motion_metric + self.temporal.epsilon)
        active = (raw_loss.detach() >= self.temporal.loss_threshold).to(raw_loss.dtype)
        weighted = self.temporal.weight * active * raw_loss
        return weighted, raw_loss, motion_metric

    def state_dict(self) -> dict:
        if any(self._pending_regression_count):
            raise RuntimeError("Cannot checkpoint ADV in the middle of a student optimizer step.")
        return {
            "version": self.state_version,
            "num_inference_steps": self.num_inference_steps,
            "regression_ema": list(self.regression_ema),
            "regression_updates": list(self.regression_updates),
        }

    def load_state_dict(self, state: Mapping):
        steps = int(state.get("num_inference_steps", -1))
        if steps != self.num_inference_steps:
            raise ValueError("ADV checkpoint uses num_inference_steps={} but the current config uses {}.".format(steps, self.num_inference_steps))
        ema = list(state.get("regression_ema", ()))
        updates = list(state.get("regression_updates", ()))
        if len(ema) != steps or len(updates) != steps:
            raise ValueError("ADV checkpoint has malformed regression EMA state.")
        self.regression_ema = [None if item is None else float(item) for item in ema]
        self.regression_updates = [int(item) for item in updates]
        self._pending_regression_sum = [0.0] * self.num_inference_steps
        self._pending_regression_count = [0] * self.num_inference_steps
