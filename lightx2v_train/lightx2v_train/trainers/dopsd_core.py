"""DOPSD configuration and objective, independent of any model family.

The implementation follows the privileged editing-teacher / text-only
student setup introduced by https://github.com/vvvvvjdy/D-OPSD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from lightx2v_train.model_capabilities import (
    AdapterBankCapability,
    DopsdCapability,
)


@dataclass(frozen=True)
class DopsdConfig:
    num_training_steps: int = 4
    ema_decay: float = 0.999
    ema_decay_warmup: float | None = None
    ema_decay_warmup_iters: int = 0
    step_loss_weights: tuple[float, ...] | None = None
    student_adapter: str = "student"
    teacher_adapter: str = "teacher"
    edit_sys_prompt: str = "The output must be exactly the same as the reference image."
    teacher_use_dataset_prompt: bool = False
    trajectory_every_iters: int | None = None

    @classmethod
    def from_training_config(cls, training_config: Mapping[str, Any]) -> "DopsdConfig":
        train_type = training_config.get("train_type")
        if train_type != "lora":
            raise ValueError(
                "DOPSD trains a student LoRA and an EMA teacher LoRA; "
                f"training.train_type must be 'lora', got {train_type!r}."
            )
        raw = training_config.get("dopsd", {}) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("training.dopsd must be a mapping.")

        raw_weights = raw.get("step_loss_weights")
        if raw_weights is None:
            weights = None
        else:
            if isinstance(raw_weights, (str, bytes, Mapping)):
                raise ValueError("training.dopsd.step_loss_weights must be a sequence of numbers.")
            try:
                weights = tuple(float(value) for value in raw_weights)
            except TypeError as error:
                raise ValueError("training.dopsd.step_loss_weights must be a sequence of numbers.") from error
        warmup = raw.get("ema_decay_warmup")
        trajectory_interval = raw.get("trajectory_every_iters")
        config = cls(
            num_training_steps=int(raw.get("num_training_steps", 4)),
            ema_decay=float(raw.get("ema_decay", 0.999)),
            ema_decay_warmup=None if warmup is None else float(warmup),
            ema_decay_warmup_iters=int(raw.get("ema_decay_warmup_iters", 0)),
            step_loss_weights=weights,
            student_adapter=str(raw.get("student_adapter", "student")).strip(),
            teacher_adapter=str(raw.get("teacher_adapter", "teacher")).strip(),
            edit_sys_prompt=str(
                raw.get(
                    "edit_sys_prompt",
                    "The output must be exactly the same as the reference image.",
                )
            ),
            teacher_use_dataset_prompt=bool(raw.get("teacher_use_dataset_prompt", False)),
            trajectory_every_iters=None if trajectory_interval is None else int(trajectory_interval),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.num_training_steps <= 0:
            raise ValueError(f"training.dopsd.num_training_steps must be positive, got {self.num_training_steps}.")
        self._validate_decay("ema_decay", self.ema_decay)
        if self.ema_decay_warmup is not None:
            self._validate_decay("ema_decay_warmup", self.ema_decay_warmup)
        if self.ema_decay_warmup_iters < 0:
            raise ValueError("training.dopsd.ema_decay_warmup_iters may not be negative.")
        if not self.student_adapter or not self.teacher_adapter:
            raise ValueError("DOPSD adapter names must be non-empty.")
        if self.student_adapter == self.teacher_adapter:
            raise ValueError("DOPSD student_adapter and teacher_adapter must be different.")
        if "." in self.student_adapter or "." in self.teacher_adapter:
            raise ValueError("DOPSD adapter names may not contain '.'.")
        if self.trajectory_every_iters is not None and self.trajectory_every_iters <= 0:
            raise ValueError("training.dopsd.trajectory_every_iters must be positive when set.")

        if self.step_loss_weights is not None:
            if not self.step_loss_weights:
                raise ValueError("training.dopsd.step_loss_weights may not be empty.")
            if any(not math.isfinite(weight) or weight < 0.0 for weight in self.step_loss_weights):
                raise ValueError("training.dopsd.step_loss_weights must be finite and non-negative.")
            if sum(self.loss_weights(self.num_training_steps)) <= 0.0:
                raise ValueError("DOPSD loss weights must contain at least one positive value.")

    @staticmethod
    def _validate_decay(name: str, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError(f"training.dopsd.{name} must be in [0, 1), got {decay}.")

    def loss_weights(self, num_steps: int) -> tuple[float, ...]:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}.")
        if self.step_loss_weights is None:
            return (1.0,) * num_steps
        weights = list(self.step_loss_weights[:num_steps])
        if len(weights) < num_steps:
            weights.extend([weights[-1]] * (num_steps - len(weights)))
        return tuple(weights)

    def ema_decay_at(self, iteration: int) -> float:
        if self.ema_decay_warmup is not None and 0 < iteration <= self.ema_decay_warmup_iters:
            return self.ema_decay_warmup
        return self.ema_decay

    def teacher_prompts(self, prompts) -> list[str]:
        base_prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not base_prompts:
            raise ValueError("DOPSD requires at least one prompt.")
        suffix = self.edit_sys_prompt.strip()
        if not suffix:
            return [str(prompt) for prompt in base_prompts]
        if not self.teacher_use_dataset_prompt:
            return [suffix] * len(base_prompts)
        return [f"{prompt} {suffix}".strip() for prompt in base_prompts]

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "version": 2,
            "num_training_steps": self.num_training_steps,
            "student_adapter": self.student_adapter,
            "teacher_adapter": self.teacher_adapter,
        }


@dataclass(frozen=True)
class DopsdRolloutResult:
    state: Tensor
    state_ids: Tensor
    height: int
    width: int


@dataclass(frozen=True)
class DopsdLossResult:
    loss: Tensor
    student_trajectory: tuple[Tensor, ...] = ()
    teacher_trajectory: tuple[Tensor, ...] = ()
    state_ids: Tensor | None = None


class DopsdObjective:
    """The model-independent on-policy student/EMA-teacher objective."""

    def __init__(
        self,
        model: DopsdCapability,
        adapters: AdapterBankCapability,
        scheduler,
        config: DopsdConfig,
        running_dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.adapters = adapters
        self.scheduler = scheduler
        self.config = config
        self.running_dtype = running_dtype

    @staticmethod
    def _expand_time(time: Tensor, state: Tensor) -> Tensor:
        if time.ndim == 0:
            time = time.reshape(1)
        if time.shape[0] not in (1, state.shape[0]):
            raise ValueError(
                f"Time batch dimension must be 1 or {state.shape[0]}, got {tuple(time.shape)}."
            )
        return time.reshape(time.shape[0], *([1] * (state.ndim - 1)))

    def _schedule(self, num_steps: int, latent_hw: tuple[int, int], state: Tensor) -> Tensor:
        return self.scheduler.build_inference_sigmas(
            num_steps,
            latent_hw=latent_hw,
        ).to(device=state.device, dtype=self.running_dtype)

    def compute_loss(self, batch: Mapping[str, Any], collect_trajectory: bool = False) -> DopsdLossResult:
        prompts = batch.get("conditioning", {}).get("prompt")
        if prompts is None:
            raise ValueError("DOPSD requires conditioning.prompt.")

        with torch.no_grad():
            prepared = self.model.prepare_training_batch(
                batch,
                self.config.teacher_prompts(prompts),
                self.running_dtype,
            )
            sigmas = self._schedule(
                self.config.num_training_steps,
                prepared.latent_hw,
                prepared.initial_state,
            )

        state = prepared.initial_state
        weights = self.config.loss_weights(sigmas.numel() - 1)
        total_loss: Tensor | None = None
        student_trajectory = []
        teacher_trajectory = []

        for step_index, (time, next_time) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            time = time.reshape(1)
            next_time = next_time.reshape(1)
            # Truncate the rollout graph between steps while retaining the
            # differentiable student regression performed at every step.
            state = state.detach().requires_grad_(True)

            with torch.no_grad(), self.adapters.activate(
                self.config.teacher_adapter,
                training=False,
            ):
                teacher_velocity = self.model.predict_velocity(
                    state,
                    time,
                    prepared.teacher_condition,
                    prepared.state_ids,
                    reference=prepared.teacher_reference,
                )
                teacher_x0 = state - self._expand_time(time, state) * teacher_velocity

            with self.adapters.activate(self.config.student_adapter, training=True):
                student_velocity = self.model.predict_velocity(
                    state,
                    time,
                    prepared.student_condition,
                    prepared.state_ids,
                )
            student_x0 = state - self._expand_time(time, state) * student_velocity
            step_loss = (student_x0.float() - teacher_x0.float()).square().mean()
            weighted_loss = weights[step_index] * step_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
            state = state + self._expand_time(next_time - time, state) * student_velocity

            if collect_trajectory:
                student_trajectory.append(student_x0.detach())
                teacher_trajectory.append(teacher_x0.detach())

        if total_loss is None:
            raise RuntimeError("DOPSD produced an empty rollout.")
        return DopsdLossResult(
            loss=total_loss / sum(weights),
            student_trajectory=tuple(student_trajectory),
            teacher_trajectory=tuple(teacher_trajectory),
            state_ids=prepared.state_ids if collect_trajectory else None,
        )

    @torch.no_grad()
    def rollout_teacher(
        self,
        reference_image: Tensor,
        teacher_prompts: list[str],
        num_steps: int,
        generator: torch.Generator | None = None,
    ) -> DopsdRolloutResult:
        prepared = self.model.prepare_teacher_batch(
            reference_image,
            teacher_prompts,
            self.running_dtype,
            generator=generator,
        )
        sigmas = self._schedule(num_steps, prepared.latent_hw, prepared.initial_state)
        state = prepared.initial_state
        with self.adapters.activate(self.config.teacher_adapter, training=False):
            for time, next_time in zip(sigmas[:-1], sigmas[1:]):
                time = time.reshape(1)
                velocity = self.model.predict_velocity(
                    state,
                    time,
                    prepared.condition,
                    prepared.state_ids,
                    reference=prepared.reference,
                )
                state = state + self._expand_time(next_time.reshape(1) - time, state) * velocity
        return DopsdRolloutResult(
            state=state,
            state_ids=prepared.state_ids,
            height=prepared.height,
            width=prepared.width,
        )
