from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F

from .base import TrainerTrick, TrickLossResult


class DiversityScheduler(Protocol):
    """Scheduler operations required by the diversity trick."""

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        latent_hw: Sequence[int],
        device: torch.device,
    ) -> None: ...

    def sigma_at(
        self,
        step_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

    def step_by_index(
        self,
        velocity: torch.Tensor,
        step_index: int,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


TeacherVelocity = Callable[
    [torch.Tensor, torch.Tensor, Any, Any],
    torch.Tensor,
]
StudentVelocity = Callable[
    [torch.Tensor, torch.Tensor, Any],
    torch.Tensor,
]
ExpandToNdim = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class DiversityConfig:
    enabled: bool = False
    weight: float = 0.05
    teacher_inference_steps: int = 30
    anchor_step: int = 5

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "DiversityConfig":
        if not isinstance(config, Mapping):
            raise ValueError("training.dmd.div_loss must be a mapping.")
        return cls(
            enabled=bool(config.get("enabled", False)),
            weight=float(config.get("weight", 0.05)),
            teacher_inference_steps=int(config.get("teacher_inference_steps", 30)),
            anchor_step=int(config.get("anchor_step", 5)),
        )


@dataclass(frozen=True)
class DiversitySetupContext:
    scheduler: DiversityScheduler | None


@dataclass(frozen=True)
class DiversityTrainerConstraints:
    supported: bool
    rollout_steps: int
    high_step_count: int | None = None


@dataclass(frozen=True)
class DiversityStepContext:
    initial_noise: torch.Tensor
    condition: Any
    negative_condition: Any
    latent_hw: Sequence[int]
    device: torch.device
    dtype: torch.dtype
    predict_teacher_velocity: TeacherVelocity
    predict_student_velocity: StudentVelocity
    student_scheduler: DiversityScheduler
    expand_to_ndim: ExpandToNdim


class DiversityTrick(
    TrainerTrick[
        DiversitySetupContext,
        DiversityStepContext,
        None,
        None,
    ]
):
    """Teacher-anchor velocity matching shared by video trainers."""

    name = "diversity"

    def __init__(self, config: DiversityConfig):
        super().__init__(enabled=config.enabled)
        self.config = config
        self._scheduler: DiversityScheduler | None = None
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "DiversityTrick":
        return cls(DiversityConfig.from_mapping(config))

    def validate(self) -> None:
        if self.config.weight < 0:
            raise ValueError("training.dmd.div_loss.weight must be non-negative.")
        if self.config.teacher_inference_steps < 1:
            raise ValueError("training.dmd.div_loss.teacher_inference_steps must be positive.")
        if not (1 <= self.config.anchor_step <= self.config.teacher_inference_steps):
            raise ValueError("training.dmd.div_loss.anchor_step must be in [1, teacher_inference_steps].")

    def setup(self, context: DiversitySetupContext) -> None:
        if self.enabled and context.scheduler is None:
            raise ValueError("The diversity trick requires a scheduler when enabled.")
        self._scheduler = context.scheduler

    def validate_trainer(
        self,
        constraints: DiversityTrainerConstraints,
        trainer_name: str,
    ) -> None:
        if not self.enabled:
            return
        if not constraints.supported:
            raise ValueError(f"{trainer_name} does not support training.dmd.div_loss.")
        if constraints.rollout_steps < 2:
            raise ValueError(f"{trainer_name} with div_loss enabled requires at least two student denoising steps.")
        if constraints.high_step_count is not None and constraints.high_step_count < 2:
            raise ValueError("training.dmd.div_loss requires at least two High denoising steps.")

    def prepare_models(
        self,
        teacher_transformer: Any,
        student_transformer: Any,
    ) -> None:
        if not self.enabled:
            return
        teacher_transformer.eval()
        student_transformer.train()

    @property
    def scheduler(self) -> DiversityScheduler:
        if self._scheduler is None:
            raise RuntimeError("The diversity trick is not initialized.")
        return self._scheduler

    @property
    def minimum_dmd_step_index(self) -> int:
        return int(self.enabled)

    def _compute_anchor(
        self,
        context: DiversityStepContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scheduler = self.scheduler
        scheduler.set_timesteps(
            self.config.teacher_inference_steps,
            latent_hw=context.latent_hw,
            device=context.device,
        )
        anchor_latent = context.initial_noise.detach()
        with torch.no_grad():
            for step_index in range(self.config.anchor_step):
                sigma = scheduler.sigma_at(
                    step_index,
                    device=context.device,
                    dtype=context.dtype,
                )
                velocity = context.predict_teacher_velocity(
                    anchor_latent,
                    sigma,
                    context.condition,
                    context.negative_condition,
                )
                anchor_latent, _ = scheduler.step_by_index(
                    velocity,
                    step_index,
                    anchor_latent,
                )
            anchor_sigma = scheduler.sigma_at(
                self.config.anchor_step,
                device=context.device,
                dtype=context.dtype,
            )
        return anchor_latent.detach(), anchor_sigma

    def student_loss(
        self,
        context: DiversityStepContext,
    ) -> TrickLossResult:
        if not self.enabled:
            zero = context.initial_noise.new_zeros((), dtype=torch.float32)
            return TrickLossResult(
                loss=zero,
                metrics={"div_loss": zero.detach()},
            )

        initial_noise = context.initial_noise.detach()
        anchor_latent, anchor_sigma = self._compute_anchor(context)
        sigma_first = context.student_scheduler.sigma_at(
            0,
            device=context.device,
            dtype=context.dtype,
        )
        velocity_first = context.predict_student_velocity(
            initial_noise,
            sigma_first,
            context.condition,
        )
        denominator = (
            1.0
            - context.expand_to_ndim(
                anchor_sigma.float(),
                initial_noise.ndim,
            )
        ).clamp_min(1.0e-8)
        with torch.no_grad():
            velocity_target = (initial_noise.float() - anchor_latent.float()) / denominator
        raw_loss = F.mse_loss(
            velocity_first.float(),
            velocity_target.detach(),
            reduction="mean",
        )
        return TrickLossResult(
            loss=self.config.weight * raw_loss,
            metrics={"div_loss": raw_loss.detach()},
        )

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        return {
            "div_loss_enabled": self.enabled,
            "div_loss_weight": self.config.weight,
            "div_loss_teacher_inference_steps": (self.config.teacher_inference_steps),
            "div_loss_anchor_step": self.config.anchor_step,
        }
