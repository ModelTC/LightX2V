from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

import torch

from .base import TrainerTrick, TrickLossResult


class CdmScheduler(Protocol):
    """Scheduler operations required by consistency distribution matching."""

    def sigma_at(
        self,
        step_index: int,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

    def sample_renoise_sigma(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor: ...

    def add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor: ...

    def _expand_to_ndim(
        self,
        value: torch.Tensor,
        ndim: int,
    ) -> torch.Tensor: ...


VelocityPrediction = Callable[
    [torch.Tensor, torch.Tensor, Any],
    torch.Tensor,
]
DmdLoss = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class CdmConfig:
    enabled: bool = False
    weight: float = 1.0
    warmup_iters: int = 0
    norm_clip_min: float = 0.1

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CdmConfig":
        if not isinstance(config, Mapping):
            raise ValueError("training.dmd.cdm must be a mapping.")
        return cls(
            enabled=bool(config.get("enabled", False)),
            weight=float(config.get("weight", 1.0)),
            warmup_iters=int(config.get("warmup_iters", 0)),
            norm_clip_min=float(config.get("norm_clip_min", 0.1)),
        )


@dataclass(frozen=True)
class CdmTrainerConstraints:
    supported: bool
    model_name: str
    supported_model_names: frozenset[str]


@dataclass(frozen=True)
class CdmStepContext:
    trajectory_latent: torch.Tensor
    trajectory_velocity: torch.Tensor
    end_step_index: int
    condition: Any
    current_iteration: int | None
    device: torch.device
    dtype: torch.dtype
    scheduler: CdmScheduler
    predict_student_velocity: VelocityPrediction
    predict_fake_velocity: VelocityPrediction
    predict_teacher_velocity: VelocityPrediction
    prepare_fake_model: Callable[[], None]
    dmd_loss: DmdLoss


class CdmTrick(
    TrainerTrick[
        None,
        CdmStepContext,
        None,
        None,
    ]
):
    """Consistency distribution matching for image DMD trainers."""

    name = "consistency_distribution_matching"

    def __init__(self, config: CdmConfig):
        super().__init__(enabled=config.enabled)
        self.config = config
        self.validate()

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
    ) -> "CdmTrick":
        return cls(CdmConfig.from_mapping(config))

    def validate(self) -> None:
        if self.config.weight < 0:
            raise ValueError("training.dmd.cdm.weight must be non-negative.")
        if self.config.warmup_iters < 0:
            raise ValueError("training.dmd.cdm.warmup_iters must be non-negative.")
        if self.config.norm_clip_min <= 0:
            raise ValueError("training.dmd.cdm.norm_clip_min must be positive.")

    def validate_trainer(
        self,
        constraints: CdmTrainerConstraints,
        trainer_name: str,
    ) -> None:
        if not self.enabled:
            return
        if not constraints.supported:
            raise ValueError(f"{trainer_name} does not support training.dmd.cdm.")
        if constraints.model_name not in constraints.supported_model_names:
            supported = ", ".join(repr(name) for name in sorted(constraints.supported_model_names))
            raise ValueError(f"training.dmd.cdm is only supported for model.name in {{{supported}}}, got {constraints.model_name!r}.")

    def effective_weight(self, current_iteration: int | None) -> float:
        if self.config.warmup_iters <= 0 or current_iteration is None:
            return self.config.weight
        progress = min(
            1.0,
            max(
                0.0,
                float(current_iteration) / float(self.config.warmup_iters),
            ),
        )
        return progress * self.config.weight

    def student_loss(
        self,
        context: CdmStepContext,
    ) -> TrickLossResult:
        weight = self.effective_weight(context.current_iteration)
        if not self.enabled or weight == 0:
            zero = context.trajectory_latent.new_zeros(
                (),
                dtype=torch.float32,
            )
            return TrickLossResult(
                loss=zero,
                metrics={
                    "cdm": zero.detach(),
                    "cdm_weight": weight,
                },
            )

        batch_size = context.trajectory_latent.shape[0]
        trajectory_sigma = context.scheduler.sigma_at(
            context.end_step_index,
            batch_size,
            device=context.device,
            dtype=torch.float32,
        )
        student_sigma = context.scheduler.sample_renoise_sigma(
            batch_size,
            device=context.device,
            dtype=torch.float32,
        )
        trajectory_sigma = context.scheduler._expand_to_ndim(
            trajectory_sigma,
            context.trajectory_velocity.ndim,
        )
        student_sigma_expanded = context.scheduler._expand_to_ndim(
            student_sigma,
            context.trajectory_velocity.ndim,
        )
        student_xt = context.trajectory_latent + (student_sigma_expanded - trajectory_sigma) * context.trajectory_velocity
        student_prediction = context.predict_student_velocity(
            student_xt.to(context.dtype),
            student_sigma,
            context.condition,
        )
        student_x0 = (student_xt - student_sigma_expanded * student_prediction).to(context.dtype)

        teacher_sigma = context.scheduler.sample_renoise_sigma(
            batch_size,
            device=context.device,
            dtype=context.dtype,
        )
        teacher_noise = torch.randn_like(student_x0)
        teacher_xt = context.scheduler.add_noise(
            student_x0,
            teacher_noise,
            teacher_sigma,
        )
        with torch.no_grad():
            context.prepare_fake_model()
            velocity_fake = context.predict_fake_velocity(
                teacher_xt,
                teacher_sigma,
                context.condition,
            )
            velocity_teacher = context.predict_teacher_velocity(
                teacher_xt,
                teacher_sigma,
                context.condition,
            )

        teacher_sigma_expanded = context.scheduler._expand_to_ndim(
            teacher_sigma,
            teacher_xt.ndim,
        )
        x_pred_fake = teacher_xt - teacher_sigma_expanded * velocity_fake
        x_pred_teacher = teacher_xt - teacher_sigma_expanded * velocity_teacher

        raw_loss = context.dmd_loss(
            student_x0,
            x_pred_fake,
            x_pred_teacher,
            norm_clip_min=self.config.norm_clip_min,
        )
        return TrickLossResult(
            loss=weight * raw_loss,
            metrics={
                "cdm": raw_loss.detach(),
                "cdm_weight": weight,
            },
        )
