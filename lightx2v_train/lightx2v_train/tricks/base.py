from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Sequence, TypeVar

import torch

SetupContextT = TypeVar("SetupContextT")
StudentContextT = TypeVar("StudentContextT")
FakeContextT = TypeVar("FakeContextT")
PostStudentContextT = TypeVar("PostStudentContextT")


@dataclass(frozen=True)
class TrickLossResult:
    """A weighted loss contribution and detached logging metrics."""

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | float] = field(default_factory=dict)


class TrainerTrick(
    ABC,
    Generic[
        SetupContextT,
        StudentContextT,
        FakeContextT,
        PostStudentContextT,
    ],
):
    """Composable training behavior with explicit runtime dependencies."""

    name = "trainer_trick"

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration that is independent of a trainer."""

    def setup(self, context: SetupContextT) -> None:
        """Initialize optional runtime resources."""

    def student_loss(
        self,
        context: StudentContextT,
    ) -> TrickLossResult | None:
        """Return an optional student-side loss contribution."""
        return None

    def fake_loss(
        self,
        context: FakeContextT,
    ) -> TrickLossResult | None:
        """Return an optional fake-model loss contribution."""
        return None

    def after_student_step(
        self,
        context: PostStudentContextT,
    ) -> None:
        """Apply an optional update after the student optimizer step."""

    def named_models(self) -> Mapping[str, Any]:
        """Return models owned by this trick."""
        return {}

    def named_optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        """Return optimizers owned by this trick."""
        return {}

    def named_schedulers(self) -> Mapping[str, Any]:
        """Return learning-rate schedulers owned by this trick."""
        return {}

    def named_trainable_parameters(
        self,
    ) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return trainable parameters grouped by role."""
        return {}

    def set_gradient_sync(
        self,
        enabled: bool,
        role: str | None = None,
    ) -> None:
        """Control distributed gradient synchronization for owned models."""

    def checkpoint_metadata(self) -> Mapping[str, Any]:
        """Return serializable metadata used to validate resume."""
        return {"enabled": self.enabled}

    def state_dict(self) -> Mapping[str, Any]:
        """Return mutable runtime state owned by this trick."""
        return {}

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> None:
        """Restore mutable runtime state owned by this trick."""
        if strict and state_dict:
            unexpected = ", ".join(sorted(state_dict))
            raise RuntimeError(f"{self.name} received unexpected state keys: {unexpected}")
