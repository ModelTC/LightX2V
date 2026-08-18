from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import Tensor

_PREDICTION_TYPE_ALIASES = {
    "flow": "velocity",
    "velocity": "velocity",
    "x0": "x0",
    "sample": "x0",
    "noise": "noise",
    "epsilon": "noise",
}


def canonical_prediction_type(prediction_type: str) -> str:
    try:
        return _PREDICTION_TYPE_ALIASES[prediction_type.lower()]
    except (AttributeError, KeyError) as exc:
        supported = ", ".join(sorted(_PREDICTION_TYPE_ALIASES))
        raise ValueError(f"Unsupported prediction type {prediction_type!r}; expected one of: {supported}.") from exc


def expand_time(time: Tensor, ndim: int) -> Tensor:
    """Expand a scalar or batch time tensor over non-batch dimensions."""
    if time.ndim == 0:
        time = time.reshape(1)
    return time.reshape(time.shape[0], *([1] * (ndim - 1)))


def require_singleton_clean(clean: Tensor) -> None:
    if clean.ndim == 0 or clean.shape[0] != 1:
        raise ValueError("Consistency training only supports physical batch size 1.")


class RectifiedFlowPath:
    """Conversions for the straight path x_t=(1-t)x_0+t*noise."""

    def interpolate(self, clean: Tensor, noise: Tensor, time: Tensor) -> Tensor:
        time = expand_time(time, clean.ndim).to(device=clean.device, dtype=clean.dtype)
        return (1.0 - time) * clean + time * noise

    def euler_step(self, sample: Tensor, velocity: Tensor, time: Tensor, next_time: Tensor) -> Tensor:
        delta = expand_time(next_time - time, sample.ndim).to(device=sample.device, dtype=sample.dtype)
        return (sample + delta * velocity.to(dtype=sample.dtype)).to(dtype=sample.dtype)

    def convert_prediction(
        self,
        sample: Tensor,
        prediction: Tensor,
        time: Tensor,
        *,
        source_type: str,
        target_type: str,
    ) -> Tensor:
        source_type = canonical_prediction_type(source_type)
        target_type = canonical_prediction_type(target_type)
        if source_type == target_type:
            return prediction

        time_expanded = expand_time(time, sample.ndim).to(device=sample.device, dtype=sample.dtype)
        prediction = prediction.to(dtype=sample.dtype)

        if source_type == "velocity":
            velocity = prediction
        elif source_type == "x0":
            denominator = torch.clamp(time_expanded, min=torch.finfo(sample.dtype).tiny)
            velocity = (sample - prediction) / denominator
        else:  # noise
            denominator = torch.clamp(1.0 - time_expanded, min=torch.finfo(sample.dtype).tiny)
            velocity = (prediction - sample) / denominator

        if target_type == "velocity":
            return velocity
        if target_type == "x0":
            return sample - time_expanded * velocity
        return sample + (1.0 - time_expanded) * velocity


@dataclass(frozen=True)
class DenoiserRequest:
    """Model-agnostic request issued by a consistency objective."""

    sample: Tensor
    time: Tensor
    condition: Any
    prediction_type: str = "velocity"
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


class CapabilityDenoiser:
    """Adapt a consistency-model capability to the objective interface."""

    def __init__(self, capability, path: RectifiedFlowPath):
        self.capability = capability
        self.path = path

    def predict(self, request: DenoiserRequest) -> Tensor:
        return self.capability.predict(request, self.path)

    def predict_log_variance(self, time: Tensor) -> Tensor:
        """Return the model-owned scalar log-variance head used by sCM."""
        return self.capability.predict_log_variance(time)


@dataclass(frozen=True)
class ConsistencyBatch:
    clean: Tensor
    condition: Any
    negative_condition: Any = None


@dataclass(frozen=True)
class ConsistencyStepContext:
    iteration: int
    global_batch_size: int
    latent_hw: tuple[int, int]


@dataclass
class ObjectiveOutput:
    loss: Tensor
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceModelSpec:
    """Description of an algorithm-owned frozen model.

    ``checkpoint`` is a LightX2V training checkpoint directory.  Model
    overrides are read from ``model.<role>`` so a future objective can change
    the reference architecture without teaching the trainer about it.
    """

    role: str
    checkpoint: str
    training_mode: bool = False


class ConsistencyObjective(ABC):
    """Extension point implemented by CM, sCM, TCM, PCM, MeanFlow, and others."""

    algorithm_name = "base"
    requires_teacher = False
    requires_negative_condition = False
    negative_prompt: Optional[str] = None
    model_capabilities: frozenset[str] = frozenset()

    @property
    def reference_model_specs(self) -> tuple[ReferenceModelSpec, ...]:
        return ()

    @property
    def student_initialization_checkpoint(self) -> Optional[str]:
        return None

    @abstractmethod
    def sample_training_state(
        self,
        clean: Tensor,
        scheduler,
        context: ConsistencyStepContext,
    ) -> Mapping[str, Tensor]:
        """Sample all stochastic state before sequence-parallel broadcast."""

    @abstractmethod
    def compute(
        self,
        batch: ConsistencyBatch,
        training_state: Mapping[str, Tensor],
        student: CapabilityDenoiser,
        teacher: Optional[CapabilityDenoiser] = None,
        references: Optional[Mapping[str, CapabilityDenoiser]] = None,
    ) -> ObjectiveOutput:
        """Compute a differentiable scalar loss and detached metrics."""
