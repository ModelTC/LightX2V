from __future__ import annotations

from abc import abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, TypedDict

import torch
from torch import Tensor

from .base import ModelCapability


@dataclass(frozen=True)
class LossResult:
    loss: Tensor
    metrics: Mapping[str, Tensor | float] = field(default_factory=dict)


class TrainingCachePayload(TypedDict):
    inputs: Mapping[str, Any]
    conditioning: Mapping[str, Any]
    meta: Mapping[str, Any]


@dataclass(frozen=True)
class SFTStepContext:
    noise_scheduler: Any
    running_dtype: torch.dtype
    broadcast: Callable[[Any], Any]


class FlowMatchingSFTCapability(ModelCapability):
    def encode_training_cache(self, batch: Mapping[str, Any]) -> TrainingCachePayload:
        """Encode the static sample data consumed by flow-matching training."""
        raise NotImplementedError(f"{type(self).__name__} does not support training-cache encoding.")

    @abstractmethod
    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: SFTStepContext,
    ) -> LossResult:
        """Compute a differentiable supervised flow-matching loss."""


@dataclass(frozen=True)
class TeacherForcingStepContext:
    scheduler: Any
    running_dtype: torch.dtype
    num_frame_per_chunk: int
    noise_augmentation_max_timestep: int
    broadcast: Callable[[Any], Any]


class TeacherForcingCapability(ModelCapability):
    def encode_training_cache(self, batch: Mapping[str, Any]) -> TrainingCachePayload:
        """Encode the static target and condition used by teacher forcing."""
        raise NotImplementedError(f"{type(self).__name__} does not support training-cache encoding.")

    @abstractmethod
    def compute_loss(
        self,
        batch: Mapping[str, Any],
        context: TeacherForcingStepContext,
    ) -> LossResult:
        """Compute a differentiable teacher-forcing loss."""


class ConsistencyModelCapability(ModelCapability):
    def encode_training_cache(self, batch: Mapping[str, Any]) -> TrainingCachePayload:
        """Encode the static sample data consumed by consistency training."""
        raise NotImplementedError(f"{type(self).__name__} does not support training-cache encoding.")

    @abstractmethod
    def configure(self, features: Collection[str]) -> None:
        """Install model structure required by the selected objective."""

    @abstractmethod
    def restore_trainable_auxiliary(self) -> None:
        """Make objective-specific parameters trainable after backbone setup."""

    @abstractmethod
    def auxiliary_parameter_names(self) -> tuple[str, ...]:
        """Return configured denoiser keys stored beside parameter-efficient weights."""

    @abstractmethod
    def encode_latent(self, batch):
        """Encode one training sample into the objective's clean state."""

    @abstractmethod
    def encode_condition(self, batch):
        """Encode one training sample's conditioning inputs."""

    @abstractmethod
    def sampling_latent_hw(self, batch, clean) -> tuple[int, int]:
        """Return latent spatial geometry used by resolution-aware schedulers."""

    @abstractmethod
    def predict(self, request, path):
        """Run the model and convert its output to the requested parameterization."""

    @abstractmethod
    def predict_log_variance(self, time):
        """Predict the learned loss log-variance required by sCM."""

    @abstractmethod
    def set_frozen(self, training: bool = False) -> None:
        """Freeze the denoiser while selecting its train/eval execution mode."""

    @abstractmethod
    def denoiser(self):
        """Return the module owned by this capability."""


@dataclass(frozen=True)
class DistributionMatchingProfile:
    """Feature and precision constraints published by a distribution-matching adapter."""

    supports_guidance: bool = True
    supports_ida: bool = True
    supports_diversity: bool = True
    supports_real_data_fake: bool = True
    supports_warped_denoising_schedule: bool = True
    default_latent_dtype: torch.dtype | None = None
    # ``None`` keeps the capability available to every DMD-family trainer.
    supported_training_methods: frozenset[str] | None = None


class DistributionMatchingCapability(ModelCapability):
    """Model operations required by distribution-matching algorithms."""

    @property
    def profile(self) -> DistributionMatchingProfile:
        return DistributionMatchingProfile()

    def encode_training_cache(self, batch: Mapping[str, Any]) -> TrainingCachePayload:
        """Encode the static sample data consumed by distribution matching."""
        raise NotImplementedError(f"{type(self).__name__} does not support training-cache encoding.")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abstractmethod
    def default_negative_prompt(self) -> str | None:
        pass

    @property
    @abstractmethod
    def default_lora_target_modules(self):
        pass

    @property
    @abstractmethod
    def generation_shape_dimensions(self) -> int:
        """Number of dimensions required by one generation shape."""
        pass

    @abstractmethod
    def latent_shape(
        self,
        batch: Mapping[str, Any],
        generation_shapes,
        broadcast: Callable[[Any], Any],
    ):
        pass

    @abstractmethod
    def encode_conditions(
        self,
        batch: Mapping[str, Any],
        negative_prompt: str | None,
        guidance_scale: float,
        broadcast: Callable[[Any], Any],
    ):
        pass

    @abstractmethod
    def predict_velocity(self, latents, sigma, condition):
        pass

    @abstractmethod
    def predict_guided_velocity(
        self,
        latents,
        sigma,
        condition,
        negative_condition,
        guidance_scale,
        cfg_norm,
    ):
        pass

    @abstractmethod
    def initial_latents(self, latent_shape, dtype, broadcast):
        pass

    @abstractmethod
    def latent_hw(self, latent_shape):
        pass

    @abstractmethod
    def random_noise_like(self, latents, dtype, broadcast):
        pass

    @abstractmethod
    def add_noise(self, scheduler, latents, noise, sigma):
        pass

    @abstractmethod
    def training_target(self, latents, noise):
        pass

    @abstractmethod
    def step(self, scheduler, velocity, step_index, sample):
        pass

    @abstractmethod
    def x0_from_velocity(self, sample, velocity, sigma):
        pass

    @abstractmethod
    def regression_loss(self, prediction, target):
        pass

    @abstractmethod
    def dmd_loss(self, latents, fake_x0, teacher_x0):
        pass

    @abstractmethod
    def detach(self, value):
        pass

    @abstractmethod
    def to_dtype(self, value, dtype):
        pass

    @abstractmethod
    def extract_real_latents(self, batch, dtype, broadcast):
        pass

    @abstractmethod
    def set_training(self, enabled: bool) -> None:
        pass

    @abstractmethod
    def denoiser(self):
        pass


@dataclass(frozen=True)
class AutoregressiveRolloutContext:
    denoising_steps: Tensor
    denoising_scheduler: Any
    trajectory_scheduler: Any
    running_dtype: torch.dtype
    frames_per_chunk: int
    same_step_across_blocks: bool
    context_noise: float
    sequence_parallel_cache: bool
    grad_enabled: bool


class AutoregressiveDistributionMatchingCapability(ModelCapability):
    @abstractmethod
    def rollout(
        self,
        condition,
        latent_shape,
        initial_latents,
        context: AutoregressiveRolloutContext,
    ):
        """Run a differentiable chunk-wise autoregressive rollout."""


class TrainableModelCapability(ModelCapability):
    @abstractmethod
    def configure(self, train_type: str, lora_config: Mapping[str, Any]) -> None:
        pass

    @abstractmethod
    def restore(self, train_type: str) -> None:
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        pass

    @abstractmethod
    def set_eval(self) -> None:
        pass

    @abstractmethod
    def log_structure(self) -> None:
        pass


class ParallelCapability(ModelCapability):
    @abstractmethod
    def apply(self, config) -> None:
        pass

    @abstractmethod
    def set_gradient_sync(self, enabled: bool) -> None:
        pass

    @abstractmethod
    def is_fsdp(self) -> bool:
        pass

    @abstractmethod
    def state_module(self):
        pass


class CheckpointCapability(ModelCapability):
    @abstractmethod
    def save_weights(self, save_dir, train_type) -> None:
        pass

    @abstractmethod
    def load_weights(self, save_dir, train_type) -> None:
        pass

    @abstractmethod
    def save_consolidated(self, output_path) -> None:
        pass

    @abstractmethod
    def save_full_model(self, output_path) -> None:
        pass
