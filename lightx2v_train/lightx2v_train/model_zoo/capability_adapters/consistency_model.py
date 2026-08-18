"""Composable consistency-model extensions for rectified-flow backbones."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Collection
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .common import GenericConsistencyModelCapability


def _resolve_attribute(root, path: str):
    value = root
    for name in path.split("."):
        try:
            value = getattr(value, name)
        except AttributeError as exc:
            raise AttributeError(f"Cannot resolve {path!r}: {type(value).__name__} has no attribute {name!r}.") from exc
    return value


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except StopIteration as exc:
        raise ValueError(f"{type(module).__name__} must contain parameters to define its device and dtype.") from exc
    return parameter.device, parameter.dtype


class TimeEmbeddingAdapter(ABC):
    """Describe how one denoiser creates its primary timestep embedding."""

    @abstractmethod
    def hook_modules(self, denoiser: nn.Module) -> tuple[nn.Module, ...]:
        """Return modules whose outputs receive endpoint conditioning."""

    @abstractmethod
    def base_embedder(self, denoiser: nn.Module) -> nn.Module:
        """Return the train-time timestep MLP used by the pretrained model."""

    def build_endpoint_embedder(self, denoiser: nn.Module) -> nn.Module:
        return copy.deepcopy(self.base_embedder(denoiser))

    @abstractmethod
    def embedding_dimension(self, denoiser: nn.Module) -> int:
        """Return the final timestep embedding width."""

    @abstractmethod
    def encode(
        self,
        denoiser: nn.Module,
        embedder: nn.Module,
        time: Tensor,
    ) -> Tensor:
        """Encode normalized flow time with ``embedder``."""

    def add_endpoint_embedding(
        self,
        denoiser: nn.Module,
        endpoint_embedder: nn.Module,
        endpoint_time: Tensor,
        hook_module: nn.Module,
        output,
    ):
        del hook_module
        if not torch.is_tensor(output):
            raise TypeError(f"Endpoint conditioning requires a tensor timestep embedding, got {type(output).__name__}.")
        embedding = self.encode(denoiser, endpoint_embedder, endpoint_time)
        while embedding.ndim < output.ndim:
            embedding = embedding.unsqueeze(-2)
        return output + embedding.to(device=output.device, dtype=output.dtype)


@dataclass(frozen=True)
class ProjectedTimeEmbeddingAdapter(TimeEmbeddingAdapter):
    """Adapter for backbones exposing separate time projection and MLP modules."""

    hook_module_path: str
    projection_module_path: str
    embedding_module_path: str
    embedding_dimension_path: str
    time_scale: float = 1.0

    def hook_modules(self, denoiser: nn.Module) -> tuple[nn.Module, ...]:
        return (_resolve_attribute(denoiser, self.hook_module_path),)

    def base_embedder(self, denoiser: nn.Module) -> nn.Module:
        return _resolve_attribute(denoiser, self.embedding_module_path)

    def embedding_dimension(self, denoiser: nn.Module) -> int:
        dimension = int(_resolve_attribute(denoiser, self.embedding_dimension_path))
        if dimension <= 0:
            raise ValueError(f"Timestep embedding dimension must be positive, got {dimension}.")
        return dimension

    def encode(
        self,
        denoiser: nn.Module,
        embedder: nn.Module,
        time: Tensor,
    ) -> Tensor:
        projection = _resolve_attribute(denoiser, self.projection_module_path)
        device, dtype = _module_device_dtype(embedder)
        scaled_time = time.to(device=device)
        if self.time_scale != 1.0:
            scaled_time = scaled_time * self.time_scale
        projected = projection(scaled_time)
        return embedder(projected.to(device=device, dtype=dtype))


@dataclass(frozen=True)
class SinusoidalTimeEmbeddingAdapter(TimeEmbeddingAdapter):
    """Adapter for Wan-style MLPs whose sinusoidal projection lives in forward."""

    embedding_module_path: str
    embedding_dimension_path: str
    frequency_dimension_path: str
    time_scale: float = 1.0

    def hook_modules(self, denoiser: nn.Module) -> tuple[nn.Module, ...]:
        return (self.base_embedder(denoiser),)

    def base_embedder(self, denoiser: nn.Module) -> nn.Module:
        return _resolve_attribute(denoiser, self.embedding_module_path)

    def embedding_dimension(self, denoiser: nn.Module) -> int:
        dimension = int(_resolve_attribute(denoiser, self.embedding_dimension_path))
        if dimension <= 0:
            raise ValueError(f"Timestep embedding dimension must be positive, got {dimension}.")
        return dimension

    def encode(
        self,
        denoiser: nn.Module,
        embedder: nn.Module,
        time: Tensor,
    ) -> Tensor:
        frequency_dimension = int(_resolve_attribute(denoiser, self.frequency_dimension_path))
        if frequency_dimension <= 0 or frequency_dimension % 2:
            raise ValueError(f"Sinusoidal frequency dimension must be a positive even integer, got {frequency_dimension}.")

        device, dtype = _module_device_dtype(embedder)
        position = time.to(device=device).reshape(-1)
        if self.time_scale != 1.0:
            position = position * self.time_scale
        projection = self._sinusoidal_embedding(frequency_dimension, position)
        return embedder(projection.to(device=device, dtype=dtype))

    @staticmethod
    def _sinusoidal_embedding(dimension: int, position: Tensor) -> Tensor:
        half = dimension // 2
        position = position.to(torch.float64)
        frequencies = torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float64) / half,
        )
        phases = torch.outer(position, frequencies)
        return torch.cat((torch.cos(phases), torch.sin(phases)), dim=1)


class TimeConditionedConsistencyModelCapability(GenericConsistencyModelCapability):
    """Generic CM adapter with optional endpoint conditioning and loss variance.

    The capability owns objective-specific state and modules. The model wrapper
    only supplies a small, immutable description of its timestep embedding.
    """

    _SUPPORTED_FEATURES = frozenset({"endpoint_time", "log_variance"})

    def __init__(
        self,
        model,
        time_adapter: TimeEmbeddingAdapter,
        *,
        endpoint_module_name: str = "consistency_endpoint_embedder",
        log_variance_module_name: str = "consistency_logvar_head",
    ) -> None:
        super().__init__(model)
        self._time_adapter = time_adapter
        self._endpoint_module_name = endpoint_module_name
        self._log_variance_module_name = log_variance_module_name
        self._configured_features: set[str] = set()
        self._endpoint_time: Tensor | None = None
        self._endpoint_hooks = []

    def configure(self, features: Collection[str]) -> None:
        features = frozenset(features)
        unsupported = features - self._SUPPORTED_FEATURES
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise NotImplementedError(f"{type(self).__name__} does not support consistency features: {names}.")

        denoiser = self.denoiser()
        if "log_variance" in features and not hasattr(denoiser, self._log_variance_module_name):
            source_embedder = self._time_adapter.base_embedder(denoiser)
            device, dtype = _module_device_dtype(source_embedder)
            head = nn.Linear(
                self._time_adapter.embedding_dimension(denoiser),
                1,
                device=device,
                dtype=dtype,
            )
            setattr(denoiser, self._log_variance_module_name, head)

        if "endpoint_time" in features:
            if not hasattr(denoiser, self._endpoint_module_name):
                endpoint_embedder = self._time_adapter.build_endpoint_embedder(denoiser)
                setattr(denoiser, self._endpoint_module_name, endpoint_embedder)
            if not self._endpoint_hooks:
                self._endpoint_hooks = [module.register_forward_hook(self._add_endpoint_time_embedding) for module in self._time_adapter.hook_modules(denoiser)]

        self._configured_features.update(features)

    def restore_trainable_auxiliary(self) -> None:
        denoiser = self.denoiser()
        for module_name in self._configured_module_names():
            getattr(denoiser, module_name).requires_grad_(True)

    def auxiliary_parameter_names(self) -> tuple[str, ...]:
        module_names = self._configured_module_names()
        return tuple(name for name, _ in self.denoiser().named_parameters() if name.partition(".")[0] in module_names)

    def predict(self, request, path):
        with self._configured_prediction(request) as prepared_request:
            return super().predict(prepared_request, path)

    @contextmanager
    def _configured_prediction(self, request):
        model_kwargs = dict(request.model_kwargs)
        endpoint_time = model_kwargs.pop("endpoint_time", None)
        if endpoint_time is not None and "endpoint_time" not in self._configured_features:
            raise RuntimeError("endpoint_time was provided before endpoint conditioning was configured.")

        request = replace(request, model_kwargs=model_kwargs)
        with self._use_endpoint_time(endpoint_time):
            yield request

    def predict_log_variance(self, time: Tensor) -> Tensor:
        if "log_variance" not in self._configured_features:
            raise RuntimeError("The consistency log-variance head has not been configured.")
        denoiser = self.denoiser()
        head = getattr(denoiser, self._log_variance_module_name)
        embedding = self._time_adapter.encode(
            denoiser,
            self._time_adapter.base_embedder(denoiser),
            time,
        )
        return head(embedding.to(device=head.weight.device, dtype=head.weight.dtype))

    def _configured_module_names(self) -> set[str]:
        names = set()
        if "endpoint_time" in self._configured_features:
            names.add(self._endpoint_module_name)
        if "log_variance" in self._configured_features:
            names.add(self._log_variance_module_name)
        return names

    @contextmanager
    def _use_endpoint_time(self, endpoint_time):
        previous = self._endpoint_time
        self._endpoint_time = endpoint_time
        try:
            yield
        finally:
            self._endpoint_time = previous

    def _add_endpoint_time_embedding(self, module, inputs, output):
        del inputs
        if self._endpoint_time is None:
            return output
        denoiser = self.denoiser()
        endpoint_embedder = getattr(denoiser, self._endpoint_module_name)
        return self._time_adapter.add_endpoint_embedding(
            denoiser,
            endpoint_embedder,
            self._endpoint_time,
            module,
            output,
        )
