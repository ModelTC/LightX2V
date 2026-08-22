from __future__ import annotations

import weakref
from abc import ABC
from typing import TypeVar, cast


class ModelCapability(ABC):
    """Root interface for behavior exposed by a loaded model."""


CapabilityT = TypeVar("CapabilityT", bound=ModelCapability)


class CapabilityNotSupportedError(RuntimeError):
    """Raised when an algorithm is paired with an incompatible model."""


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[type[ModelCapability], ModelCapability] = {}

    def register(
        self,
        interface: type[CapabilityT],
        implementation: CapabilityT,
    ) -> None:
        if not isinstance(implementation, interface):
            raise TypeError(f"{type(implementation).__name__} does not implement {interface.__name__}.")
        if interface in self._capabilities:
            raise RuntimeError(f"Capability already registered: {interface.__name__}.")
        self._capabilities[interface] = implementation

    def supports(self, interface: type[ModelCapability]) -> bool:
        return interface in self._capabilities

    def require(self, interface: type[CapabilityT]) -> CapabilityT:
        implementation = self._capabilities.get(interface)
        if implementation is None:
            raise CapabilityNotSupportedError(f"Missing capability: {interface.__name__}.")
        return cast(CapabilityT, implementation)

    def missing(
        self,
        interfaces: tuple[type[ModelCapability], ...],
    ) -> tuple[type[ModelCapability], ...]:
        return tuple(interface for interface in interfaces if not self.supports(interface))


class CapabilityProvider:
    """Base for model wrappers that explicitly publish loaded capabilities."""

    def __init__(self) -> None:
        self.capabilities = CapabilityRegistry()
        self._capabilities_registered = False

    def ensure_capabilities(self) -> CapabilityRegistry:
        if not self._capabilities_registered:
            self.capabilities = CapabilityRegistry()
            try:
                self.register_capabilities()
            except Exception:
                self.capabilities = CapabilityRegistry()
                raise
            self._capabilities_registered = True
        return self.capabilities

    def register_capabilities(self) -> None:
        """Register capabilities supported by the loaded model instance."""


class BoundCapability(ModelCapability):
    """Capability adapter that references, but never owns or copies, a model."""

    def __init__(self, model: CapabilityProvider) -> None:
        self._model_ref = weakref.ref(model)

    @property
    def model(self):
        model = self._model_ref()
        if model is None:
            raise RuntimeError("Capability owner has been released.")
        return model
