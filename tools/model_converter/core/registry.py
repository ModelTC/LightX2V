"""
Registry system for converters, quantizers, formats, and backends.

Provides a centralized registration mechanism for all conversion components.
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """Generic registry for component registration."""

    def __init__(self, name: str):
        """
        Args:
            name: Registry name for debugging
        """
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(
        self, name: str, obj: Optional[Any] = None
    ) -> Callable[[Type], Type]:
        """
        Register a component.

        Can be used as decorator or direct call:
        
        @registry.register("name")
        class MyClass: ...
        
        Or:
        registry.register("name", MyClass)
        
        Args:
            name: Registration name
            obj: Object to register (if not used as decorator)
        
        Returns:
            Decorator function or registered object
        """

        def _register(obj_to_register: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"{name} already registered in {self._name} registry"
                )
            self._registry[name] = obj_to_register
            return obj_to_register

        if obj is None:
            # Used as decorator
            return _register
        else:
            # Direct registration
            return _register(obj)

    def get(self, name: str) -> Any:
        """
        Get registered component by name.
        
        Args:
            name: Registration name
            
        Returns:
            Registered component
            
        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(
                f"{name} not found in {self._name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry({self._name}, items={list(self._registry.keys())})"


# Global registries
CONVERTER_REGISTRY = Registry("converter")
FORMAT_REGISTRY = Registry("format")
QUANTIZER_REGISTRY = Registry("quantizer")

