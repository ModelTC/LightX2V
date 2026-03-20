from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock


class SchedulingPolicy(ABC):
    """Base class for instance scheduling policies."""

    def __init__(self):
        self._instances: list[str] = []
        self._lock = Lock()

    def add_instance(self, instance_address: str):
        """Register an instance address if it is not empty and not duplicated."""
        if not instance_address:
            raise ValueError("instance_address cannot be empty")

        with self._lock:
            if instance_address not in self._instances:
                self._instances.append(instance_address)

    def remove_instance(self, instance_address: str):
        """Remove an instance address from scheduler."""
        if not instance_address:
            raise ValueError("instance_address cannot be empty")

        with self._lock:
            self._instances.remove(instance_address)

    @abstractmethod
    def schedule(self) -> str:
        """Return one selected instance address."""
        raise NotImplementedError()
