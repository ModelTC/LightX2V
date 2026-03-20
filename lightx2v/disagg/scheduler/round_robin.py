from __future__ import annotations

from lightx2v.disagg.scheduler.base import SchedulingPolicy


class RoundRobinPolicy(SchedulingPolicy):
    """Round-robin scheduler implementation."""

    def __init__(self):
        super().__init__()
        self._next_index = 0

    def schedule(self) -> str:
        with self._lock:
            if not self._instances:
                raise RuntimeError("no available instances to schedule")

            if self._next_index >= len(self._instances):
                self._next_index = 0

            selected = self._instances[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._instances)
            return selected
