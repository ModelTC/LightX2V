from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, ClassVar

import torch


class BaseBackend(ABC):
    """Common device operations shared by all compute backends."""

    name: ClassVar[str]
    device_type: ClassVar[str]
    process_group_backend: ClassVar[str]

    def __init__(self):
        self._device_module: Any | None = None
        self._initialized = False

    @abstractmethod
    def load_device_module(self) -> Any | None:
        """Load the optional device extension and return its torch device module."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return whether this backend can be initialized in the current environment."""

    def initialize(self, local_rank: int | None = None) -> BaseBackend:
        if self._initialized:
            return self

        self._device_module = self.load_device_module()
        if self.device_type != "cpu":
            if self._device_module is None or not self._device_module.is_available():
                raise RuntimeError(f"Backend {self.name!r} requested, but torch.{self.device_type} is not available.")

        if local_rank is not None:
            self.set_device(local_rank)
        self.configure_math()
        self._initialized = True
        return self

    @property
    def device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        device_module = self._require_device_module()
        return torch.device(self.device_type, device_module.current_device())

    @property
    def host_device(self) -> torch.device:
        return torch.device("cpu")

    def _require_device_module(self):
        if self._device_module is None:
            raise RuntimeError(f"Backend {self.name!r} has not been initialized.")
        return self._device_module

    def _device_id_kwargs(self) -> dict:
        return {"device_ids": [self.device.index]}

    def set_device(self, local_rank: int) -> None:
        if self.device_type != "cpu":
            self._require_device_module().set_device(local_rank)

    def empty_cache(self) -> None:
        if self._device_module is not None and hasattr(self._device_module, "empty_cache"):
            self._device_module.empty_cache()

    def synchronize(self) -> None:
        if self._device_module is not None and hasattr(self._device_module, "synchronize"):
            self._device_module.synchronize()

    def memory_gb(self):
        if self._device_module is None:
            return None
        memory_allocated = getattr(self._device_module, "memory_allocated", None)
        memory_reserved = getattr(self._device_module, "memory_reserved", None)
        if memory_allocated is None or memory_reserved is None:
            return None
        return memory_allocated() / 1024**3, memory_reserved() / 1024**3

    def manual_seed_all(self, seed: int) -> None:
        if self._device_module is not None and hasattr(self._device_module, "manual_seed_all"):
            self._device_module.manual_seed_all(seed)

    def ddp_kwargs(self) -> dict:
        if self._device_module is None:
            return {}
        index = self.device.index
        return {
            "device_ids": [index],
            "output_device": index,
        }

    def barrier_kwargs(self, process_group_backend: str) -> dict:
        return {}

    def configure_math(self) -> None:
        pass

    def resolve_attention_backend(self, requested: str | None = None) -> str | None:
        """Resolve an operator policy without knowing the consuming model."""
        if requested is None or str(requested).lower() == "auto":
            return None
        aliases = {"sdpa": "native", "torch_sdpa": "native", "npu": "_native_npu"}
        name = str(requested).lower()
        return aliases.get(name, name)

    def autocast(self, dtype: torch.dtype | None = None, enabled: bool = True):
        if not enabled or dtype not in (None, torch.float16, torch.bfloat16):
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=dtype, enabled=enabled)


class CpuBackend(BaseBackend):
    """CPU fallback kept with the common backend implementation."""

    name = "cpu"
    device_type = "cpu"
    process_group_backend = "gloo"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load_device_module(self):
        return None
