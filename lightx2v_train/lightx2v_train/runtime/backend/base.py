from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
import torch.distributed as dist

from lightx2v_train.runtime.ops.attention import sdpa_attention
from lightx2v_train.runtime.ops.rope import rope_apply_complex


class BaseBackend(ABC):
    """Common device operations shared by all compute backends."""

    positional_dtype = torch.float64
    diffusers_attention_backend = None
    rope_apply = staticmethod(rope_apply_complex)
    attention_compute = staticmethod(sdpa_attention)

    def __init__(self):
        self._device_module = None

    @abstractmethod
    def load_device_module(self):
        """Load the optional device extension and return its torch device module."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return whether this backend can be initialized in the current environment."""

    def initialize(self, local_rank: int | None = None) -> BaseBackend:
        self._device_module = self.load_device_module()
        if self.device_type != "cpu" and (self._device_module is None or not self._device_module.is_available()):
            raise RuntimeError(f"Backend {self.name!r} requested, but torch.{self.device_type} is not available.")

        if local_rank is not None:
            self.set_device(local_rank)
        self.configure_math()
        return self

    @property
    def device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(self.device_type, self._device_module.current_device())

    @property
    def host_device(self) -> torch.device:
        return torch.device("cpu")

    def prepare_rotary_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        return frequencies.to(self.device)

    def set_device(self, local_rank: int) -> None:
        if self.device_type != "cpu":
            self._device_module.set_device(local_rank)

    def empty_cache(self) -> None:
        if self._device_module is not None:
            self._device_module.empty_cache()

    def synchronize(self) -> None:
        if self._device_module is not None:
            self._device_module.synchronize()

    def memory_gb(self):
        if self._device_module is None:
            return None
        return self._device_module.memory_allocated() / 1024**3, self._device_module.memory_reserved() / 1024**3

    def manual_seed_all(self, seed: int) -> None:
        if self._device_module is not None:
            self._device_module.manual_seed_all(seed)

    def ddp_kwargs(self) -> dict:
        if self._device_module is None:
            return {}
        index = self.device.index
        return {
            "device_ids": [index],
            "output_device": index,
        }

    def barrier(self) -> None:
        backend = dist.get_backend().lower()
        if self.device_type != "cpu" and (backend == self.process_group_backend or backend.endswith(f":{self.process_group_backend}")):
            dist.barrier(device_ids=[self.device.index])
        else:
            dist.barrier()

    def configure_math(self) -> None:
        pass

    def resolve_attention_backend(self, requested: str | None = None) -> str | None:
        """Resolve an operator policy without knowing the consuming model."""
        if requested is None or requested.lower() == "auto":
            return self.diffusers_attention_backend
        name = requested.lower()
        return "native" if name in {"sdpa", "torch_sdpa"} else name

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
