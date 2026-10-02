from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

_PLATFORM_ALIASES = {
    "ascend": "ascend_npu",
    "npu": "ascend_npu",
}


@dataclass(frozen=True)
class RuntimeBackend:
    """Small device facade shared by entrypoints, models, and parallel strategies."""

    platform: str
    device_type: str
    distributed_backend: str
    device_module: Any | None

    @property
    def device(self) -> torch.device:
        if self.device_module is None:
            return torch.device("cpu")
        return torch.device(self.device_type, self.device_module.current_device())

    def set_device(self, local_rank: int) -> None:
        if self.device_module is not None:
            self.device_module.set_device(local_rank)

    def empty_cache(self) -> None:
        if self.device_module is not None and hasattr(self.device_module, "empty_cache"):
            self.device_module.empty_cache()

    def synchronize(self) -> None:
        if self.device_module is not None and hasattr(self.device_module, "synchronize"):
            self.device_module.synchronize()

    def memory_gb(self):
        if self.device_module is None:
            return None
        memory_allocated = getattr(self.device_module, "memory_allocated", None)
        memory_reserved = getattr(self.device_module, "memory_reserved", None)
        if memory_allocated is None or memory_reserved is None:
            return None
        return memory_allocated() / 1024**3, memory_reserved() / 1024**3

    def manual_seed_all(self, seed: int) -> None:
        if self.device_module is not None and hasattr(self.device_module, "manual_seed_all"):
            self.device_module.manual_seed_all(seed)

    def ddp_kwargs(self) -> dict:
        if self.device_module is None:
            return {}
        index = self.device.index
        return {
            "device_ids": [index],
            "output_device": index,
        }

    def barrier_kwargs(self, backend: str) -> dict:
        backend = str(backend).lower()
        if (backend in {"nccl", "hccl"} or backend.endswith(":hccl")) and self.device_module is not None:
            return {"device_ids": [self.device.index]}
        return {}

    def configure_math(self) -> None:
        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_RUNTIME: RuntimeBackend | None = None


def _resolve_platform(config=None) -> str:
    runtime_config = (config or {}).get("runtime", {})
    configured = runtime_config.get("platform")
    value = configured or os.getenv("PLATFORM")
    if value is None or str(value).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    value = str(value).lower()
    return _PLATFORM_ALIASES.get(value, value)


def _build_runtime(platform: str) -> RuntimeBackend:
    if platform == "ascend_npu":
        importlib.import_module("torch_npu")
        device_type = "npu"
        distributed_backend = "hccl"
    elif platform == "cuda":
        device_type = "cuda"
        distributed_backend = "nccl"
    elif platform == "cpu":
        return RuntimeBackend(
            platform="cpu",
            device_type="cpu",
            distributed_backend="gloo",
            device_module=None,
        )
    else:
        raise RuntimeError(f"Unsupported training platform {platform!r}; expected cuda, ascend_npu, or cpu.")

    device_module = getattr(torch, device_type, None)
    if device_module is None or not device_module.is_available():
        raise RuntimeError(f"Training platform {platform!r} requested, but torch.{device_type} is not available.")
    return RuntimeBackend(
        platform=platform,
        device_type=device_type,
        distributed_backend=distributed_backend,
        device_module=device_module,
    )


def init_runtime(config=None) -> RuntimeBackend:
    global _RUNTIME
    if _RUNTIME is not None and config is None:
        return _RUNTIME
    platform = _resolve_platform(config)
    if _RUNTIME is None:
        _RUNTIME = _build_runtime(platform)
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            _RUNTIME.set_device(int(local_rank))
        _RUNTIME.configure_math()
        logger.info(
            "Runtime initialized: platform={} device={} distributed_backend={}",
            _RUNTIME.platform,
            _RUNTIME.device,
            _RUNTIME.distributed_backend,
        )
    elif _RUNTIME.platform != platform:
        raise RuntimeError(f"Runtime is already initialized for {_RUNTIME.platform!r}, not {platform!r}.")
    return _RUNTIME


def get_runtime() -> RuntimeBackend:
    return _RUNTIME if _RUNTIME is not None else init_runtime()
