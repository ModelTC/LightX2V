from __future__ import annotations

import importlib
import os

import torch
from loguru import logger

from lightx2v_train.runtime.backend.base import BaseBackend

_BACKEND_SPECS = {
    "cpu": ("lightx2v_train.runtime.backend.base", "CpuBackend"),
    "nvidia": ("lightx2v_train.runtime.backend.nvidia", "NvidiaBackend"),
    "ascend_npu": ("lightx2v_train.runtime.backend.ascend", "AscendBackend"),
}
_AUTO_BACKENDS = ("nvidia",)

_BACKEND: BaseBackend | None = None


def _resolve_backend_name() -> str:
    value = os.getenv("PLATFORM")
    if value is None or str(value).lower() == "auto":
        for name in _AUTO_BACKENDS:
            if _load_backend_class(name).is_available():
                return name
        return "cpu"

    name = str(value).lower()
    if name not in _BACKEND_SPECS:
        expected = ", ".join(_BACKEND_SPECS)
        raise RuntimeError(f"Unsupported training platform {name!r}; expected one of: {expected}.")
    return name


def _load_backend_class(name: str):
    module_name, class_name = _BACKEND_SPECS[name]
    backend_class = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(backend_class, BaseBackend):
        raise TypeError(f"{class_name} must inherit from BaseBackend.")
    return backend_class


def _create_backend(name: str) -> BaseBackend:
    return _load_backend_class(name)()


def init_backend() -> BaseBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    name = _resolve_backend_name()
    backend = _create_backend(name)
    local_rank = os.getenv("LOCAL_RANK")
    backend.initialize(None if local_rank is None else int(local_rank))
    _BACKEND = backend
    logger.info(
        "Backend initialized: name={} device={} process_group_backend={}",
        backend.name,
        backend.device,
        backend.process_group_backend,
    )
    return _BACKEND


def get_backend() -> BaseBackend:
    if _BACKEND is None:
        raise RuntimeError("Backend is not initialized; call init_backend() first.")
    return _BACKEND


def get_device() -> torch.device:
    return init_backend().device
