import importlib

import torch

from lightx2v_train.runtime.backend.base import BaseBackend


class MusaBackend(BaseBackend):
    name = "musa"
    device_type = "musa"
    process_group_backend = "mccl"
    diffusers_attention_backend = "native"

    @staticmethod
    def _load_extension() -> None:
        try:
            importlib.import_module("torch_musa")
        except ImportError as exc:
            raise RuntimeError("MUSA backend requires torch_musa to be installed.") from exc

    @classmethod
    def is_available(cls) -> bool:
        try:
            cls._load_extension()
        except RuntimeError:
            return False
        device_module = getattr(torch, cls.device_type, None)
        return device_module is not None and device_module.is_available()

    def load_device_module(self):
        self._load_extension()
        return getattr(torch, self.device_type, None)
