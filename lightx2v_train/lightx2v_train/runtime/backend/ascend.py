import importlib

import torch

from lightx2v_train.runtime.backend.base import BaseBackend


class AscendBackend(BaseBackend):
    name = "ascend_npu"
    device_type = "npu"
    process_group_backend = "hccl"

    @staticmethod
    def _load_extension() -> None:
        try:
            importlib.import_module("torch_npu")
        except ImportError as exc:
            raise RuntimeError("Ascend backend requires torch_npu to be installed.") from exc

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

    def barrier_kwargs(self, process_group_backend: str) -> dict:
        backend = str(process_group_backend).lower()
        if backend == self.process_group_backend or backend.endswith(f":{self.process_group_backend}"):
            return self._device_id_kwargs()
        return {}
