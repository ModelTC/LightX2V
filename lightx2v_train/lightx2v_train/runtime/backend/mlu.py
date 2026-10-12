import importlib

import torch

from lightx2v_train.runtime.backend.base import BaseBackend
from lightx2v_train.runtime.ops.rope import rope_apply_real


class MluBackend(BaseBackend):
    name = "cambricon_mlu"
    device_type = "mlu"
    process_group_backend = "cncl"
    positional_dtype = torch.float32
    diffusers_attention_backend = "native"
    rope_apply = staticmethod(rope_apply_real)

    @staticmethod
    def _load_extension() -> None:
        try:
            importlib.import_module("torch_mlu")
        except ImportError as exc:
            raise RuntimeError("MLU backend requires torch_mlu to be installed.") from exc

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

    def prepare_rotary_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(frequencies).to(device=self.device, dtype=self.positional_dtype)
