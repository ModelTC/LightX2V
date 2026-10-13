import importlib

import torch

from lightx2v_train.runtime.backend.base import BaseBackend
from lightx2v_train.runtime.ops.attention import npu_attention
from lightx2v_train.runtime.ops.rope import rope_apply_real


class AscendBackend(BaseBackend):
    name = "ascend_npu"
    device_type = "npu"
    process_group_backend = "hccl"
    positional_dtype = torch.float32
    diffusers_attention_backend = "_native_npu"
    rope_apply = staticmethod(rope_apply_real)
    attention_compute = staticmethod(npu_attention)

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

    def prepare_rotary_frequencies(self, frequencies: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(frequencies).to(device=self.device, dtype=self.positional_dtype)
