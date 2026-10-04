import torch

from lightx2v_train.runtime.backend.base import BaseBackend


class NvidiaBackend(BaseBackend):
    name = "nvidia"
    device_type = "cuda"
    process_group_backend = "nccl"

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    def load_device_module(self):
        return torch.cuda

    def barrier_kwargs(self, process_group_backend: str) -> dict:
        backend = str(process_group_backend).lower()
        if backend == self.process_group_backend or backend.endswith(f":{self.process_group_backend}"):
            return self._device_id_kwargs()
        return {}

    def configure_math(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
