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
