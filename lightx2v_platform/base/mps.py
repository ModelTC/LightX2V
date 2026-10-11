import torch

from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("mps")
class MpsDevice:
    name = "mps"

    @staticmethod
    def init_device_env():
        pass

    @staticmethod
    def is_available() -> bool:
        return torch.backends.mps.is_available()

    @staticmethod
    def get_device() -> str:
        return "mps"

    @staticmethod
    def init_parallel_env():
        raise NotImplementedError("MPS backend only supports single-device execution.")
