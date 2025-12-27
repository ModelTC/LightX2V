from lightx2v_platform.base.nvidia import CudaDevice
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("musa")
class MusaDevice(CudaDevice):
    name = "cuda"

    @staticmethod
    def is_available() -> bool:
        try:
            import torch
            import torchada  # noqa: F401

            return hasattr(torch, "musa") and torch.musa.is_available()
        except ImportError:
            return False
