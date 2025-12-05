from lightx2v_platform.base.nvidia import CudaDevice
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


@PLATFORM_DEVICE_REGISTER("metax")
class MetaxDevice(CudaDevice):
    name = "cuda"
