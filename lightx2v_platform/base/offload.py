"""Memory operations selected through the existing platform device registry."""

import torch

from lightx2v_platform.base import global_var
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


class TorchBlockOffload:
    validate_checkpoint = False

    @staticmethod
    def prepare():
        pass

    @staticmethod
    def allocate(nbytes, device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device, pin_memory=device.type == "cpu")

    @staticmethod
    def copy(destination, source, stream):
        destination.copy_(source, non_blocking=True)
        destination.record_stream(stream)
        if source.device.type != "cpu":
            source.record_stream(stream)

    @staticmethod
    def device_module(device):
        return getattr(torch, device.type)


# Key by platform identity, not the torch device type shared by other vendors.
_DEFAULT_BLOCK_OFFLOAD_BACKENDS = {"cuda": TorchBlockOffload}


def get_block_offload_backend(required=True):
    platform_name = global_var.PLATFORM
    platform = PLATFORM_DEVICE_REGISTER[platform_name]
    backend = getattr(platform, "block_offload_backend", None)
    if backend is None:
        backend = _DEFAULT_BLOCK_OFFLOAD_BACKENDS.get(platform_name)
    if backend is None and required:
        raise ValueError(f"Platform {platform_name!r} has not declared contiguous block offload support")
    return backend
