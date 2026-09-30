"""Memory operations selected through the existing platform device registry."""

from importlib import import_module

from lightx2v_platform.base import global_var
from lightx2v_platform.ops.offload.template import TorchBlockOffload
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER

# Key by platform identity, not the torch device type shared by other vendors.
_DEFAULT_BLOCK_OFFLOAD_BACKENDS = {"cuda": TorchBlockOffload}


def get_block_offload_backend(required=True):
    platform_name = global_var.PLATFORM
    platform = PLATFORM_DEVICE_REGISTER[platform_name]
    backend = getattr(platform, "block_offload_backend", None)
    if backend is None:
        backend = _DEFAULT_BLOCK_OFFLOAD_BACKENDS.get(platform_name)
    if isinstance(backend, str):
        module_name, class_name = backend.rsplit(".", 1)
        backend = getattr(import_module(module_name), class_name)
    if backend is None and required:
        raise ValueError(f"Platform {platform_name!r} has not declared contiguous block offload support")
    return backend


def copy_to_cpu(destination, source, non_blocking=False):
    """Preserve the host destination, using a platform override when needed."""
    platform = PLATFORM_DEVICE_REGISTER[global_var.PLATFORM]
    copy = getattr(platform, "copy_to_cpu", None)
    if copy is not None:
        return copy(destination, source, non_blocking=non_blocking)
    return destination.copy_(source, non_blocking=non_blocking)


def get_transposed_weight_copy():
    """Optional H2D override for a transposed view of a contiguous CPU matrix."""
    platform = PLATFORM_DEVICE_REGISTER[global_var.PLATFORM]
    return getattr(platform, "copy_transposed_weight_to_device", None)
