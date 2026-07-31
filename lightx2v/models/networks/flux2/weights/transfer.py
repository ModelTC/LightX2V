import torch

from lightx2v.common.ops.utils import move_tensor_to_device
from lightx2v_platform.base.global_var import AI_DEVICE


def _is_transposed_cpu_view(tensor):
    """Return whether ``tensor`` is a 2-D transpose of contiguous CPU storage."""

    return isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dim() == 2 and not tensor.is_contiguous() and tensor.t().is_contiguous()


def _move_transposed_cpu_tensor_to_device(tensor, device, non_blocking):
    """Copy contiguous backing storage first, then restore the transpose view."""

    return tensor.t().to(device, non_blocking=non_blocking).t()


def move_flux2_leaf_to_cuda(module, non_blocking=True):
    """Move one Flux2 leaf weight while avoiding strided CPU-to-NPU copies.

    Flux2 matrix weights are loaded into pinned contiguous CPU storage and then
    exposed as transposed views.  Copying those non-contiguous views directly
    to an NPU is much slower than copying their contiguous backing layout.  The
    device-side transpose below is only a view, so values, shape, and strides
    remain identical without allocating another CPU or NPU tensor.
    """

    base_attrs = getattr(module, "base_attrs", ())
    fast_attrs = {attr_name for _, attr_name, transpose in base_attrs if transpose and _is_transposed_cpu_view(getattr(module, f"pin_{attr_name}", None))}
    if AI_DEVICE != "npu" or not fast_attrs:
        module.to_cuda(non_blocking=non_blocking)
        return False

    for _, attr_name, _ in base_attrs:
        if attr_name in fast_attrs:
            pin_tensor = getattr(module, f"pin_{attr_name}")
            device_tensor = _move_transposed_cpu_tensor_to_device(
                pin_tensor,
                AI_DEVICE,
                non_blocking,
            )
            setattr(module, attr_name, device_tensor)
        else:
            move_tensor_to_device(
                module,
                attr_name,
                AI_DEVICE,
                non_blocking=non_blocking,
            )

    for lora_attr in getattr(module, "lora_attrs", {}):
        value = getattr(module, lora_attr, None)
        if isinstance(value, torch.Tensor):
            setattr(
                module,
                lora_attr,
                value.to(AI_DEVICE, non_blocking=non_blocking),
            )
    return True
