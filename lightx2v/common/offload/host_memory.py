"""CPU weight allocation shared by operators and native module offload."""

import torch
from loguru import logger


def create_pin_tensor(tensor, transpose=False, dtype=None):
    """Copy into pinned CPU memory, falling back to ordinary CPU allocation."""
    dtype = dtype or tensor.dtype
    try:
        pin_tensor = torch.empty(tensor.shape, device="cpu", pin_memory=True, dtype=dtype)
    except Exception as error:
        logger.warning(f"Failed to allocate pinned memory (shape={tensor.shape}, dtype={dtype}): {error}. Falling back to regular CPU memory.")
        pin_tensor = torch.empty(tensor.shape, device="cpu", dtype=dtype)
    pin_tensor.copy_(tensor)
    return pin_tensor.t() if transpose else pin_tensor
