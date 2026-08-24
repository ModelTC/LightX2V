from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def math_attention_for_forward_ad(device_type: str):
    """Use the SDPA backend that supports forward-mode automatic differentiation."""
    if device_type != "cuda":
        yield
        return

    backend = torch.backends.cuda
    flash = backend.flash_sdp_enabled()
    memory_efficient = backend.mem_efficient_sdp_enabled()
    cudnn = backend.cudnn_sdp_enabled()
    math = backend.math_sdp_enabled()
    backend.enable_flash_sdp(False)
    backend.enable_mem_efficient_sdp(False)
    backend.enable_cudnn_sdp(False)
    backend.enable_math_sdp(True)
    try:
        yield
    finally:
        backend.enable_flash_sdp(flash)
        backend.enable_mem_efficient_sdp(memory_efficient)
        backend.enable_cudnn_sdp(cudnn)
        backend.enable_math_sdp(math)
