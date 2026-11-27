"""Preserve the native mixed precision computation on Ascend only."""
import os
from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def native_precision():
    # HF32 changes FP32 arithmetic even when tensor dtypes stay FP32. Do not
    # change BF16 weights, autocast scopes, or any other platform's settings.
    if os.getenv("PLATFORM") != "ascend_npu":
        yield
        return
    import torch_npu  # noqa: F401

    previous = (torch.npu.conv.allow_hf32, torch.npu.matmul.allow_hf32)
    try:
        torch.npu.conv.allow_hf32 = False
        torch.npu.matmul.allow_hf32 = False
        yield
    finally:
        torch.npu.conv.allow_hf32, torch.npu.matmul.allow_hf32 = previous


def preserve_native_precision(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with native_precision():
            return function(*args, **kwargs)
    return wrapped
