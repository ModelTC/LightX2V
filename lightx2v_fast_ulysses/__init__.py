# Copied and adapted from https://github.com/triple-mu/fast-ulysses
"""LightX2V-vendored fast-ulysses custom ops over the NVSHMEM symmetric heap."""

import torch  # noqa: F401  load libtorch before dlopen of _C

try:
    from . import _C  # noqa: F401,E402  trigger TORCH_LIBRARY registration
except ImportError as exc:
    raise ImportError(
        "lightx2v_fast_ulysses native extension is not built. "
        "Build it with: NVSHMEM_HOME=/path/to/nvshmem pip install ./lightx2v_fast_ulysses"
    ) from exc

from .comm import AsyncA2AHandle, UlyssesGroup  # noqa: E402

__all__ = ["UlyssesGroup", "AsyncA2AHandle", "_C"]
