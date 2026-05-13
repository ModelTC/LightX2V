# Minimal utilities extracted from lyra_2._ext.imaginaire.utils.misc
# Only the functions needed for the Lyra-2 ZoomGS inference path are kept.

import collections.abc
import functools
import random
import time
from contextlib import ContextDecorator
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist

try:
    import termcolor

    _has_termcolor = True
except ImportError:
    _has_termcolor = False

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Color
# ---------------------------------------------------------------------------


class Color:
    """Colorize strings for console output."""

    @staticmethod
    def _c(x: str, color: str) -> str:
        if _has_termcolor:
            return termcolor.colored(str(x), color=color)
        return str(x)

    @staticmethod
    def red(x: str) -> str:
        return Color._c(x, "red")

    @staticmethod
    def green(x: str) -> str:
        return Color._c(x, "green")

    @staticmethod
    def blue(x: str) -> str:
        return Color._c(x, "blue")

    @staticmethod
    def cyan(x: str) -> str:
        return Color._c(x, "cyan")

    @staticmethod
    def yellow(x: str) -> str:
        return Color._c(x, "yellow")

    @staticmethod
    def magenta(x: str) -> str:
        return Color._c(x, "magenta")

    @staticmethod
    def grey(x: str) -> str:
        return Color._c(x, "grey")


# ---------------------------------------------------------------------------
# to — recursive tensor cast
# ---------------------------------------------------------------------------


def to(
    data: Any,
    device: "str | torch.device | None" = None,
    dtype: "torch.dtype | None" = None,
    memory_format: torch.memory_format = torch.preserve_format,
) -> Any:
    """Recursively cast tensors / lists / dicts to device/dtype."""
    assert device is not None or dtype is not None or memory_format is not None

    if isinstance(data, torch.Tensor):
        if memory_format == torch.channels_last and data.dim() != 4 or memory_format == torch.channels_last_3d and data.dim() != 5:
            memory_format = torch.preserve_format
        is_cpu = (isinstance(device, str) and device == "cpu") or (isinstance(device, torch.device) and device.type == "cpu")
        return data.to(
            device=device,
            dtype=dtype,
            memory_format=memory_format,
            non_blocking=(not is_cpu),
        )
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({k: to(data[k], device=device, dtype=dtype, memory_format=memory_format) for k in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        return type(data)([to(elem, device=device, dtype=dtype, memory_format=memory_format) for elem in data])
    else:
        return data


# ---------------------------------------------------------------------------
# set_random_seed
# ---------------------------------------------------------------------------


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def set_random_seed(seed: int, by_rank: bool = False) -> None:
    """Set random seed for random / numpy / torch."""
    if by_rank:
        seed += _get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# timer — context manager / decorator for timing
# ---------------------------------------------------------------------------


class timer(ContextDecorator):  # noqa: N801
    """Simple timer that logs elapsed time on exit."""

    def __init__(self, context: str, debug: bool = False) -> None:
        self.context = context
        self.debug = debug

    def __enter__(self) -> None:
        self.tic = time.time()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        from loguru import logger

        elapsed = time.time() - self.tic
        msg = f"Time spent on {self.context}: {elapsed:.4f} seconds"
        if self.debug:
            logger.debug(msg)
        else:
            logger.info(msg)

    def __call__(self, func: T) -> T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            from loguru import logger

            tic = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - tic
            msg = f"Time spent on {self.context}: {elapsed:.4f} seconds"
            if self.debug:
                logger.debug(msg)
            else:
                logger.info(msg)
            return result

        return wrapper  # type: ignore
