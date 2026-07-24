"""Low-overhead ``record_function`` regions for targeted transformer profiles."""

from __future__ import annotations

import contextvars
import functools
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_regions_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lightx2v_transformer_profile_regions",
    default=False,
)
_active_profile: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "lightx2v_transformer_profile",
    default=None,
)


def get_active_profile() -> Any:
    return _active_profile.get()


@contextmanager
def active_profile(profile: Any):
    token = _active_profile.set(profile)
    try:
        yield profile
    finally:
        _active_profile.reset(token)


@contextmanager
def transformer_profile_regions():
    token = _regions_enabled.set(True)
    try:
        yield
    finally:
        _regions_enabled.reset(token)


def region_profile(
    region: str,
    *,
    emit: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Annotate ``fn`` only while a targeted transformer block is profiled."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            profile = _active_profile.get()
            if profile is not None and emit is not None:
                hook = getattr(profile, emit, None)
                if callable(hook):
                    hook()

            if not _regions_enabled.get():
                return fn(*args, **kwargs)

            from torch.profiler import record_function

            with record_function(region):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
