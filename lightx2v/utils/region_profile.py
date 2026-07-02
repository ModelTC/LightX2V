"""Small ``record_function`` decorator plus an optional active profile object."""

from __future__ import annotations

import contextvars
import functools
import os
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_ANNOTATE_ENV = "LIGHTX2V_REGION_PROFILE_ANNOTATE"
_active_profile: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "lightx2v_region_profile",
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
def _record_region(
    region: str,
    *,
    annotate_env: str | None = None,
    emit: str | None = None,
):
    profile = _active_profile.get()
    if profile is not None and emit is not None:
        hook = getattr(profile, emit, None)
        if callable(hook):
            hook()
    env_key = annotate_env or _ANNOTATE_ENV
    if os.environ.get(env_key) == "1":
        from torch.profiler import record_function

        with record_function(region):
            yield
        return
    yield


def region_profile(
    region: str,
    *,
    annotate_env: str | None = None,
    emit: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Wrap a function in ``record_function`` when ``annotate_env`` is enabled."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            with _record_region(region, annotate_env=annotate_env, emit=emit):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
