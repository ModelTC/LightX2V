

from typing import Optional

_NORM_LIMIT = float("inf")


def get_norm_limit():
    return _NORM_LIMIT


def set_norm_limit(value: Optional[float] = None):
    global _NORM_LIMIT
    if value is None:
        value = float("inf")
    _NORM_LIMIT = value
