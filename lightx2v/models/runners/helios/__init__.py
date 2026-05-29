from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightx2v.models.runners.helios.helios_runner import HeliosRunner

__all__ = ["HeliosRunner"]


def __getattr__(name):
    if name == "HeliosRunner":
        from lightx2v.models.runners.helios.helios_runner import HeliosRunner

        return HeliosRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
