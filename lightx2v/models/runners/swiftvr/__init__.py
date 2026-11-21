__all__ = ["SwiftVRRunner"]


def __getattr__(name):
    if name == "SwiftVRRunner":
        from .swiftvr_runner import SwiftVRRunner

        globals()[name] = SwiftVRRunner
        return SwiftVRRunner
    raise AttributeError(name)
