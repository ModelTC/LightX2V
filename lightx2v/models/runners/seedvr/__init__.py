__all__ = ["SeedVRRunner"]


def __getattr__(name):
    if name == "SeedVRRunner":
        from .seedvr_runner import SeedVRRunner

        globals()[name] = SeedVRRunner
        return SeedVRRunner
    raise AttributeError(name)
