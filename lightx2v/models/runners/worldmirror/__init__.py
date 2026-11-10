__all__ = ["WorldMirrorRunner"]


def __getattr__(name):
    if name == "WorldMirrorRunner":
        from .worldmirror_runner import WorldMirrorRunner

        globals()[name] = WorldMirrorRunner
        return WorldMirrorRunner
    raise AttributeError(name)
