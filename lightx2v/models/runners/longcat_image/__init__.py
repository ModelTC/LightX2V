__all__ = ["LongCatImageRunner"]


def __getattr__(name):
    if name == "LongCatImageRunner":
        from .longcat_image_runner import LongCatImageRunner

        globals()[name] = LongCatImageRunner
        return LongCatImageRunner
    raise AttributeError(name)
