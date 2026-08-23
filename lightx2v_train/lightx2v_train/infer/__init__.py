import importlib

from lightx2v_train.utils.registry import build_inferencer

_LAZY_EXPORTS = {
    "ImageInferencer": (".image", "ImageInferencer"),
    "LingBotVideoT2VInferencer": (
        ".video",
        "LingBotVideoT2VInferencer",
    ),
    "WanT2VDualInferencer": (
        ".video",
        "WanT2VDualInferencer",
    ),
    "WanT2VInferencer": (".video", "WanT2VInferencer"),
    "WanT2VARInferencer": (".video", "WanT2VARInferencer"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "build_inferencer",
    "ImageInferencer",
    "LingBotVideoT2VInferencer",
    "WanT2VDualInferencer",
    "WanT2VInferencer",
    "WanT2VARInferencer",
]
