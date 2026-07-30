import importlib

_LAZY_EXPORTS = {
    "DmdTrainer": (".trainer", "DmdTrainer"),
    "VideoDmdTrainer": (".video_trainer", "VideoDmdTrainer"),
    "LingBotVideoDmdTrainer": (
        ".video_trainer",
        "LingBotVideoDmdTrainer",
    ),
    "VideoArDmdTrainer": (
        ".video_ar_trainer",
        "VideoArDmdTrainer",
    ),
    "ARDmdTrainer": (".video_ar_trainer", "VideoArDmdTrainer"),
    "LTX2T2AVDmdTrainer": (
        ".ltx_trainer",
        "LTX2T2AVDmdTrainer",
    ),
    "LTX2T2AVArDmdTrainer": (
        ".ltx_trainer",
        "LTX2T2AVArDmdTrainer",
    ),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(
        importlib.import_module(module_name, __name__),
        attribute_name,
    )
    globals()[name] = value
    return value


__all__ = list(_LAZY_EXPORTS)
