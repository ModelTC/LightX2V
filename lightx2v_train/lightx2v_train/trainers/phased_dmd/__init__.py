import importlib

_LAZY_EXPORTS = {
    "VideoPhasedDmdTrainer": (
        ".trainer",
        "VideoPhasedDmdTrainer",
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
