import importlib

from lightx2v_train.utils.registry import build_trainer

_LAZY_EXPORTS = {
    "AutoregressiveDmdTrainer": (
        ".dmd.autoregressive_dmd",
        "AutoregressiveDmdTrainer",
    ),
    "ConsistencyTrainer": (
        ".consistency.trainer",
        "ConsistencyTrainer",
    ),
    "DmdTrainer": (".dmd.trainer", "DmdTrainer"),
    "FlowMatchingTrainer": (
        ".flow_matching",
        "FlowMatchingTrainer",
    ),
    "PhasedDmdTrainer": (
        ".phased_dmd.trainer",
        "PhasedDmdTrainer",
    ),
    "SgmdTrainer": (".sgmd", "SgmdTrainer"),
    "TeacherForcingTrainer": (
        ".teacher_forcing",
        "TeacherForcingTrainer",
    ),
    "TrainingCacheTrainer": (
        ".training_cache",
        "TrainingCacheTrainer",
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


__all__ = ["build_trainer", *_LAZY_EXPORTS]
