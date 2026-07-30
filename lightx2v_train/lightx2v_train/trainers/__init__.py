import importlib

from lightx2v_train.utils.registry import build_trainer

_LAZY_EXPORTS = {
    "ARDmdTrainer": (".dmd.video_ar_trainer", "VideoArDmdTrainer"),
    "DmdTrainer": (".dmd.trainer", "DmdTrainer"),
    "DopsdTrainer": (".dopsd", "DopsdTrainer"),
    "FastWAMTrainer": (".fastwam", "FastWAMTrainer"),
    "FlowMatchingTrainer": (".flow", "FlowMatchingTrainer"),
    "LTX2T2AVArDmdTrainer": (
        ".dmd.ltx_trainer",
        "LTX2T2AVArDmdTrainer",
    ),
    "LTX2T2AVDmdTrainer": (
        ".dmd.ltx_trainer",
        "LTX2T2AVDmdTrainer",
    ),
    "LTX2T2AVFlowTrainer": (".flow", "LTX2T2AVFlowTrainer"),
    "LTX2T2AVTeacherForcingTrainer": (
        ".tf",
        "LTX2T2AVTeacherForcingTrainer",
    ),
    "LingBotVideoDmdTrainer": (
        ".dmd.video_trainer",
        "LingBotVideoDmdTrainer",
    ),
    "TFTrainer": (".tf", "TFTrainer"),
    "VideoArDmdTrainer": (
        ".dmd.video_ar_trainer",
        "VideoArDmdTrainer",
    ),
    "VideoDmdTrainer": (
        ".dmd.video_trainer",
        "VideoDmdTrainer",
    ),
    "VideoPhasedDmdTrainer": (
        ".phased_dmd.trainer",
        "VideoPhasedDmdTrainer",
    ),
    "VideoSgmdTrainer": (".sgmd", "VideoSgmdTrainer"),
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
    "build_trainer",
    "ARDmdTrainer",
    "DmdTrainer",
    "FlowMatchingTrainer",
    "LTX2T2AVArDmdTrainer",
    "LTX2T2AVDmdTrainer",
    "LTX2T2AVFlowTrainer",
    "LTX2T2AVTeacherForcingTrainer",
    "LingBotVideoDmdTrainer",
    "TFTrainer",
    "VideoArDmdTrainer",
    "VideoDmdTrainer",
    "VideoPhasedDmdTrainer",
    "VideoSgmdTrainer",
    "DopsdTrainer",
    "FastWAMTrainer",
]
