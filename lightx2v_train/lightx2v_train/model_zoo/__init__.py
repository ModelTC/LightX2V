import importlib

from lightx2v_train.utils.registry import build_model

_LAZY_EXPORTS = {
    "Flux2DevModel": (".flux2_dev", "Flux2DevModel"),
    "Flux2KleinModel": (".flux2_klein", "Flux2KleinModel"),
    "LingBotVideoModel": (".lingbot_video", "LingBotVideoModel"),
    "LongCatImageModel": (".longcat_image", "LongCatImageModel"),
    "LTX2T2AVModel": (".ltx_t2av", "LTX2T2AVModel"),
    "MiniMaxH3T2AVModel": (".minimax_h3_t2av", "MiniMaxH3T2AVModel"),
    "QwenImageModel": (".qwen_image", "QwenImageModel"),
    "QwenImageEditModel": (".qwen_image_edit", "QwenImageEditModel"),
    "WanFastWAMModel": (".wan_fastwam", "WanFastWAMModel"),
    "WanT2VModel": (".wan_t2v", "WanT2VModel"),
    "WanTI2V5BModel": (".wan_ti2v_5b", "WanTI2V5BModel"),
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
    "build_model",
    "QwenImageModel",
    "QwenImageEditModel",
    "LongCatImageModel",
    "Flux2DevModel",
    "Flux2KleinModel",
    "LingBotVideoModel",
    "LTX2T2AVModel",
    "MiniMaxH3T2AVModel",
    "WanT2VModel",
    "WanTI2V5BModel",
    "WanFastWAMModel",
]
