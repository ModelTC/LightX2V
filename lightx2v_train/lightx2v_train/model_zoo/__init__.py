import importlib

from lightx2v_train.utils.registry import build_model

_LAZY_EXPORTS = {
    "Flux2DevModel": (".flux2.flux2_dev", "Flux2DevModel"),
    "Flux2KleinModel": (".flux2.flux2_klein", "Flux2KleinModel"),
    "LingBotVideoModel": (".wan.lingbot_video", "LingBotVideoModel"),
    "LongCatImageModel": (".longcat_image.longcat_image", "LongCatImageModel"),
    "LTX2T2AVModel": (".ltx.ltx_t2av", "LTX2T2AVModel"),
    "QwenImageModel": (".qwen_image.qwen_image", "QwenImageModel"),
    "QwenImageEditModel": (".qwen_image.qwen_image_edit", "QwenImageEditModel"),
    "WanFastWAMModel": (".wan.wan_fastwam", "WanFastWAMModel"),
    "WanT2VModel": (".wan.wan_t2v", "WanT2VModel"),
    "WanTI2V5BModel": (".wan.wan_ti2v_5b", "WanTI2V5BModel"),
}


def build_loaded_model(
    config,
    *,
    transformer_only=False,
    reference_model=None,
):
    """Build a model wrapper, load its components, then publish capabilities."""
    model = build_model(config)
    model.load_components(
        transformer_only=transformer_only,
        reference_model=reference_model,
    )
    model.ensure_capabilities()
    return model


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
    "build_loaded_model",
    "QwenImageModel",
    "QwenImageEditModel",
    "LongCatImageModel",
    "Flux2DevModel",
    "Flux2KleinModel",
    "LingBotVideoModel",
    "LTX2T2AVModel",
    "WanT2VModel",
    "WanTI2V5BModel",
    "WanFastWAMModel",
]
