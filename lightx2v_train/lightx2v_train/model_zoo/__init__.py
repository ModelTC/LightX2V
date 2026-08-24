import importlib

from lightx2v_train.utils.registry import build_model

_LAZY_EXPORTS = {
    "Flux2DevModel": (".flux2.flux2_dev", "Flux2DevModel"),
    "Flux2DevEditModel": (".flux2.flux2_dev_edit", "Flux2DevEditModel"),
    "Flux2KleinModel": (".flux2.flux2_klein", "Flux2KleinModel"),
    "Flux2KleinEditModel": (".flux2.flux2_klein_edit", "Flux2KleinEditModel"),
    "LingBotVideoModel": (".wan.lingbot_video", "LingBotVideoModel"),
    "LongCatImageModel": (".longcat_image.longcat_image", "LongCatImageModel"),
    "LongCatImageEditModel": (".longcat_image.longcat_image_edit", "LongCatImageEditModel"),
    "MiniMaxH3T2AVModel": (".minimax_h3.minimax_h3_t2av", "MiniMaxH3T2AVModel"),
    "QwenImageModel": (".qwen_image.qwen_image", "QwenImageModel"),
    "QwenImageEditModel": (".qwen_image.qwen_image_edit", "QwenImageEditModel"),
    "WanT2VModel": (".wan.wan_t2v", "WanT2VModel"),
}


def build_loaded_model(
    config,
    *,
    load_transformer,
    load_vae,
    load_condition_encoder,
):
    """Build a model wrapper, load its components, then publish capabilities."""
    model = build_model(config)
    model.load_components(
        load_transformer=load_transformer,
        load_vae=load_vae,
        load_condition_encoder=load_condition_encoder,
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
    "LongCatImageEditModel",
    "Flux2DevModel",
    "Flux2DevEditModel",
    "Flux2KleinModel",
    "Flux2KleinEditModel",
    "LingBotVideoModel",
    "MiniMaxH3T2AVModel",
    "WanT2VModel",
]
