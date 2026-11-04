"""Model capabilities shared by Wan component loading and execution."""


def has_image_context(config):
    """Whether this model projects CLIP image tokens into cross-attention."""
    if not config.get("use_image_encoder", True):
        return False
    model = config.get("model_cls")
    if model in {"wan21_edenlive", "wan21_edenlive_ar"}:
        return config.get("task") == "v2v"
    if model in {"wan22_edenlive", "wan22_edenlive_ar"}:
        return False
    return config.get("task") in {"i2v", "flf2v", "animate", "s2v", "rs2v"}
