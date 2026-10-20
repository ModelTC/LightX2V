from loguru import logger

from lightx2v.common.magi_custom_op_mode import set_magi_custom_op_mode, use_magi_custom_ops  # noqa: F401


def get_qwen_image_size(config, size, aspect_ratio=None):
    default_aspect_ratios = {
        "16:9": [1664, 928],
        "9:16": [928, 1664],
        "1:1": [1328, 1328],
        "4:3": [1472, 1104],
        "3:4": [1104, 1472],
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    as_maps = {**default_aspect_ratios, **config.get("aspect_ratios", {})}
    max_size = config.get("max_custom_size", 1664)
    min_size = config.get("min_custom_size", 256)

    if len(size) == 2:
        height, width = size
        height, width = int(height), int(width)
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            width, height = int(width * scale), int(height * scale)
            logger.warning(f"Custom shape is too large, scaled to {width}x{height}")
        width, height = max(width, min_size), max(height, min_size)
        logger.info(f"Qwen Image Runner got custom shape: {width}x{height}")
        return (width, height)

    aspect_ratio = aspect_ratio if aspect_ratio else config.get("aspect_ratio", None)
    if aspect_ratio in as_maps:
        logger.info(f"Qwen Image Runner got aspect ratio: {aspect_ratio}")
        width, height = as_maps[aspect_ratio]
        return (width, height)
    logger.warning(f"Invalid aspect ratio: {aspect_ratio}, not in {as_maps.keys()}")

    return None
