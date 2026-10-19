"""CPU input preparation shared by inference and the single-block estimator."""

import math

from PIL import Image
from loguru import logger

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def image_dimensions(resolution, ratio):
    width = math.sqrt(resolution * resolution * ratio)
    height = width / ratio
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def prepare_images(task, resolution, size=None, aspect_ratio=None, image_path=None):
    if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    images = []
    if task == "i2i":
        if not image_path:
            raise ValueError("Image editing requires --image_path (comma separated for multiple images)")
        for path in image_path.split(","):
            with Image.open(path.strip()) as image:
                image = image.convert("RGBA")
                dimensions = image_dimensions(resolution, image.width / image.height)
                images.append(image.resize(dimensions, Image.Resampling.LANCZOS))
    if size:
        height, width = size
    elif aspect_ratio:
        a, b = (float(s) for s in aspect_ratio.split(":"))
        width, height = image_dimensions(resolution, a / b)
    elif images:
        width, height = images[-1].size
    else:
        height = width = max(32, resolution)
    if height < 32 or width < 32:
        raise ValueError("Output height and width must be at least 32")
    if height % 32 or width % 32:
        logger.warning(f"Output height and width ({height}, {width}) are not divisible by 32 and will be rounded down to multiples of 32")
    return images, [height // 32 * 32, width // 32 * 32]


def condition_token_metadata(processor):
    system_tokens = processor.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}], tokenize=True, return_dict=False)
    return len(system_tokens[0]), processor.tokenizer.encode("<|image_pad|>")[0]


def prepare_condition_inputs(processor, prompt, images=None):
    prefix = ""
    vision_images = []
    for index, image in enumerate(images or []):
        prefix += (" " if index else "") + f"<image{index + 1}><|vision_start|><|image_pad|><|vision_end|>"
        white = Image.new("RGB", image.size, "white")
        white.paste(image, mask=image.getchannel("A"))
        vision_images.append(white)
    text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    kwargs = {"text": [text], "padding": True, "padding_side": "left", "return_tensors": "pt"}
    if vision_images:
        # Images already share the VAE grid; a second resize would change slots.
        kwargs.update(images=vision_images, do_resize=False)
    return processor(**kwargs)
