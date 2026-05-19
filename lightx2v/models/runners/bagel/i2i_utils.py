import os

import numpy as np
import torch
from PIL import Image

from lightx2v.models.networks.bagel.data_utils import pil_img2rgb
from lightx2v.models.runners.bagel.t2i_utils import get_bagel_latent_downsample, get_config_value


_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def _validate_target_shape(target_shape, latent_downsample):
    if len(target_shape) == 2:
        try:
            height, width = int(target_shape[0]), int(target_shape[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BAGEL I2I target_shape must be two positive integers [H W], got: {target_shape}") from exc
        if height <= 0 or width <= 0:
            raise ValueError(f"BAGEL I2I target_shape must be positive [H W], got: {target_shape}")
        if height % latent_downsample != 0 or width % latent_downsample != 0:
            raise ValueError(f"BAGEL I2I target_shape must be divisible by latent downsample {latent_downsample}, got: {[height, width]}")
        return (height, width)

    if target_shape:
        raise ValueError(f"BAGEL I2I target_shape must be [H W] when set, got: {target_shape}")
    return None


def _round_to_multiple(value, multiple):
    return max(multiple, int(round(float(value) / multiple) * multiple))


def resolve_bagel_i2i_image_shape(input_info, config, input_image_size):
    latent_downsample = get_bagel_latent_downsample(config)
    target_shape = getattr(input_info, "target_shape", None) or []
    resolved = _validate_target_shape(target_shape, latent_downsample)
    if resolved is not None:
        return resolved

    if len(input_image_size) != 2:
        raise ValueError(f"BAGEL I2I input_image_size must be [W H], got: {input_image_size}")

    width, height = int(input_image_size[0]), int(input_image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"BAGEL I2I input image size must be positive, got: {(width, height)}")

    max_size = int(get_config_value(config, "i2i_max_image_size", 1024))
    if max_size <= 0:
        raise ValueError(f"BAGEL I2I i2i_max_image_size must be positive, got: {max_size}")

    scale = min(1.0, max_size / max(width, height))
    resolved_height = _round_to_multiple(height * scale, latent_downsample)
    resolved_width = _round_to_multiple(width * scale, latent_downsample)
    return (resolved_height, resolved_width)


def load_bagel_i2i_input_image(image_path):
    if not image_path:
        raise ValueError("BAGEL I2I requires `image_path`.")
    if isinstance(image_path, str) and "," in image_path:
        raise NotImplementedError("BAGEL I2I MVP supports exactly one input image; comma-separated `image_path` values are not supported.")
    if os.path.isdir(image_path):
        raise NotImplementedError("BAGEL I2I MVP supports a single image file; directory `image_path` is not supported.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"BAGEL I2I input image not found: {image_path}")

    with Image.open(image_path) as image:
        return pil_img2rgb(image.copy())


def resize_pil_to_shape(image, image_shape):
    height, width = image_shape
    if image.size == (width, height):
        return image
    return image.resize((width, height), resample=_BICUBIC)


def pil_to_bagel_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.sub_(0.5).div_(0.5)


def resize_pil_for_vit(image, max_size=980, min_size=224, stride=14, max_pixels=14 * 14 * 9 * 1024):
    width, height = image.size
    if height <= 0 or width <= 0:
        raise ValueError(f"BAGEL ViT input image size must be positive, got: {(width, height)}")

    def make_divisible(value):
        return max(stride, int(round(value / stride) * stride))

    def apply_scale(w, h, scale):
        return make_divisible(round(w * scale)), make_divisible(round(h * scale))

    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    new_width, new_height = apply_scale(width, height, scale)

    if new_width * new_height > max_pixels:
        scale = max_pixels / (new_width * new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale)

    if max(new_width, new_height) > max_size:
        scale = max_size / max(new_width, new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale)

    if (new_width, new_height) == image.size:
        return image
    return image.resize((new_width, new_height), resample=_BICUBIC)
