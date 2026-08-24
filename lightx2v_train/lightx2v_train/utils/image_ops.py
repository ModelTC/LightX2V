import math

import numpy as np
import torch
from PIL import Image

_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def pil_to_image_tensor(image):
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected a PIL image, got {type(image)}")
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def image_tensor_to_pil(image):
    if not torch.is_tensor(image):
        raise TypeError(f"Expected an image tensor, got {type(image)}")
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(f"Image preprocessing requires batch_size=1, got {tuple(image.shape)}")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB tensor with shape (3, H, W), got {tuple(image.shape)}")
    if image.dtype != torch.uint8:
        raise TypeError(f"Expected a raw uint8 image tensor, got {image.dtype}")
    array = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array)


def calculate_area_dimensions(target_area, ratio, multiple):
    if target_area <= 0:
        raise ValueError(f"target_area must be positive, got {target_area}")
    if ratio <= 0:
        raise ValueError(f"Image aspect ratio must be positive, got {ratio}")
    width = max(multiple, round(math.sqrt(target_area * ratio) / multiple) * multiple)
    height = max(multiple, round(math.sqrt(target_area / ratio) / multiple) * multiple)
    return width, height


def resize_and_center_crop(image, width, height):
    scale = max(width / image.width, height / image.height)
    scaled_width = max(width, round(image.width * scale))
    scaled_height = max(height, round(image.height * scale))
    image = image.resize((scaled_width, scaled_height), resample=_BICUBIC)
    left = (scaled_width - width) // 2
    top = (scaled_height - height) // 2
    return image.crop((left, top, left + width, top + height))


def align_dimension(value, multiple):
    if value <= 0:
        raise ValueError(f"Image dimensions must be positive, got {value}")
    return max(multiple, value // multiple * multiple)
