from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def _process_image(path, target_size):
    with Image.open(path) as source:
        if source.mode == "RGBA":
            image = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
        else:
            image = source.convert("RGB")
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        square = Image.new("RGB", (side, side))
        square.paste(image, (left, top))
        resized = square.resize((target_size, target_size), Image.Resampling.BICUBIC)
        scale = target_size / side
        coords = [left * scale, top * scale, (left + width) * scale, (top + height) * scale, width, height]
        return to_tensor(resized), coords


def load_images(paths, target_size=518, workers=1):
    """Pad to square, resize, and preserve view order in either scheduling mode."""
    if not paths:
        raise ValueError("At least one image is required")
    if workers < 1:
        raise ValueError("preprocess_workers must be positive")
    process = partial(_process_image, target_size=target_size)
    if workers == 1:
        results = list(map(process, paths))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(process, paths))
    images, coords = zip(*results)
    return torch.stack(images), torch.from_numpy(np.asarray(coords)).float()
