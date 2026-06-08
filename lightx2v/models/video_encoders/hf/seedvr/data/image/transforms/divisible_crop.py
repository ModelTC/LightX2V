from typing import Union

import torch
from PIL import Image
from torchvision.transforms import functional as TVF


class DivisibleCrop:
    def __init__(self, factor):
        if not isinstance(factor, tuple):
            factor = (factor, factor)

        self.height_factor, self.width_factor = factor[0], factor[1]

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        pad_height = (-height) % self.height_factor
        pad_width = (-width) % self.width_factor
        if pad_height == 0 and pad_width == 0:
            return image

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        image = TVF.pad(
            img=image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            padding_mode="edge",
        )
        return image
