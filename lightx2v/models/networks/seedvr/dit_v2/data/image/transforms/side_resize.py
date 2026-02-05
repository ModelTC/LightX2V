

from typing import Union
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


class SideResize:
    def __init__(
        self,
        size: int,
        downsample_only: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        self.size = size
        self.downsample_only = downsample_only
        self.interpolation = interpolation

    def __call__(self, image: Union[torch.Tensor, Image.Image]):
        """
        Args:
            image (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        elif isinstance(image, Image.Image):
            width, height = image.size
        else:
            raise NotImplementedError

        if self.downsample_only and min(width, height) < self.size:
            # keep original height and width for small pictures.
            size = min(width, height)
        else:
            size = self.size

        return TVF.resize(image, size, self.interpolation)
