

from typing import Literal
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from .area_resize import AreaResize
from .side_resize import SideResize


def NaResize(
    resolution: int,
    mode: Literal["area", "side"],
    downsample_only: bool,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):
    if mode == "area":
        return AreaResize(
            max_area=resolution**2,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    if mode == "side":
        return SideResize(
            size=resolution,
            downsample_only=downsample_only,
            interpolation=interpolation,
        )
    if mode == "square":
        return Compose(
            [
                Resize(
                    size=resolution,
                    interpolation=interpolation,
                ),
                CenterCrop(resolution),
            ]
        )
    raise ValueError(f"Unknown resize mode: {mode}")
