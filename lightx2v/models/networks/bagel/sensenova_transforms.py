# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """The image resize used by the official SenseNova-Vision inference code."""

    def __init__(
        self,
        max_size,
        min_size,
        stride,
        max_pixels,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    @staticmethod
    def _make_divisible(value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = self._make_divisible(round(width * scale), self.stride)
        new_height = self._make_divisible(round(height * scale), self.stride)
        return new_width, new_height

    def forward(self, image, img_num=1):
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            width, height = image.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Keep this formula byte-for-byte equivalent to the official project.
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(
            image,
            (new_height, new_width),
            self.interpolation,
            antialias=self.antialias,
        )


class ImageTransform:
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        self.stride = image_stride
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            mean=image_mean,
            std=image_std,
            inplace=True,
        )

    def __call__(self, image, img_num=1):
        image = self.resize_transform(image, img_num=img_num)
        image = self.to_tensor_transform(image)
        return self.normalize_transform(image)


def build_sensenova_transforms():
    return {
        "vae": ImageTransform(1024, 512, 16),
        "vit": ImageTransform(980, 224, 14),
        "recon3d_vae": ImageTransform(512, 256, 16),
        "recon3d_vit": ImageTransform(448, 224, 14),
        "camera_vit": ImageTransform(560, 378, 14),
    }
