import os
from typing import Any

import numpy as np
import rembg
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image

import lightx2v.models.networks.sf3d.utils.base as sf3d_utils


def get_device():
    if os.environ.get("SF3D_USE_CPU", "0") == "1":
        return "cpu"

    if os.environ.get("PLATFORM") == "cambricon_mlu":
        if getattr(torch, "mlu", None) is None or not torch.mlu.is_available():
            raise RuntimeError("PLATFORM=cambricon_mlu requires an available torch.mlu device")
        return "mlu"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    intrinsic = sf3d_utils.get_intrinsic_from_fov(
        np.deg2rad(fov_deg),
        H=cond_height,
        W=cond_width,
    )
    intrinsic_normed_cond = intrinsic.clone()
    intrinsic_normed_cond[..., 0, 2] /= cond_width
    intrinsic_normed_cond[..., 1, 2] /= cond_height
    intrinsic_normed_cond[..., 0, 0] /= cond_width
    intrinsic_normed_cond[..., 1, 1] /= cond_height

    return intrinsic, intrinsic_normed_cond


def default_cond_c2w(distance: float):
    c2w_cond = torch.as_tensor(
        [
            [0, 0, 1, distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).float()
    return c2w_cond


def remove_background(
    image: Image,
    rembg_session: Any = None,
    force: bool = False,
    **rembg_kwargs,
) -> Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session, **rembg_kwargs)
    return image


def get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def get_bbox_from_mask(mask, thr=0.5):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))
    return x0, y0, x1, y1


def resize_foreground(
    image: Image.Image | np.ndarray,
    ratio: float,
    out_size=None,
) -> Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image, mode="RGBA")
    assert image.mode == "RGBA"
    # Get bounding box
    mask_np = np.array(image)[:, :, -1]
    x1, y1, x2, y2 = get_bbox_from_mask(mask_np, thr=0.5)
    h, w = y2 - y1, x2 - x1
    yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
    scale = max(h, w) / ratio

    new_image = torchvision_F.crop(
        image,
        top=int(yc - scale / 2),
        left=int(xc - scale / 2),
        height=int(scale),
        width=int(scale),
    )
    if out_size is not None:
        new_image = new_image.resize(out_size)

    return new_image
