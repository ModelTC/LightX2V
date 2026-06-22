import os

import torch

from lightx2v_platform.base.global_var import AI_DEVICE


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def has_no_video_payload(video):
    if video is None:
        return True
    if isinstance(video, torch.Tensor):
        return video.numel() == 0
    return False


def should_skip_rank_postprocess(video, rank, enabled):
    return enabled and rank != 0 and has_no_video_payload(video)


def crop_spatial_to_size(video, target_height=None, target_width=None):
    if not isinstance(video, torch.Tensor):
        return video
    if video.numel() == 0:
        return video
    if target_height is None and target_width is None:
        return video

    height_dim, width_dim = _spatial_dims(video)
    height = video.shape[height_dim]
    width = video.shape[width_dim]
    crop_h = min(height, int(target_height)) if target_height is not None else height
    crop_w = min(width, int(target_width)) if target_width is not None else width
    if crop_h == height and crop_w == width:
        return video

    slices = [slice(None)] * video.ndim
    slices[height_dim] = slice(0, crop_h)
    slices[width_dim] = slice(0, crop_w)
    return video[tuple(slices)].contiguous()


def sync_device_if_available():
    device_module = getattr(torch, AI_DEVICE, None)
    if device_module is None:
        return
    synchronize = getattr(device_module, "synchronize", None)
    if synchronize is not None:
        synchronize()


def _spatial_dims(video):
    if video.ndim != 5:
        return -2, -1
    # VAE tensors are usually B,C,T,H,W before wan_vae_to_comfy and
    # B,T,H,W,C afterwards. In both layouts, H/W are the two dims before
    # channels only for the postprocessed form; rank0 crop is done before it.
    if video.shape[1] in (1, 3, 4, 16, 32):
        return 3, 4
    return 2, 3
