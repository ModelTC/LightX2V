"""Image helpers copied from OpenPI's torch preprocessing path (Apache-2.0)."""

import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    """Resize without distortion and pad with black / -1, preserving layout."""
    input_was_unbatched = images.dim() == 3
    channels_last = images.shape[-1] <= 4
    if input_was_unbatched:
        images = images.unsqueeze(0)
    if channels_last:
        images = images.permute(0, 3, 1, 2)

    _, _, current_height, current_width = images.shape
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
        pad_value = 0
    elif images.dtype == torch.float32:
        resized = resized.clamp(-1.0, 1.0)
        pad_value = -1.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    resized = F.pad(
        resized,
        (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h),
        mode="constant",
        value=pad_value,
    )
    if channels_last:
        resized = resized.permute(0, 2, 3, 1)
    if input_was_unbatched:
        resized = resized.squeeze(0)
    return resized
