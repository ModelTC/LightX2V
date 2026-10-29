"""Write camera rays directly to LingBot's packed conditioning tensor."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_camera(camera, directions, output, F: tl.constexpr, H: tl.constexpr, W: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    spatial = H * W
    frame = offset // spatial % F
    y = offset // W % H
    x = offset % W
    packed_channel = offset // (F * spatial)
    component = packed_channel // (SH * SW)
    pixel_y = y * SH + packed_channel // SW % SH
    pixel_x = x * SW + packed_channel % SW
    pixel = pixel_y * (W * SW) + pixel_x
    mask = offset < 6 * SH * SW * F * spatial
    d0 = tl.load(directions + pixel * 3, mask, 0)
    d1 = tl.load(directions + pixel * 3 + 1, mask, 0)
    d2 = tl.load(directions + pixel * 3 + 2, mask, 0)
    row = tl.maximum(component - 3, 0)
    r0 = tl.load(camera + frame * 16 + row * 4, mask, 0)
    r1 = tl.load(camera + frame * 16 + row * 4 + 1, mask, 0)
    r2 = tl.load(camera + frame * 16 + row * 4 + 2, mask, 0)
    ray = tl.fma(r2, d2, tl.fma(r1, d1, r0 * d0))
    origin = tl.load(camera + frame * 16 + tl.minimum(component, 2) * 4 + 3, mask, 0)
    tl.store(output + offset, tl.where(component < 3, origin, ray), mask)


def pack_camera_embedding(camera, directions, latent_height, latent_width, *, height, width, dtype):
    """Preserve FP32 ray math and cast only when storing the final packed output.

    ``directions`` is the shared normalized pixel grid, [height*width, 3].
    The CUDA path avoids the frame-expanded FP32 ray/pixel/packing tensors.
    """
    if camera.ndim != 3 or camera.shape[1:] != (4, 4):
        raise ValueError("camera must have shape [frames, 4, 4]")
    if height % latent_height or width % latent_width:
        raise ValueError("pixel dimensions must divide into complete latent cells")
    if directions.shape != (height * width, 3):
        raise ValueError("directions must contain one three-vector per pixel")
    camera, directions = camera.contiguous(), directions.contiguous()
    frames = camera.shape[0]
    sh, sw = height // latent_height, width // latent_width
    if camera.device.type != "cuda":
        rays = (camera[:, :3, :3] @ directions.T).transpose(1, 2)
        pixels = torch.cat((camera[:, :3, 3].unsqueeze(1).expand_as(rays), rays), dim=-1)
        pixels = pixels.reshape(frames, latent_height, sh, latent_width, sw, 6)
        return pixels.permute(5, 2, 4, 0, 1, 3).contiguous().reshape(-1, frames, latent_height, latent_width).to(dtype)
    output = torch.empty((6 * sh * sw, frames, latent_height, latent_width), dtype=dtype, device=camera.device)
    _pack_camera[(triton.cdiv(output.numel(), 256),)](
        camera,
        directions,
        output,
        frames,
        latent_height,
        latent_width,
        sh,
        sw,
        256,
    )
    return output
