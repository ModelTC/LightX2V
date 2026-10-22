"""RGB preprocessing and independent streaming control encoding for MG4."""

from typing import NamedTuple

import imageio_ffmpeg
import numpy as np
import torch
from PIL import Image, ImageOps

from lightx2v.models.video_encoders.hf.wan.matrix_game4_vae import MatrixGame4TAEStreamingEncoder, MatrixGame4TAEVAE
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import count_conv3d, patchify


class MatrixGame4VideoCondition(NamedTuple):
    control_latents: torch.Tensor
    frame_count: int
    fps: float


def crop_and_resize(image, height, width):
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = round(source_width * scale), round(source_height * scale)
    image = image.resize(resized, resample=Image.Resampling.BILINEAR)
    left, top = round((resized[0] - width) / 2), round((resized[1] - height) / 2)
    return image.crop((left, top, left + width, top + height))


def frames_to_tensor(frames):
    pixels = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
    return torch.from_numpy(pixels).permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0


class MatrixGame4StreamingEncoder:
    """Keep low-poly video features separate from sink and output VAE state."""

    def __init__(self, vae):
        self.model = vae.encoder_model if vae.encoder_model is not None else vae.model
        parameter = next(self.model.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        self.mean = vae.scale[0].to(device=self.device, dtype=self.dtype).view(1, 48, 1, 1, 1)
        self.inv_std = vae.scale[1].to(device=self.device, dtype=self.dtype).view(1, 48, 1, 1, 1)
        self.features = [None] * count_conv3d(self.model.encoder)
        self.first_chunk = True

    @torch.no_grad()
    def encode(self, frames):
        expected = 1 if self.first_chunk else 4
        if frames.shape[0] != expected or frames.shape[1] != 3:
            raise ValueError("Matrix-Game-4 control chunks require one RGB frame initially, then four")
        pixels = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        hidden = self.model.encoder(patchify(pixels, patch_size=2), feat_cache=self.features, feat_idx=[0])
        mean = self.model.conv1(hidden).chunk(2, dim=1)[0]
        latent = (mean - self.mean) * self.inv_std
        self.first_chunk = False
        if latent.shape[1:3] != (48, 1):
            raise RuntimeError("Matrix-Game-4 VAE encoder must produce one 48-channel latent per chunk")
        return latent[0]

    def clear(self):
        self.features.clear()


def create_streaming_encoder(vae):
    if isinstance(vae, MatrixGame4TAEVAE):
        return MatrixGame4TAEStreamingEncoder(vae)
    return MatrixGame4StreamingEncoder(vae)


@torch.no_grad()
def encode_reference_image(vae, image_path, size, sink_mode):
    height, width = size
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        if sink_mode == "first_frame":
            image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        else:
            image = crop_and_resize(image, height, width)
        frames = frames_to_tensor([image])
    parameter = next(vae.model.parameters())
    pixels = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
    return vae.encode(pixels).to(device=parameter.device, dtype=parameter.dtype)


@torch.no_grad()
def encode_control_video(vae, video_path, size, max_frames=None):
    height, width = size
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    encoder = create_streaming_encoder(vae)
    latents = []
    frame_count = 0
    try:
        metadata = next(reader)
        source_width, source_height = metadata["size"]
        fps = float(metadata["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Matrix-Game-4 input video must have a positive finite FPS")
        chunk_frames = 1
        while max_frames is None or frame_count < max_frames:
            frames = []
            for _ in range(chunk_frames):
                if max_frames is not None and frame_count + len(frames) >= max_frames:
                    break
                data = next(reader, None)
                if data is None:
                    break
                image = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(source_height, source_width, 3))
                frames.append(crop_and_resize(image, height, width))
            if not frames:
                break
            valid_frames = len(frames)
            frame_count += valid_frames
            frames.extend([frames[-1]] * (chunk_frames - valid_frames))
            latents.append(encoder.encode(frames_to_tensor(frames)))
            if valid_frames < chunk_frames:
                break
            chunk_frames = 4
        if not latents:
            raise ValueError("Matrix-Game-4 input video contains no frames")
        return MatrixGame4VideoCondition(torch.cat(latents, dim=1), frame_count, fps)
    finally:
        reader.close()
        encoder.clear()
