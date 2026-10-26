"""RGB preprocessing and independent streaming control encoding for MG4."""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
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


@lru_cache(maxsize=16)
def _video_frame_count(path, size, mtime_ns):
    # Count metadata once per unchanged file, without retaining decoded pixels.
    return imageio_ffmpeg.count_frames_and_secs(path)[0]


class MatrixGame4ControlStream:
    """Prefetch one CPU RGB chunk; encode only the current GPU control latent."""

    def __init__(self, vae, video_path, size, max_frames=None):
        self.reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
        self.encoder = None
        self.executor = None
        self.pending = None
        self.closed = False
        try:
            metadata = next(self.reader)
            self.source_width, self.source_height = metadata["size"]
            self.fps = float(metadata["fps"])
            if not np.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("Matrix-Game-4 input video must have a positive finite FPS")
            path = Path(video_path)
            info = path.stat()
            available = _video_frame_count(str(path.resolve()), info.st_size, info.st_mtime_ns)
            self.frame_count = available if max_frames is None else min(available, max_frames)
            if self.frame_count <= 0:
                raise ValueError("Matrix-Game-4 input video contains no frames")
            self.latent_frames = 1 + (self.frame_count + 2) // 4
            self.size = size
            self.read_frames = 0
            self.encoded_chunks = 0
            self.encoder = create_streaming_encoder(vae)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="matrix-game4-control")
            self.pending = self.executor.submit(self._read_chunk, 1)
        except BaseException:
            self.close()
            raise

    def _read_chunk(self, count):
        frames = []
        needed = min(count, self.frame_count - self.read_frames)
        for _ in range(needed):
            data = next(self.reader, None)
            if data is None:
                raise ValueError("Matrix-Game-4 input ended before its declared frame count")
            image = Image.fromarray(np.frombuffer(data, dtype=np.uint8).reshape(self.source_height, self.source_width, 3))
            frames.append(crop_and_resize(image, *self.size))
        self.read_frames += needed
        frames.extend([frames[-1]] * (count - needed))
        return frames_to_tensor(frames)

    def next_latent(self):
        if self.closed or self.pending is None:
            raise StopIteration("Matrix-Game-4 control stream is exhausted")
        frames = self.pending.result()
        latent = self.encoder.encode(frames)
        self.encoded_chunks += 1
        self.pending = self.executor.submit(self._read_chunk, 4) if self.encoded_chunks < self.latent_frames else None
        return latent

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.reader.close()
        if self.encoder is not None:
            self.encoder.clear()


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
