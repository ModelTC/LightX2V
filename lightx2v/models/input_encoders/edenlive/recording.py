import math
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class RecordingReader:
    """Decode once and prefetch one RGB uint8 chunk while the GPU runs."""

    def __init__(self, video_path, height, width, fps, max_frames, chunk_size=None):
        self.video_path, self.height, self.width = video_path, height, width
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            available = math.floor(float(stream.duration * stream.time_base) * fps)
        if max_frames is not None:
            available = min(available, max_frames)
        self.num_frames = 4 * ((available - 1) // 4) + 1
        if self.num_frames < 5:
            raise ValueError("EdenLive needs at least five decoded frames")
        self.targets = np.arange(0, self.num_frames, 4) / fps
        self.remaining = len(self.targets)
        self.chunk_size = self.remaining if chunk_size is None else chunk_size
        self.frames = self._decode()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edenlive-recording")
        self.pending = self.executor.submit(self._read_chunk)

    def _decode(self):
        with av.open(self.video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 16
            source_width, source_height = stream.width, stream.height
            crop_width = min(source_width, round(source_height * self.width / self.height))
            crop_height = min(source_height, round(source_width * self.height / self.width))
            left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2

            def rgb(frame):
                pixels = frame.to_ndarray(format="rgb24")
                if (source_width, source_height) != (self.width, self.height):
                    image = Image.fromarray(pixels).crop((left, top, left + crop_width, top + crop_height))
                    pixels = np.asarray(image.resize((self.width, self.height), Image.Resampling.BILINEAR))
                return pixels

            decoded = container.decode(video=0)
            previous = next(decoded)
            origin, previous_time, target = float(previous.time), 0.0, 0
            for frame in decoded:
                frame_time = float(frame.time) - origin
                midpoint = (previous_time + frame_time) / 2
                if self.targets[target] <= midpoint:
                    pixels = rgb(previous)
                    # Midpoint ties choose the earlier frame, including VFR input.
                    while target < len(self.targets) and self.targets[target] <= midpoint:
                        target += 1
                        yield pixels
                    if target == len(self.targets):
                        return
                previous, previous_time = frame, frame_time
            pixels = rgb(previous)
            while target < len(self.targets):
                target += 1
                yield pixels

    def _read_chunk(self):
        pixels = np.stack(list(islice(self.frames, self.chunk_size)))
        return torch.from_numpy(pixels).permute(3, 0, 1, 2)

    def next_chunk(self):
        pixels = self.pending.result()
        self.remaining -= pixels.shape[1]
        self.pending = self.executor.submit(self._read_chunk) if self.remaining else None
        return pixels

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.frames.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def prepare_cover(image_path, height, width):
    cover = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1).float()
    scale = max(height / cover.shape[1], width / cover.shape[2])
    cover = F.interpolate(cover[None], size=(math.ceil(cover.shape[1] * scale), math.ceil(cover.shape[2] * scale)), mode="bilinear", align_corners=False)[0]
    top, left = (cover.shape[1] - height) // 2, (cover.shape[2] - width) // 2
    cover = cover[:, top : top + height, left : left + width].contiguous() / 127.5 - 1
    return cover


def prepare_recording(video_path, image_path, height, width, fps, max_frames):
    with RecordingReader(video_path, height, width, fps, max_frames) as reader:
        cover = prepare_cover(image_path, height, width)
        return reader.next_chunk(), cover, reader.num_frames


def frame_prompts(prompt_travel, latent_frames, chunk_size, temporal_stride):
    texts = prompt_travel["prompt_travel_text"]
    starts = prompt_travel["prompt_travel_start_frames"]
    if len(texts) != len(starts):
        raise ValueError("prompt_travel_text and prompt_travel_start_frames must have the same length")
    start_latents = [frame // temporal_stride for frame in starts]
    if starts[0] != 0 or any(left >= right for left, right in zip(start_latents, start_latents[1:])):
        raise ValueError("prompt_travel_start_frames must start at 0 and map to strictly increasing latent frames")
    return [texts[bisect_right(start_latents, start) - 1] for start in range(0, latent_frames, chunk_size)]
