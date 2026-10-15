import math
from bisect import bisect_right

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def prepare_recording(video_path, image_path, height, width, fps, max_frames):
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        source_width, source_height = stream.width, stream.height
        duration = float(stream.duration * stream.time_base)
        timestamps = np.array([float(frame.time) for frame in container.decode(video=0)])
    timestamps -= timestamps[0]
    available = math.floor(duration * fps)
    if max_frames is not None:
        available = min(available, max_frames)
    num_frames = 4 * ((available - 1) // 4) + 1
    if num_frames < 5:
        raise ValueError("EdenLive needs at least five decoded frames")
    indices = np.searchsorted((timestamps[:-1] + timestamps[1:]) / 2, np.arange(num_frames) / fps)
    crop_width = min(source_width, round(source_height * width / height))
    crop_height = min(source_height, round(source_width * height / width))
    left, top = (source_width - crop_width) // 2, (source_height - crop_height) // 2
    proxy = torch.empty(3, (num_frames - 1) // 4 + 1, height, width)
    target = 0
    with av.open(video_path) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if target == num_frames:
                break
            if index != indices[target]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if (source_width, source_height) != (width, height):
                image = Image.fromarray(rgb).crop((left, top, left + crop_width, top + crop_height))
                rgb = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
            while target < num_frames and indices[target] == index:
                if target % 4 == 0:
                    proxy[:, target // 4] = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 127.5 - 1
                target += 1
    if target != num_frames:
        raise ValueError(f"Decoded {target} frames, expected {num_frames}")
    cover = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1).float()
    scale = max(height / cover.shape[1], width / cover.shape[2])
    cover = F.interpolate(cover[None], size=(math.ceil(cover.shape[1] * scale), math.ceil(cover.shape[2] * scale)), mode="bilinear", align_corners=False)[0]
    top, left = (cover.shape[1] - height) // 2, (cover.shape[2] - width) // 2
    cover = cover[:, top : top + height, left : left + width].contiguous() / 127.5 - 1
    return proxy, cover, num_frames


def frame_prompts(prompt_travel, latent_frames, chunk_size, temporal_stride):
    texts = prompt_travel["prompt_travel_text"]
    starts = prompt_travel["prompt_travel_start_frames"]
    if len(texts) != len(starts):
        raise ValueError("prompt_travel_text and prompt_travel_start_frames must have the same length")
    start_latents = [frame // temporal_stride for frame in starts]
    if starts[0] != 0 or any(left >= right for left, right in zip(start_latents, start_latents[1:])):
        raise ValueError("prompt_travel_start_frames must start at 0 and map to strictly increasing latent frames")
    return [texts[bisect_right(start_latents, start) - 1] for start in range(0, latent_frames, chunk_size)]
