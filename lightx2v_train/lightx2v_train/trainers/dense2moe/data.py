"""Prompt records retain their native geometry and empirical frequency."""

import hashlib
import json
from pathlib import Path

import torch

from lightx2v_train.model_zoo.native.minimax_h3 import audio_latent_num_frames


class PromptDataset:
    def __init__(self, path, seed):
        with Path(path).open() as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.seed = seed
        self.epoch = -1
        self.order = []

    def sample(self, step, rank, world_size):
        epoch, position = divmod(step * world_size + rank, len(self.records))
        if epoch != self.epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(len(self.records), generator=generator).tolist()
            self.epoch = epoch
        index = self.order[position]
        return index, self.records[index]


class CachedAVDataset(PromptDataset):
    def __init__(self, path, cache_dir, seed):
        super().__init__(path, seed)
        self.cache_dir = Path(cache_dir)
        complete = json.loads((self.cache_dir / "complete.json").read_text())
        if complete["metadata_sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest() or complete["records"] != len(self.records):
            raise ValueError("The completed AV cache does not match the training metadata")

    def load(self, index, device):
        payload = torch.load(self.cache_dir / "samples" / f"{index:06d}.pt", map_location=device, weights_only=True)
        if payload["record_index"] != index:
            raise ValueError(f"Cache record identity mismatch: {index}")
        return payload["condition"], payload["latents"]


def load_media(path, record):
    """Read native RGB frames and aligned 32 kHz stereo, without resizing."""
    import av
    import numpy as np

    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        timestamp_resolution = float(stream.time_base)
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.time))
    expected = (record["num_frames"], record["height"], record["width"], 3)
    pixels = np.stack(frames)
    if pixels.shape != expected or not np.allclose(np.array(times) - times[0], np.arange(len(times)) / 24, atol=timestamp_resolution, rtol=0):
        raise ValueError(f"Video geometry/timeline differs from metadata: {path}, decoded={pixels.shape}, expected={expected}")

    # H3 rounds the audio grid to 40 Hz. The final cell may extend by < 800
    # samples beyond the 24 fps video duration; pad just that rounding tail.
    count = audio_latent_num_frames(record["num_frames"]) * 800
    waveform = np.zeros((2, count), dtype=np.float32)
    covered = np.zeros(count, dtype=bool)
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(resampler.resample(frame))
        chunks.extend(resampler.resample(None))
        for frame in chunks:
            values = frame.to_ndarray()
            start = round((float(frame.time) - times[0]) * 32000)
            left, right = max(0, start), min(count, start + values.shape[1])
            if right > left:
                waveform[:, left:right] = values[:, left - start : right - start]
                covered[left:right] = True
    required = min(count, round(record["num_frames"] / 24 * 32000))
    if not covered[:required].all():
        raise ValueError(f"Audio does not cover the video timeline: {path}")
    video = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return video, torch.from_numpy(waveform)
