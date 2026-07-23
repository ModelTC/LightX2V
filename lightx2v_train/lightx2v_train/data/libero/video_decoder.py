import importlib.util
import logging
import warnings
from pathlib import Path

import torch
import torchvision


def _default_backend():
    if importlib.util.find_spec("torchcodec") is not None:
        return "torchcodec"
    return "pyav"


def decode_video_frames(video_path, timestamps, tolerance_s, backend=None):
    backend = backend or _default_backend()
    if backend == "torchcodec":
        try:
            return _decode_torchcodec(video_path, timestamps, tolerance_s)
        except Exception as error:
            warnings.warn(f"torchcodec decode failed ({type(error).__name__}: {error}); falling back to pyav.")
            backend = "pyav"
    if backend not in {"pyav", "video_reader"}:
        raise ValueError(f"Unsupported video backend: {backend}")
    return _decode_torchvision(video_path, timestamps, tolerance_s, backend)


def _decode_torchvision(video_path, timestamps, tolerance_s, backend):
    torchvision.set_video_backend(backend)
    reader = torchvision.io.VideoReader(str(video_path), "video")
    reader.seek(min(timestamps), keyframes_only=backend == "pyav")
    frames = []
    frame_timestamps = []
    for frame in reader:
        frames.append(frame["data"])
        frame_timestamps.append(frame["pts"])
        if frame["pts"] >= max(timestamps):
            break
    if backend == "pyav":
        reader.container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return _select_frames(frames, frame_timestamps, timestamps, tolerance_s, video_path)


def _decode_torchcodec(video_path, timestamps, tolerance_s):
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
    fps = decoder.metadata.average_fps
    batch = decoder.get_frames_at(indices=[round(timestamp * fps) for timestamp in timestamps])
    frames = list(batch.data)
    frame_timestamps = batch.pts_seconds.tolist()
    return _select_frames(frames, frame_timestamps, timestamps, tolerance_s, video_path)


def _select_frames(frames, frame_timestamps, query_timestamps, tolerance_s, video_path):
    query = torch.as_tensor(query_timestamps, dtype=torch.float32)
    available = torch.as_tensor(frame_timestamps, dtype=torch.float32)
    distances = torch.cdist(query[:, None], available[:, None], p=1)
    minimum, indices = distances.min(dim=1)
    if not bool((minimum <= tolerance_s).all()):
        logging.warning("Video timestamp mismatch for %s: max error %.6fs", video_path, minimum.max())
        raise RuntimeError(f"Unable to decode requested timestamps from {Path(video_path)}")
    selected = torch.stack([frames[index] for index in indices.tolist()])
    return selected.to(torch.float32).div_(255.0)
