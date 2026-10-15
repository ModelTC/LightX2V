"""Lossless pixel transport and the public reward metric contract."""

import io
import json
import math
import zipfile
from collections.abc import Mapping

import numpy as np
import torch

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 512 * 1024 * 1024
HPS_METRICS = ("hpsv3_standard", "hpsv3_general", "hpsv3_percentile")
VIDEOALIGN_METRICS = ("videoalign_mq", "videoalign_ta", "videoalign_vq")
METRICS = HPS_METRICS + VIDEOALIGN_METRICS
DEFAULT_WEIGHTS = {"hpsv3_general": 1.0, "hpsv3_percentile": 1.0, "videoalign_mq": 1.0}


def video_to_tchw(video, layout="TCHW") -> torch.Tensor:
    """Accept explicit layouts without guessing whether a size of 3 means T or C."""
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(np.ascontiguousarray(video))
    if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8:
        raise ValueError("Reward input must be decoded uint8 pixels, not latents or floating-point pixels")
    if layout not in {"TCHW", "THWC", "CTHW"}:
        raise ValueError("video_layout must be TCHW, THWC, or CTHW")
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError("Reward input must contain one 4D video (optional singleton batch)")
    if layout == "THWC":
        video = video.permute(0, 3, 1, 2)
    elif layout == "CTHW":
        video = video.permute(1, 0, 2, 3)
    if video.shape[1] != 3 or min(video.shape) <= 0:
        raise ValueError("Reward input must have shape [T, 3, H, W] with nonempty dimensions")
    if video.numel() > MAX_REQUEST_BYTES - 65536:
        raise ValueError("Reward video exceeds the 512 MiB transport limit")
    return video.detach().to(device="cpu").contiguous()


def validate_metrics(metrics):
    if not isinstance(metrics, (list, tuple)) or not metrics:
        raise ValueError("metrics must be a nonempty list")
    if any(not isinstance(name, str) or name not in METRICS for name in metrics):
        raise ValueError(f"Supported reward metrics: {', '.join(METRICS)}")
    if len(set(metrics)) != len(metrics):
        raise ValueError("Duplicate reward metrics")
    return tuple(metrics)


def positive_fps(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Video source FPS must be positive and finite")
    return value


def resolve_source_fps(extra_info, fallback=16.0) -> float:
    """Use FewStepRL's metadata precedence, skipping unusable metadata values."""
    if isinstance(extra_info, Mapping):
        for name in ("source_fps", "video_fps", "fps"):
            try:
                return positive_fps(extra_info[name])
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
    return positive_fps(fallback)


def encode_request(video, prompt, metrics, source_fps) -> bytes:
    video = video_to_tchw(video)
    metrics = validate_metrics(list(metrics))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Reward ground_truth must contain the nonempty generation prompt")
    metadata = json.dumps({
        "version": PROTOCOL_VERSION, "prompt": prompt, "metrics": metrics,
        "source_fps": positive_fps(source_fps),
    }, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(metadata) > 65536:
        raise ValueError("Reward metadata exceeds 64 KiB")
    stream = io.BytesIO()
    np.savez_compressed(stream, video=video.numpy(), metadata=np.frombuffer(metadata, dtype=np.uint8))
    payload = stream.getvalue()
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Encoded reward request exceeds 512 MiB")
    return payload


def decode_request(payload):
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Reward request exceeds 512 MiB")
    stream = io.BytesIO(payload)
    with zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {entry.filename for entry in entries} != {"video.npy", "metadata.npy"}:
            raise ValueError("Expected video and metadata arrays")
        if sum(entry.file_size for entry in entries) > MAX_REQUEST_BYTES:
            raise ValueError("Expanded reward request exceeds 512 MiB")
        if archive.getinfo("metadata.npy").file_size > 65536 + 1024:
            raise ValueError("Reward metadata exceeds 64 KiB")
    stream.seek(0)
    with np.load(stream, allow_pickle=False) as arrays:
        metadata_array = arrays["metadata"]
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise ValueError("Reward metadata must be UTF-8 JSON bytes")
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        video = video_to_tchw(arrays["video"])
    if not isinstance(metadata, dict) or metadata.get("version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported reward protocol version")
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Reward prompt must be nonempty text")
    return video, prompt, validate_metrics(metadata.get("metrics")), positive_fps(metadata.get("source_fps"))
