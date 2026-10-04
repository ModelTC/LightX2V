#!/usr/bin/env python3
"""Encode downloaded MiniMax-H3 teacher MP4s into normalized H3 AV latents.

This is deliberately a separate stage from API acquisition: downloading can
run on a CPU machine and resume by task id, while the two VAEs are loaded only
on the GPU preprocessing machine.  The output manifest keeps the existing
prompt condition path and adds ``video_latent_path`` / ``audio_latent_path``
fields understood by ``latent_dataset``.

Two success-only manifests are emitted:

* ``metadata.jsonl`` references the condition plus both teacher AV latents;
* ``condition_metadata.jsonl`` references only the same successful condition
  rows, avoiding needless AV-latent I/O when the active loss does not use them.

The cached tensors are the normalized, *unpacked* model latents:

* video: ``[24, 37, H/16, W/16]`` for a 124-frame sample;
* audio: ``[2, 32, 207]`` for the matching 32 kHz stereo soundtrack.

No DMD loss consumes these real/teacher latents yet.  They are produced now so
an opt-in real-data objective can be added without regenerating the videos.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_SAMPLE_RATE = 32_000
AUDIO_HOP_LENGTH = 800
AUDIO_LATENTS_PER_SECOND = 40
VAE_SPATIAL_SCALE = 16
DEFAULT_ALLOWED_RESOLUTIONS = ((768, 1344), (1344, 768))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="JSONL produced by the H3 API downloader.")
    parser.add_argument(
        "--condition-metadata",
        type=Path,
        help=("Optional condition-cache metadata.jsonl. Use this when API videos were acquired before prompt/image caches existed; rows are joined strictly by source_id."),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-field", default="video_path")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--posterior", choices=("mode", "sample"), default="mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be non-negative.")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive.")
    if args.num_frames <= 0 or args.num_frames % 17 != 5:
        parser.error("MiniMax-H3 --num-frames must have the form 17*n+5.")
    if args.fps != 24.0:
        parser.error("MiniMax-H3 training geometry is fixed at --fps 24.")
    return args


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _normalized_identity(value, *, field: str, location: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{location} has no valid {field}.")
    if isinstance(value, (str, int)):
        normalized = str(value).strip()
        if normalized:
            return normalized
    raise ValueError(f"{location} has no valid {field}: {value!r}.")


def _resolved_optional_media(row: dict, field: str, base_dir: Path, location: str) -> Path | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return resolve_path(value, base_dir, field)
    except Exception as error:
        raise ValueError(f"{location} has invalid {field}: {error}") from error


def attach_condition_cache(
    api_rows: list[dict],
    api_metadata: Path,
    condition_metadata: Path,
) -> list[dict]:
    """Attach an existing prompt/image cache to API-success rows.

    API completion order and failure filtering make line-number joins unsafe.
    ``source_id`` is stable across the prepared input, condition cache and API
    manifest, so it is the authoritative key. Caption, task, variant, geometry,
    media and source index are also checked to prevent a plausible-but-wrong
    condition/video pairing.
    """

    condition_rows = read_jsonl(condition_metadata)
    condition_by_source: dict[str, tuple[int, dict]] = {}
    for line_number, row in enumerate(condition_rows, start=1):
        location = f"{condition_metadata}:{line_number}"
        source_id = _normalized_identity(row.get("source_id"), field="source_id", location=location)
        if source_id in condition_by_source:
            raise ValueError(f"Duplicate condition source_id={source_id!r} at {location}.")
        condition_by_source[source_id] = (line_number, row)

    attached = []
    seen: set[str] = set()
    api_base = api_metadata.parent
    condition_base = condition_metadata.parent
    for line_number, api_row in enumerate(api_rows, start=1):
        api_location = f"{api_metadata}:{line_number}"
        source_id = _normalized_identity(api_row.get("source_id"), field="source_id", location=api_location)
        if source_id in seen:
            raise ValueError(f"Duplicate API source_id={source_id!r} at {api_location}.")
        seen.add(source_id)
        match = condition_by_source.get(source_id)
        if match is None:
            raise ValueError(f"No condition cache row matches source_id={source_id!r} at {api_location}.")
        condition_line, condition_row = match
        condition_location = f"{condition_metadata}:{condition_line}"

        for field in ("task", "prompt_variant", "caption"):
            api_value = api_row.get(field)
            condition_value = condition_row.get(field)
            if api_value != condition_value:
                raise ValueError(f"Condition/API {field} mismatch for source_id={source_id!r}: {api_location}={api_value!r}, {condition_location}={condition_value!r}.")
        for field in ("target_height", "target_width", "num_frames"):
            if int(api_row.get(field, -1)) != int(condition_row.get(field, -2)):
                raise ValueError(f"Condition/API {field} mismatch for source_id={source_id!r}: {api_row.get(field)!r} != {condition_row.get(field)!r}.")
        api_index = int(api_row.get("source_metadata_index", -1))
        condition_index = int(condition_row.get("source_index", -2))
        if api_index != condition_index:
            raise ValueError(f"Condition/API source index mismatch for source_id={source_id!r}: {api_index} != {condition_index}.")
        for field in ("first_frame", "last_frame"):
            api_media = _resolved_optional_media(api_row, field, api_base, api_location)
            condition_media = _resolved_optional_media(condition_row, field, condition_base, condition_location)
            if api_media != condition_media:
                raise ValueError(f"Condition/API {field} mismatch for source_id={source_id!r}: {api_media} != {condition_media}.")

        condition_path = resolve_path(condition_row.get("condition_path"), condition_base, "condition_path")
        cache_fingerprint = condition_row.get("cache_fingerprint")
        if not isinstance(cache_fingerprint, str) or not cache_fingerprint:
            raise ValueError(f"Missing cache_fingerprint at {condition_location}.")
        row = dict(api_row)
        row["condition_path"] = str(condition_path)
        row["cache_fingerprint"] = cache_fingerprint
        row["condition_source_index"] = condition_index
        row["condition_metadata_path"] = str(condition_metadata)
        attached.append(row)

    print(
        f"Attached condition cache rows={len(attached)} from {condition_metadata} to API successes in {api_metadata}.",
        flush=True,
    )
    return attached


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def condition_only_row(row: dict) -> dict:
    """Keep the successful sample identity/condition but omit heavy AV inputs."""
    filtered = dict(row)
    for key in (
        "video",
        "video_path",
        "teacher_video_path",
        "audio",
        "audio_path",
        "video_latent_path",
        "audio_latent_path",
        "video_latent_shape",
        "audio_latent_shape",
    ):
        filtered.pop(key, None)
    filtered["teacher_av_latents_available"] = True
    return filtered


def resolve_path(value, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty {field!r}.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} points to a missing file: {path}")
    return path


def row_geometry(row: dict) -> tuple[int, int]:
    try:
        height, width = int(row["target_height"]), int(row["target_width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each row must define integer target_height and target_width.") from error
    if (height, width) not in DEFAULT_ALLOWED_RESOLUTIONS:
        raise ValueError(f"Unsupported teacher target {height}x{width}; expected one of {DEFAULT_ALLOWED_RESOLUTIONS}.")
    return height, width


def _cover_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    if image.size == (width, height):
        return np.asarray(image, dtype=np.uint8)
    scale = max(width / image.width, height / image.height)
    resized_width = max(width, round(image.width * scale))
    resized_height = max(height, round(image.height * scale))
    image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    left = max(0, (resized_width - width) // 2)
    top = max(0, (resized_height - height) // 2)
    return np.asarray(image.crop((left, top, left + width, top + height)), dtype=np.uint8)


def decode_video(path: Path, height: int, width: int, num_frames: int, fps: float) -> torch.Tensor:
    decoded, timestamps = [], []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"MP4 has no video stream: {path}")
        stream = container.streams.video[0]
        fallback_rate = float(stream.average_rate) if stream.average_rate is not None else fps
        for index, frame in enumerate(container.decode(stream)):
            decoded.append(frame.to_ndarray(format="rgb24"))
            timestamp = float(frame.time) if frame.time is not None else index / fallback_rate
            timestamps.append(timestamp)
    if not decoded:
        raise ValueError(f"Could not decode any video frames from {path}.")
    source_height, source_width = decoded[0].shape[:2]
    if (source_width > source_height) != (width > height):
        raise ValueError(f"Video orientation {source_width}x{source_height} does not match target {width}x{height}: {path}.")

    times = np.asarray(timestamps, dtype=np.float64)
    times -= times[0]
    if np.any(np.diff(times) < 0):
        times = np.arange(len(decoded), dtype=np.float64) / fps
    target_times = np.arange(num_frames, dtype=np.float64) / fps
    if times[-1] + (0.5 / fps) < target_times[-1]:
        raise ValueError(f"Video is shorter than the required {num_frames / fps:.3f}s: last decoded timestamp={times[-1]:.3f}s for {path}.")
    right = np.searchsorted(times, target_times, side="left").clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(times[left] - target_times) <= np.abs(times[right] - target_times)
    indices = np.where(choose_left, left, right)
    selected = np.stack([_cover_resize(decoded[int(i)], height, width) for i in indices])
    # [1, 3, frames, height, width], kept uint8 until the GPU transfer.
    return torch.from_numpy(selected.copy()).permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def decode_audio(path: Path, target_samples: int) -> torch.Tensor:
    chunks = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"MP4 has no audio stream: {path}. H3 DMD is joint AV; do not silently substitute zeros.")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_SAMPLE_RATE)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray())
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray())
    if not chunks:
        raise ValueError(f"Could not decode any audio samples from {path}.")
    waveform = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if waveform.shape[0] != 2:
        raise ValueError(f"Audio resampler did not produce stereo [2, samples], got {waveform.shape} for {path}.")
    if waveform.shape[1] < target_samples:
        raise ValueError(f"Audio is shorter than the required {target_samples} samples at {AUDIO_SAMPLE_RATE} Hz: decoded {waveform.shape[1]} samples from {path}.")
    waveform = waveform[:, :target_samples]
    return torch.from_numpy(waveform.copy()).contiguous()


def release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_video_latent(vae, pixels: torch.Tensor, device: torch.device, posterior_mode: str, seed: int):
    pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
    pixels = (pixels - pixel_mean) / pixel_std
    with torch.inference_mode():
        posterior = vae.encode(pixels, return_dict=False)[0]
        if posterior_mode == "sample":
            generator = torch.Generator(device=device).manual_seed(seed)
            latents = posterior.sample(generator=generator)
        else:
            latents = posterior.mode()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    return ((latents.float() - mean) / std)[0].cpu().contiguous()


def encode_audio_latent(audio_vae, waveform: torch.Tensor, device: torch.device):
    with torch.inference_mode():
        posterior = audio_vae.encode(waveform.to(device)[:, None, :], return_dict=False)[0]
        latents = posterior.mode().float()
    mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    return ((latents - mean) / std).cpu().contiguous()


def latent_fingerprint(
    row: dict,
    source_video: Path,
    model_path: Path,
    height: int,
    width: int,
    num_frames: int,
    posterior: str,
) -> str:
    video_stat = source_video.stat()
    descriptor = {
        "source_index": row.get("source_index"),
        "sample_key": row.get("sample_key"),
        "request_fingerprint": row.get("request_fingerprint"),
        "generation_task_id": row.get("generation_task_id"),
        "video_path": str(source_video),
        "video_size": video_stat.st_size,
        "video_mtime_ns": video_stat.st_mtime_ns,
        "model_path": str(model_path),
        "target_height": height,
        "target_width": width,
        "num_frames": num_frames,
        "fps": 24,
        "posterior": posterior,
        "cache_version": 1,
    }
    encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_video_cache(path: Path, expected_shape: tuple[int, ...], expected_fingerprint: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("latents")):
        raise ValueError(f"Invalid video latent cache payload: {path}")
    if tuple(payload["latents"].shape) != expected_shape:
        raise ValueError(f"Video latent cache {path} has shape {tuple(payload['latents'].shape)}, expected {expected_shape}.")
    if not bool(payload.get("normalized", False)):
        raise ValueError(f"Video latent cache is not marked normalized: {path}")
    if payload.get("latent_fingerprint") != expected_fingerprint:
        raise ValueError(f"Video latent cache {path} belongs to another source/request or preprocessing contract; use a new --output-dir or pass --overwrite.")


def validate_audio_cache(path: Path, expected_shape: tuple[int, ...]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(payload) or tuple(payload.shape) != expected_shape:
        shape = tuple(payload.shape) if torch.is_tensor(payload) else type(payload).__name__
        raise ValueError(f"Audio latent cache {path} has {shape}, expected {expected_shape}.")


def main() -> None:
    args = parse_args()
    args.metadata = args.metadata.expanduser().resolve()
    if args.condition_metadata is not None:
        args.condition_metadata = args.condition_metadata.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    if not args.metadata.is_file():
        raise FileNotFoundError(f"Metadata does not exist: {args.metadata}")
    if args.condition_metadata is not None and not args.condition_metadata.is_file():
        raise FileNotFoundError(f"Condition metadata does not exist: {args.condition_metadata}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    all_rows = read_jsonl(args.metadata)
    if args.condition_metadata is not None:
        all_rows = attach_condition_cache(
            all_rows,
            args.metadata,
            args.condition_metadata,
        )
    stop = None if args.max_samples is None else args.start_index + args.max_samples
    selected = all_rows[args.start_index : stop]
    if not selected:
        raise RuntimeError("Selected metadata slice is empty.")

    output_dir = args.output_dir
    video_dir, audio_dir = output_dir / "video_latents", output_dir / "audio_latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = args.metadata.parent
    # H3's chunked causal VAE maps 17*n+5 pixel frames to 5*n+2
    # latent frames; it is not a simple global divide-by-four.
    latent_frames = (args.num_frames - 5) // 17 * 5 + 2
    audio_frames = round((args.num_frames / args.fps) * AUDIO_LATENTS_PER_SECOND)
    target_audio_samples = audio_frames * AUDIO_HOP_LENGTH

    jobs = []
    failures = []
    seen_source_indices = set()
    selected_source_indices = set()
    for local_index, source_row in enumerate(selected, start=args.start_index):
        source_index = local_index
        try:
            source_index = int(source_row.get("source_index", source_row.get("source_metadata_index", local_index)))
            if source_index in selected_source_indices:
                raise ValueError(f"Duplicate stable source_index={source_index} in the selected metadata slice.")
            selected_source_indices.add(source_index)
            height, width = row_geometry(source_row)
            source_video = resolve_path(source_row.get(args.video_field), base_dir, args.video_field)
            # API successes can arrive out of order, so the completed input
            # manifest may grow by inserting an earlier source row. Never use
            # its current line number as the cache identity: doing so could
            # silently associate an old latent with a different prompt after
            # resume. The source cache index is stable across compaction.
            if source_index in seen_source_indices:
                raise ValueError(f"Duplicate stable source_index={source_index} in the selected metadata slice.")
            seen_source_indices.add(source_index)
            stem = f"sample_{source_index:08d}"
            video_path = video_dir / f"{stem}.pt"
            audio_path = audio_dir / f"{stem}.pt"
            expected_video = (VIDEO_CHANNELS, latent_frames, height // VAE_SPATIAL_SCALE, width // VAE_SPATIAL_SCALE)
            expected_audio = (2, AUDIO_CHANNELS, audio_frames)
            fingerprint = latent_fingerprint(
                source_row,
                source_video,
                args.model_path,
                height,
                width,
                args.num_frames,
                args.posterior,
            )
            if video_path.is_file() and not args.overwrite:
                validate_video_cache(video_path, expected_video, fingerprint)
                need_video = False
            else:
                need_video = True
            if audio_path.is_file() and not args.overwrite:
                validate_audio_cache(audio_path, expected_audio)
                need_audio = False
            else:
                need_audio = True
            # A missing/rebuilt video cache means the source contract cannot
            # be established from a shape-only audio tensor. Re-encode audio
            # too, even when a same-shaped partial file exists.
            if need_video:
                need_audio = True
            row = dict(source_row)
            condition = resolve_path(row.get("condition_path"), base_dir, "condition_path")
            row["condition_path"] = str(condition)
            row.update(
                {
                    "source_index": source_index,
                    "video_path": str(source_video),
                    "video_latent_path": str(video_path.relative_to(output_dir)),
                    "audio_latent_path": str(audio_path.relative_to(output_dir)),
                    "video_latent_shape": list(expected_video),
                    "audio_latent_shape": list(expected_audio),
                    "vae_normalized": True,
                    "posterior_mode": args.posterior,
                    "source_kind": "teacher_generated",
                    "latent_cache_version": 1,
                    "latent_fingerprint": fingerprint,
                    "fps": args.fps,
                    "num_frames": args.num_frames,
                    "effective_duration_seconds": args.num_frames / args.fps,
                }
            )
            jobs.append(
                {
                    "index": source_index,
                    "row": row,
                    "source_video": source_video,
                    "height": height,
                    "width": width,
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "expected_video": expected_video,
                    "expected_audio": expected_audio,
                    "fingerprint": fingerprint,
                    "need_video": need_video,
                    "need_audio": need_audio,
                }
            )
        except Exception as error:
            failures.append({"source_index": source_index, "stage": "preflight", "error": str(error)})
            if args.fail_fast:
                raise

    device = torch.device(args.device)
    video_jobs = [job for job in jobs if job["need_video"]]
    if video_jobs:
        from diffusers import AutoencoderKLMiniMaxH3

        print(f"Loading H3 video VAE on {device}; encode={len(video_jobs)} ...", flush=True)
        vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(device).eval()
        for position, job in enumerate(video_jobs, start=1):
            try:
                pixels = decode_video(job["source_video"], job["height"], job["width"], args.num_frames, args.fps)
                latents = encode_video_latent(vae, pixels, device, args.posterior, args.seed + job["index"])
                if tuple(latents.shape) != job["expected_video"]:
                    raise ValueError(f"Encoded video shape {tuple(latents.shape)} != expected {job['expected_video']}.")
                atomic_torch_save(
                    {
                        "latents": latents,
                        "num_frames": latent_frames,
                        "height": job["height"] // VAE_SPATIAL_SCALE,
                        "width": job["width"] // VAE_SPATIAL_SCALE,
                        "pixel_num_frames": args.num_frames,
                        "pixel_height": job["height"],
                        "pixel_width": job["width"],
                        "normalized": True,
                        "posterior_mode": args.posterior,
                        "latent_fingerprint": job["fingerprint"],
                    },
                    job["video_path"],
                )
                job["need_video"] = False
                print(f"[video {position}/{len(video_jobs)}] {job['video_path']}", flush=True)
            except Exception as error:
                failures.append({"source_index": job["index"], "stage": "video_vae", "error": str(error)})
                job["failed"] = True
                if args.fail_fast:
                    raise
        del vae
        release_device_memory()

    audio_jobs = [job for job in jobs if job["need_audio"] and not job.get("failed")]
    if audio_jobs:
        from diffusers import AutoencoderKLMiniMaxH3Audio

        print(f"Loading H3 audio VAE on {device}; encode={len(audio_jobs)} ...", flush=True)
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model_path, subfolder="audio_vae", torch_dtype=torch.float32, local_files_only=True).to(device).eval()
        for position, job in enumerate(audio_jobs, start=1):
            try:
                waveform = decode_audio(job["source_video"], target_audio_samples)
                latents = encode_audio_latent(audio_vae, waveform, device)
                if tuple(latents.shape) != job["expected_audio"]:
                    raise ValueError(f"Encoded audio shape {tuple(latents.shape)} != expected {job['expected_audio']}.")
                atomic_torch_save(latents, job["audio_path"])
                job["need_audio"] = False
                print(f"[audio {position}/{len(audio_jobs)}] {job['audio_path']}", flush=True)
            except Exception as error:
                failures.append({"source_index": job["index"], "stage": "audio_vae", "error": str(error)})
                job["failed"] = True
                if args.fail_fast:
                    raise
        del audio_vae
        release_device_memory()

    completed_rows = []
    for job in jobs:
        if job.get("failed") or job["need_video"] or job["need_audio"]:
            continue
        validate_video_cache(job["video_path"], job["expected_video"], job["fingerprint"])
        validate_audio_cache(job["audio_path"], job["expected_audio"])
        completed_rows.append(job["row"])

    manifest_path = output_dir / "metadata.jsonl"
    existing = read_jsonl(manifest_path) if manifest_path.is_file() else []
    merged = {int(row["source_index"]): row for row in existing}
    # The selected slice is authoritative for this run. Remove its old rows
    # before adding current successes so a newly failed video/audio/condition
    # cache can never survive in the training manifest from an earlier run.
    for source_index in selected_source_indices:
        merged.pop(source_index, None)
    for row in completed_rows:
        merged[int(row["source_index"])] = row
    merged_rows = [merged[index] for index in sorted(merged)]
    atomic_write_jsonl(manifest_path, merged_rows)
    condition_manifest_path = output_dir / "condition_metadata.jsonl"
    atomic_write_jsonl(
        condition_manifest_path,
        [condition_only_row(row) for row in merged_rows],
    )
    if failures:
        failure_path = output_dir / "failed.jsonl"
        old_failures = read_jsonl(failure_path) if failure_path.is_file() else []
        atomic_write_jsonl(failure_path, old_failures + failures)
    print(
        f"Wrote completed={len(completed_rows)} total_manifest={len(merged)} failures={len(failures)} to {manifest_path}; condition-only={condition_manifest_path}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
