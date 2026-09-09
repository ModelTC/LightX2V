"""Durable, lightweight artifacts for local OpenPI LIBERO evaluation."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

EpisodeKey = tuple[str, int, int]


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` only after the complete payload reaches disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(Path(path), json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npy")
    try:
        np.save(temporary, np.asarray(array))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write a LIBERO video without policy-phase frames")
    if fps < 1:
        raise ValueError(f"Video fps must be positive, got {fps}")

    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        imageio.mimwrite(temporary, [np.asarray(frame) for frame in frames], fps=fps)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def episode_directory(output_dir: Path, benchmark: str, task_id: int, init_state_id: int) -> Path:
    return Path(output_dir) / "episodes" / benchmark / f"task_{task_id:02d}" / f"init_{init_state_id:02d}"


def episode_metrics_path(output_dir: Path, key: EpisodeKey) -> Path:
    benchmark, task_id, init_state_id = key
    return episode_directory(output_dir, benchmark, task_id, init_state_id) / "metrics.json"


def record_key(record: Mapping[str, Any]) -> EpisodeKey:
    try:
        return str(record["benchmark"]), int(record["task_id"]), int(record["init_state_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid LIBERO episode record identity: {record!r}") from exc


def read_episode_records(
    output_dir: Path,
    expected_keys: Iterable[EpisodeKey],
    protocol_id: str,
) -> dict[EpisodeKey, dict[str, Any]]:
    """Read committed episode records; temporary files are deliberately ignored."""
    expected = tuple(expected_keys)
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise ValueError("Expected LIBERO episode keys contain duplicates")

    records: dict[EpisodeKey, dict[str, Any]] = {}
    for key in expected:
        path = episode_metrics_path(output_dir, key)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read committed LIBERO metrics: {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"LIBERO metrics must contain a JSON object: {path}")
        if record_key(payload) != key:
            raise RuntimeError(f"LIBERO metrics identity does not match its path: {path}")
        if payload.get("protocol_id") != protocol_id:
            raise RuntimeError(f"Existing episode uses a different evaluation protocol: {path}")
        records[key] = payload

    episodes_root = Path(output_dir) / "episodes"
    if episodes_root.is_dir():
        for path in episodes_root.rglob("metrics.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                key = record_key(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"Invalid committed LIBERO metrics outside the requested table: {path}") from exc
            if key not in expected_set:
                raise RuntimeError(f"Evaluation output contains an episode outside the requested selection: {path}")

    return records


def save_evaluation_episode(
    output_dir: Path,
    record: Mapping[str, Any],
    frames: list[np.ndarray],
    actions: np.ndarray,
    *,
    video_policy: str,
    video_fps: int,
    save_actions: bool,
) -> dict[str, Any]:
    """Save optional payloads first and atomically commit ``metrics.json`` last."""
    saved = dict(record)
    key = record_key(saved)
    episode_dir = episode_directory(output_dir, *key)
    episode_dir.mkdir(parents=True, exist_ok=True)
    artifact_errors: list[str] = []

    action_path: Path | None = None
    if save_actions:
        action_path = episode_dir / "actions.npy"
        try:
            atomic_save_numpy(action_path, actions)
        except Exception as exc:  # Artifact failures must not erase a completed rollout.
            artifact_errors.append(f"actions: {type(exc).__name__}: {exc}")
            LOGGER.exception("Failed to save LIBERO episode actions")

    save_video = video_policy == "all" or (video_policy == "failures" and not bool(saved.get("success")))
    video_path: Path | None = None
    if save_video:
        video_path = episode_dir / "rollout.mp4"
        try:
            atomic_write_video(video_path, frames, video_fps)
        except Exception as exc:  # Metrics remain the authoritative completion marker.
            artifact_errors.append(f"video: {type(exc).__name__}: {exc}")
            LOGGER.exception("Failed to save LIBERO episode video")

    saved["action_path"] = str(action_path) if action_path is not None and action_path.is_file() else None
    saved["video_path"] = str(video_path) if video_path is not None and video_path.is_file() else None
    saved["artifact_errors"] = artifact_errors
    atomic_write_json(episode_dir / "metrics.json", saved)
    return saved


def write_aggregate_outputs(output_dir: Path, records: Iterable[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda item: (int(item["benchmark_index"]), int(item["task_id"]), int(item["init_state_id"])),
    )
    lines = "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered)
    atomic_write_text(Path(output_dir) / "episodes.jsonl", lines)
    atomic_write_json(Path(output_dir) / "summary.json", summary)
