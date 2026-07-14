import json
import os
import subprocess
from pathlib import Path

NEGATIVE_CACHE_NAMES = {"negative_condition.pt", "negative_prompt_condition.pt", "negative_prompt.pt"}


def load_audio_from_video_with_ffmpeg(audio_path, max_duration=None, sample_rate=44100):
    import torch

    from lightx2v_train.utils.ltx2_native import Audio

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
    ]
    if max_duration is not None:
        cmd.extend(["-t", f"{float(max_duration):.6f}"])
    cmd.extend(["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    if not result.stdout:
        return None
    waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32).reshape(1, -1).contiguous()
    return Audio(waveform=waveform, sampling_rate=sample_rate)


def install_video_audio_fallback(process_videos):
    original_load_audio = process_videos._load_audio_from_file
    video_suffixes = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}

    def load_audio(audio_path, max_duration=None):
        path = Path(audio_path)
        if path.suffix.lower() in video_suffixes:
            audio = load_audio_from_video_with_ffmpeg(path, max_duration=max_duration)
            if audio is not None:
                return audio
        return original_load_audio(audio_path, max_duration=max_duration)

    process_videos._load_audio_from_file = load_audio


def atomic_write_jsonl(path, rows):
    path = Path(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _condition_relative_path(rel_path):
    if rel_path.name.startswith("latent_"):
        return rel_path.parent / f"condition_{rel_path.name[7:]}"
    return rel_path


def paired_latent_manifest_rows(root, video_dir_name="latents", audio_dir_name="audio_latents", condition_dir_name="conditions"):
    root = Path(root).expanduser().resolve()
    if (root / ".precomputed").is_dir():
        root = root / ".precomputed"

    video_dir = root / video_dir_name
    audio_dir = root / audio_dir_name
    condition_dir = root / condition_dir_name
    for path in (video_dir, audio_dir, condition_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing latent dataset directory: {path}")

    audio_files = {path.relative_to(audio_dir): path for path in audio_dir.rglob("*.pt")}
    condition_files = {path.relative_to(condition_dir): path for path in condition_dir.rglob("*.pt")}
    rows = []
    skipped = 0
    for video_path in sorted(video_dir.rglob("*.pt")):
        rel_path = video_path.relative_to(video_dir)
        audio_path = audio_files.get(rel_path)
        condition_path = condition_files.get(_condition_relative_path(rel_path))
        if audio_path is None or condition_path is None:
            skipped += 1
            continue
        rows.append(
            {
                "video_latent_path": str(video_path.relative_to(root)),
                "audio_latent_path": str(audio_path.relative_to(root)),
                "condition_path": str(condition_path.relative_to(root)),
            }
        )
    return root, rows, skipped


def write_paired_latent_manifest(root, manifest_name="metadata.jsonl"):
    root, rows, skipped = paired_latent_manifest_rows(root)
    manifest_path = root / manifest_name
    atomic_write_jsonl(manifest_path, rows)
    return manifest_path, len(rows), skipped


def write_condition_manifest(root, manifest_name="metadata.jsonl"):
    root = Path(root).expanduser().resolve()
    paths = [path for path in sorted(root.rglob("*.pt")) if path.name not in NEGATIVE_CACHE_NAMES and not path.name.endswith(".tmp")]
    manifest_path = root / manifest_name
    atomic_write_jsonl(manifest_path, [{"condition_path": str(path.relative_to(root))} for path in paths])
    return manifest_path, len(paths)
