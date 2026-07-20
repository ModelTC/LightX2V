#!/usr/bin/env python3
"""Build an LTX2 T2AV latent_dataset cache via the LTX preprocessing tools."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightx2v_train.data.utils import read_records  # noqa: E402
from lightx2v_train.model_zoo.native.ltx2 import ensure_ltx2_native_path  # noqa: E402
from lightx2v_train.utils.constants import LTX2_NEGATIVE_PROMPT  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build LTX2 T2AV video/audio/prompt latent_dataset cache from json/jsonl/csv metadata.")
    parser.add_argument("metadata", help="CSV/JSON/JSONL metadata with video and caption columns.")
    parser.add_argument("--output-dir", required=True, help="Output cache directory. Writes latents/, audio_latents/, conditions/ and metadata.jsonl.")
    parser.add_argument("--model-path", default="/path/to/models/ltx-2.3/ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--text-encoder-path", default="/path/to/models/LTX-2")
    parser.add_argument(
        "--ltx2-repo",
        "--ltx2-trainer-repo",
        dest="ltx2_repo",
        default="/path/to/LTX-2",
        help="Reference LTX-2 repo path for process_dataset/process_videos.",
    )
    parser.add_argument("--resolution-buckets", default="768x768x49", help="Semicolon separated WxHxF buckets.")
    parser.add_argument("--video-column", default=None)
    parser.add_argument("--caption-column", "--prompt-column", dest="caption_column", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--remove-llm-prefixes", action="store_true")
    parser.add_argument("--skip-audio", action="store_true", help="Forwarded to LTX preprocessing. T2AV flow/tf usually need audio.")
    parser.add_argument("--audio-durations", default=None, help="Audio duration buckets, e.g. 2.0;4.0;8.0.")
    parser.add_argument("--load-text-encoder-in-8bit", action="store_true")
    parser.add_argument("--skip-negative-cache", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_ltx2_paths(ltx2_repo):
    ensure_ltx2_native_path()
    repo = Path(ltx2_repo).expanduser().resolve()
    paths = [
        repo / "packages" / "ltx-trainer" / "src",
        repo / "packages" / "ltx-trainer" / "scripts",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing LTX-2 trainer paths: {missing}")
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def write_manifest(root, require_audio):
    root = Path(root).expanduser().resolve()
    video_dir = root / "latents"
    audio_dir = root / "audio_latents"
    condition_dir = root / "conditions"
    required_dirs = [video_dir, condition_dir]
    if require_audio:
        required_dirs.append(audio_dir)
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing latent dataset directory: {path}")

    audio_files = {path.relative_to(audio_dir): path for path in audio_dir.rglob("*.pt")} if audio_dir.is_dir() else {}
    condition_files = {path.relative_to(condition_dir): path for path in condition_dir.rglob("*.pt")}
    rows = []
    skipped = 0
    for video_path in sorted(video_dir.rglob("*.pt")):
        rel_path = video_path.relative_to(video_dir)
        condition_rel_path = rel_path
        if rel_path.name.startswith("latent_"):
            condition_rel_path = rel_path.parent / f"condition_{rel_path.name[7:]}"
        audio_path = audio_files.get(rel_path)
        condition_path = condition_files.get(condition_rel_path)
        if condition_path is None or (require_audio and audio_path is None):
            skipped += 1
            continue
        row = {
            "video_latent_path": str(video_path.relative_to(root)),
            "condition_path": str(condition_path.relative_to(root)),
        }
        if audio_path is not None:
            row["audio_latent_path"] = str(audio_path.relative_to(root))
        rows.append(row)

    manifest_path = root / "metadata.jsonl"
    atomic_write_jsonl(manifest_path, rows)
    return manifest_path, len(rows), skipped


def write_negative_condition(args, output_dir, compute_captions_embeddings):
    negative_path = output_dir / "negative_condition.pt"
    if negative_path.is_file() and not args.overwrite:
        return negative_path

    metadata_path = output_dir / f"_negative_prompt.{os.getpid()}.jsonl"
    atomic_write_jsonl(metadata_path, [{"video": negative_path.name, "caption": LTX2_NEGATIVE_PROMPT}])
    try:
        compute_captions_embeddings(
            dataset_file=str(metadata_path),
            output_dir=str(output_dir),
            model_path=args.model_path,
            text_encoder_path=args.text_encoder_path,
            caption_column="caption",
            media_column="video",
            batch_size=1,
            device=args.device,
            load_in_8bit=args.load_text_encoder_in_8bit,
            overwrite=args.overwrite,
        )
    finally:
        metadata_path.unlink(missing_ok=True)
    if not negative_path.is_file():
        raise RuntimeError(f"Failed to build negative prompt cache: {negative_path}")
    return negative_path


def limited_metadata(metadata, output_dir, max_samples, caption_column):
    if max_samples is None:
        return metadata
    rows = []
    for row in read_records(metadata, prompt_column=caption_column or "caption"):
        rows.append(row)
        if len(rows) >= max_samples:
            break
    if not rows:
        raise RuntimeError(f"No rows found in metadata: {metadata}")
    limited_path = Path(output_dir) / "_limited_metadata.jsonl"
    atomic_write_jsonl(limited_path, rows)
    return str(limited_path)


def main():
    args = parse_args()
    add_ltx2_paths(args.ltx2_repo)

    from ltx_trainer.gpu_utils import free_gpu_memory_context
    from process_captions import compute_captions_embeddings
    from process_dataset import preprocess_dataset
    from process_videos import parse_resolution_buckets

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_negative_cache:
        with free_gpu_memory_context():
            negative_path = write_negative_condition(args, output_dir, compute_captions_embeddings)
        print(f"Wrote negative condition: {negative_path}", flush=True)

    metadata = limited_metadata(args.metadata, output_dir, args.max_samples, args.caption_column)
    parsed_buckets = parse_resolution_buckets(args.resolution_buckets) if args.resolution_buckets else None
    audio_durations = [float(item) for item in args.audio_durations.split(";")] if args.audio_durations else None

    preprocess_dataset(
        dataset_file=metadata,
        resolution_buckets=parsed_buckets,
        model_path=args.model_path,
        text_encoder_path=args.text_encoder_path,
        device=args.device,
        output_dir=str(output_dir),
        video_column=args.video_column,
        caption_column=args.caption_column,
        batch_size=args.batch_size,
        vae_tiling=args.vae_tiling,
        decode=args.decode,
        remove_llm_prefixes=args.remove_llm_prefixes,
        skip_audio=args.skip_audio,
        audio_durations=audio_durations,
        load_text_encoder_in_8bit=args.load_text_encoder_in_8bit,
        overwrite=args.overwrite,
    )
    manifest_path, count, skipped = write_manifest(output_dir, require_audio=not args.skip_audio)
    print(f"Wrote latent manifest: {manifest_path} ({count} samples, {skipped} skipped)", flush=True)


if __name__ == "__main__":
    main()
