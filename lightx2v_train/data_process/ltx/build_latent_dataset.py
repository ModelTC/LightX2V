#!/usr/bin/env python3
"""Build an LTX2 T2AV latent_dataset cache via the LTX preprocessing tools."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lightx2v_train.data.utils import read_records  # noqa: E402
from lightx2v_train.utils.preprocess import atomic_write_jsonl, install_video_audio_fallback, write_paired_latent_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build LTX2 T2AV video/audio/prompt latent_dataset cache from json/jsonl/csv metadata.")
    parser.add_argument("metadata", help="CSV/JSON/JSONL metadata with video and caption columns.")
    parser.add_argument("--output-dir", required=True, help="Output cache directory. Writes latents/, audio_latents/, conditions/ and metadata.jsonl.")
    parser.add_argument("--model-path", default="/data/nvme4/models/ltx-2.3/ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--text-encoder-path", default="/data/nvme0/gushiqiao/models/official_models/LTX-2")
    parser.add_argument(
        "--ltx2-repo",
        "--ltx2-trainer-repo",
        dest="ltx2_repo",
        default="/data/nvme5/gushiqiao/codes/LTX-2",
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
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def add_ltx2_paths(ltx2_repo):
    from lightx2v_train.utils.ltx2_native import ensure_ltx2_native_path

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

    import process_videos
    from process_dataset import preprocess_dataset
    from process_videos import parse_resolution_buckets

    install_video_audio_fallback(process_videos)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    manifest_path, count, skipped = write_paired_latent_manifest(output_dir)
    print(f"Wrote latent manifest: {manifest_path} ({count} samples, {skipped} skipped)", flush=True)


if __name__ == "__main__":
    main()
