#!/usr/bin/env python3
"""Extract ViPE long WebDataset tar shards into Wan T2V video dataset layout."""

import argparse
import json
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ViPE long tar shards to video + metadata.jsonl.")
    parser.add_argument(
        "--tar-dir",
        default="/data/nvme7/wangshankun/train_data/vipe_long/long",
        help="Directory containing *.tar shards.",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/nvme7/wangshankun/train_data/vipe_long/extracted",
        help="Output directory for videos/ and metadata.jsonl.",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Only process the first N tar shards (sorted by name).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after extracting this many samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tar_dir = Path(args.tar_dir)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    tar_paths = sorted(tar_dir.glob("*.tar"))
    if args.max_shards is not None:
        tar_paths = tar_paths[: args.max_shards]
    if not tar_paths:
        raise FileNotFoundError(f"No tar files found in {tar_dir}")

    samples = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as archive:
            members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
            sample_ids = sorted({name.rsplit(".", 1)[0] for name in members if name.endswith(".mp4")})
            for sample_id in sample_ids:
                mp4_name = f"{sample_id}.mp4"
                txt_name = f"{sample_id}.txt"
                if mp4_name not in members or txt_name not in members:
                    continue

                shard_prefix = tar_path.stem
                video_name = f"{shard_prefix}_{sample_id}.mp4"
                video_path = video_dir / video_name
                if not video_path.exists():
                    archive.extract(members[mp4_name], path=output_dir / "_tmp")
                    (output_dir / "_tmp" / mp4_name).replace(video_path)

                caption = archive.extractfile(members[txt_name]).read().decode("utf-8").strip()
                samples.append(
                    {
                        "video": f"video/{video_name}",
                        "caption": caption,
                    }
                )
                if args.max_samples is not None and len(samples) >= args.max_samples:
                    break
        if args.max_samples is not None and len(samples) >= args.max_samples:
            break

    tmp_dir = output_dir / "_tmp"
    if tmp_dir.exists():
        for leftover in tmp_dir.iterdir():
            leftover.unlink(missing_ok=True)
        tmp_dir.rmdir()

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Extracted {len(samples)} samples to {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
