"""Download H3 T2AV metadata and optionally its original teacher videos."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--videos", action="store_true", help="Download videos referenced by the full metadata")
    parser.add_argument("--smoke-videos", action="store_true", help="Only download the first two four-rank batches of each stage, seed=42")
    args = parser.parse_args()
    url = f"https://modelscope.cn/datasets/DiffSynth-Studio/MiniMax-H3-Self-Generated-Dataset/resolve/{args.revision}/metadata.jsonl"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise ValueError("T2AV metadata must contain nonempty prompts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.jsonl").write_bytes(response.content)
    provenance = {"url": url, "rows": len(rows), "sha256": hashlib.sha256(response.content).hexdigest()}
    (args.output_dir / "source.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance))
    if args.videos or args.smoke_videos:
        indices = range(len(rows))
        if args.smoke_videos:
            import torch

            indices = sorted({int(i) for offset in (0, 100000, 200000) for i in torch.randperm(len(rows), generator=torch.Generator().manual_seed(42 + offset))[:8]})
        base = url.rsplit("/", 1)[0]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for path in executor.map(lambda index: download_video(base, rows[index]["video_path"], args.output_dir), indices):
                print(path, flush=True)


def download_video(base, relative_path, output_dir):
    target = output_dir / relative_path
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    with requests.get(f"{base}/{relative_path}", stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    main()
