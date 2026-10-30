"""Download only the FL2VA/Ref2VA Comfy-QKV TrainingAdapter exports."""

import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "DiffSynth-Studio/MiniMax-H3-TrainingAdapter"
FILES = ("model_for_comfy_dit.safetensors", "model_ref2va_for_comfy_dit.safetensors")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output_dir, revision="master", workers=4):
    from modelscope.hub.snapshot_download import snapshot_download

    output_dir = Path(output_dir).expanduser().resolve()
    snapshot_download(model_id=MODEL_ID, revision=revision, local_dir=str(output_dir), allow_file_pattern=[*FILES, "README.md"], max_workers=workers)
    records = []
    for name in FILES:
        path = output_dir / name
        if not path.is_file() or path.stat().st_size < 16:
            raise RuntimeError(f"ModelScope download did not produce a complete adapter: {path}")
        records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {"model_id": MODEL_ID, "revision": revision, "qkv_layout": "comfy", "files": records}
    temporary = output_dir / f".download.complete.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(output_dir / "download.complete.json")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--download-workers", type=int, default=4)
    args = parser.parse_args()
    if args.download_workers < 1:
        parser.error("--download-workers must be positive")
    download(args.output_dir, args.revision, args.download_workers)
