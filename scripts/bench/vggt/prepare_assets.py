"""Prepare pinned VGGT weights and the original ETH3D latency inputs."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from scripts.bench.reconstruction.fixtures import load_requests, sha256, unique_images

MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
ETH3D_REVISION = "64154bdca815ef7712f5f2621e231bac7dc99d59"


def prepare_inputs(output):
    for relative, image in unique_images(load_requests(Path(__file__).with_name("requests.json"))).items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != image["sha256"]:
            source = hf_hub_download("Livioni/eth3d_omnivggt", relative.removeprefix("vggt/"), repo_type="dataset", revision=ETH3D_REVISION)
            shutil.copyfile(source, target)
        if target.stat().st_size != image["size_bytes"] or sha256(target) != image["sha256"]:
            raise ValueError(f"Input bytes do not match the frozen manifest: {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["weights", "inputs"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "weights":
        snapshot_download("facebook/VGGT-1B", revision=MODEL_REVISION, local_dir=args.output, allow_patterns=["model.pt", "LICENSE*", "README.md"])
    else:
        prepare_inputs(args.output)
