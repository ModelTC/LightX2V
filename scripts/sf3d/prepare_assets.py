"""Explicitly prepare the pinned SF3D tetrahedral grid, outside the code tree."""

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

REVISION = "ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2"
SHA256 = "1f4be37efc604d28d55a1a78c2aabefeeab7e63149f541aa45f9dd858ee35bb9"
URL = f"https://raw.githubusercontent.com/Stability-AI/stable-fast-3d/{REVISION}/load/tets/160_tets.npz"


def prepare_asset(model_path, source=None):
    destination = Path(model_path) / "assets" / "160_tets.npz"
    if destination.is_file():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
            return destination
        raise ValueError(f"Existing SF3D grid fails SHA256 validation: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sf3d-grid-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            if source is not None:
                with Path(source).open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
            else:
                with urllib.request.urlopen(URL, timeout=120) as response:
                    shutil.copyfileobj(response, output)
        if hashlib.sha256(Path(temporary).read_bytes()).hexdigest() != SHA256:
            raise ValueError("SF3D grid fails SHA256 validation; no asset was installed")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="Copy an existing grid instead of downloading it")
    args = parser.parse_args()
    print(prepare_asset(args.model_path, args.source))
