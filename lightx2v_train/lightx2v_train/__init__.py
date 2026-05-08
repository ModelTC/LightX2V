"""Small training framework scaffold for image/video generation models."""

from __future__ import annotations

import sys
from pathlib import Path


def prefer_local_diffusers() -> None:
    """Use the sibling diffusers checkout when this repo is run in-place."""

    repo_root = Path(__file__).resolve().parents[2]
    diffusers_src = repo_root / "diffusers" / "src"
    if diffusers_src.exists():
        src = str(diffusers_src)
        if src not in sys.path:
            sys.path.insert(0, src)


prefer_local_diffusers()
