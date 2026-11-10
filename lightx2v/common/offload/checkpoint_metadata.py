"""Metadata readers shared by checkpoint adapters; tensor payloads stay on disk."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from lightx2v_platform.checkpoint_metadata import read_checkpoint_json, read_safetensors_header

__all__ = ["checkpoint_content_digest", "read_checkpoint_json", "read_safetensors_header"]


def checkpoint_content_digest(path):
    """Reuse a matching local HF download digest, otherwise hash the payload."""
    path = Path(path)
    for root in path.parents:
        metadata = root / ".cache/huggingface/download" / path.relative_to(root)
        metadata = metadata.with_name(f"{metadata.name}.metadata")
        if not metadata.is_file():
            continue
        with metadata.open(encoding="utf-8") as source:
            source.readline()
            digest = source.readline().rstrip("\r\n")
            timestamp = source.readline().rstrip("\r\n")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid SHA-256 on line 2 of Hugging Face metadata file {metadata}")
        try:
            metadata_second = int(float(timestamp))
        except (ValueError, OverflowError):
            metadata_second = None
        if metadata_second == int(path.stat().st_mtime):
            return digest
        break

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
