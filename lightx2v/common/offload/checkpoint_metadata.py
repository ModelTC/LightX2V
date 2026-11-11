"""Metadata readers shared by checkpoint adapters; tensor payloads stay on disk."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def read_checkpoint_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=_unique_object)


def read_safetensors_header(path):
    path = Path(path)
    with path.open("rb") as source:
        raw_size = source.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"Invalid safetensors header in {path}")
        size = struct.unpack("<Q", raw_size)[0]
        if size > 100 * 1024 * 1024 or size > path.stat().st_size - 8:
            raise ValueError(f"Invalid or truncated safetensors header in {path}")
        header = json.loads(source.read(size), object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header in {path}")
    return header


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
