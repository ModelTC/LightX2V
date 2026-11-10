"""Low-level checkpoint readers shared by conversion and runtime loading."""

from __future__ import annotations

import json
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
