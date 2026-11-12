"""Small file helpers shared by H3-Image prompt preprocessing."""

import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def read_prompts(path, expected_count=None):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} prompts, found {len(rows)}: {path}")
    ids = set()
    for row in rows:
        if not isinstance(row.get("id"), int) or row["id"] in ids:
            raise ValueError("Prompt IDs must be unique integers")
        ids.add(row["id"])
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Empty prompt: {row['id']}")
    return rows


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def establish_recipe(path, recipe):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != recipe:
            raise ValueError(f"Existing preprocessing recipe differs: {path}")
    else:
        atomic_json(path, recipe)
