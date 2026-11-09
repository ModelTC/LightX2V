"""Shared JSON/JSONL prompt reader, independent of model dependencies."""

import json
from pathlib import Path


def read_json_rows(path):
    path = Path(path)
    content = path.read_text(encoding="utf-8-sig")
    if not content.strip():
        return []
    try:
        rows = json.loads(content)
    except json.JSONDecodeError as error:
        if error.msg != "Extra data":
            raise ValueError(f"Invalid prompt JSON in {path}: {error}") from error
        # Accept a JSONL dataset copied to data.json without changing contents.
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = next((rows[key] for key in ("data", "records", "examples") if key in rows), [rows])
    if not isinstance(rows, list) or not all(isinstance(row, (dict, str)) for row in rows):
        raise TypeError(f"{path} must contain prompt records or strings in a JSON list or JSONL.")
    return [{"prompt": row} if isinstance(row, str) else row for row in rows]
