"""Synthetic prompt inputs for CPU tests; no production dataset dependency."""

import json


def make_input(tmp_path, count=2000):
    source = tmp_path / "prompts.jsonl"
    rows = ({"id": f"p-{index}", "task": "t2av", "prompt": f"Action {index}", "num_frames": 124, "fps": 24} for index in range(count))
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return source
