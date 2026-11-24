"""Assemble a numbered YuE2 request from shared lyrics, scores and presets."""

import json
import sys
from pathlib import Path


def read_sample(directory, function_id):
    root = Path(directory)
    presets = json.loads((root / "presets.json").read_text(encoding="utf-8"))
    preset = presets["functions"][str(int(function_id))]
    request = presets["defaults"] | {
        "id": preset["id"],
        "cot": preset["cot"],
        "style": presets["styles"][preset["style"]],
        "lyrics": (root / preset["lyrics_file"]).read_text(encoding="utf-8").removesuffix("\n") if preset["lyrics_file"] else "",
        "abc": (root / preset["abc_file"]).read_text(encoding="utf-8") if preset["abc_file"] else None,
    }
    return {key: value for key, value in request.items() if value is not None}


if __name__ == "__main__":
    for key, value in read_sample(sys.argv[1], sys.argv[2]).items():
        value = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        sys.stdout.buffer.write(f"--{key}\0{value}\0".encode())
