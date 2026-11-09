"""External data.json must be read identically by preflight and encoding."""

import json

import pytest
import test_h3_pdmd_gan_launch as fixtures


@pytest.mark.parametrize("format", ["array", "data", "records", "examples", "jsonl_renamed", "strings"])
def test_external_json_formats_match_preflight_and_encoder(tmp_path, format):
    records = [{"id": f"action-{index}", "task": "t2va", "prompt": f"Action {index}\nSound {index}", "num_frames": 124, "fps": 24} for index in range(2000)]
    source = tmp_path / "data.json"
    if format == "jsonl_renamed":
        content = "\n".join(json.dumps(row) for row in records)
    else:
        data = records if format == "array" else [row["prompt"] for row in records] if format == "strings" else {format: records}
        content = json.dumps(data, indent=2)
    source.write_text(content, encoding="utf-8")
    validated = fixtures.preflight.validate_prompts(source)
    encoded = fixtures.builder.read_samples(source, "prompt", "t2av", 0, None)
    assert len(validated) == len(encoded) == 2000
    assert [row["prompt"] for row in validated] == [row["prompt"] for row in encoded]
    assert all(row["task"] == "t2av" for row in encoded)


def test_invalid_and_empty_sources_fail_preflight(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("\n")
    assert fixtures.builder._read_json_rows(path) == []  # Empty rank shards are valid.
    with pytest.raises(ValueError, match="found 0"):
        fixtures.preflight.validate_prompts(path)
    path.write_text('[{"prompt": "incomplete"}')
    with pytest.raises(ValueError, match="Invalid prompt JSON"):
        fixtures.preflight.validate_prompts(path)
