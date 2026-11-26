"""Cached tensors preserve precision, dataset identity and checkpoint retention."""

import hashlib
import json

import pytest
import torch

from .checkpoint import latest_checkpoint, publish_checkpoint
from .data import CachedAVDataset


def test_cached_tensors_roundtrip_and_metadata_guard(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({"prompt": "test", "width": 1344, "height": 768, "num_frames": 124}))
    cache = tmp_path / "cache"
    (cache / "samples").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        CachedAVDataset(metadata, cache, 42)
    (cache / "complete.json").write_text(json.dumps({"metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(), "records": 1}))
    payload = {
        "record_index": 0,
        "condition": {"prompt_embeds": torch.randn(1, 8, 16).bfloat16(), "text_token_tags": torch.ones(8, dtype=torch.long)},
        "latents": {"video": torch.randn(1, 16, 96), "audio": torch.randn(1, 8, 32)},
    }
    torch.save(payload, cache / "samples/000000.pt")
    dataset = CachedAVDataset(metadata, cache, 42)
    condition, latents = dataset.load(0, "cpu")
    for name, value in condition.items():
        assert torch.equal(value, payload["condition"][name])
    for name, value in latents.items():
        assert value.dtype == torch.float32 and torch.equal(value, payload["latents"][name])
    metadata.write_text(metadata.read_text().replace("test", "changed"))
    with pytest.raises(ValueError, match="does not match"):
        CachedAVDataset(metadata, cache, 42)


def test_save_limit_one_across_stages_keeps_last_complete_snapshot(tmp_path):
    exports = tmp_path / "shared.pt"
    exports.write_text("required shared initialization")
    root = tmp_path / "checkpoints"
    root.mkdir()
    for stage, step in (("shared", 100), ("shared", 200), ("moe", 100)):
        temporary = root / ".partial"
        temporary.mkdir()
        (temporary / ".metadata").write_text("complete DCP")
        previous = latest_checkpoint(root, stage)
        if previous is not None:
            assert previous.is_dir()
        name = f"{stage}-{step:08d}"
        publish_checkpoint(root, temporary, name, {"stage": stage, "step": step}, 1)
        assert latest_checkpoint(root, stage) == root / name
        assert [p.name for p in root.iterdir() if p.is_dir()] == [name]
    assert latest_checkpoint(root, "shared") is None
    assert exports.read_text() == "required shared initialization"
