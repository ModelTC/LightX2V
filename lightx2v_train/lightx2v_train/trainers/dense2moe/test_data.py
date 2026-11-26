"""Native geometry, empirical sampling frequency and deterministic resume."""

import json
from collections import Counter

import pytest
import torch

from .data import PromptDataset
from .runtime import AVSample


def write_records(path):
    sizes = [(1344, 768), (1344, 768), (768, 1344), (1344, 768), (768, 1344), (992, 1024), (1536, 672)]
    rows = [{"prompt": str(i), "width": w, "height": h, "num_frames": 124} for i, (w, h) in enumerate(sizes)]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return rows


def test_epoch_preserves_records_and_resolution_frequency(tmp_path):
    path = tmp_path / "metadata.jsonl"
    rows = write_records(path)
    data = PromptDataset(path, seed=42)
    expected = Counter((row["width"], row["height"]) for row in rows)
    epochs = []
    for epoch in range(2):
        indices, records = [], []
        for position in range(epoch * len(rows), (epoch + 1) * len(rows)):
            index, record = data.sample(position // 4, position % 4, 4)
            indices.append(index)
            records.append(record)
            assert record == rows[index]
        assert sorted(indices) == list(range(len(rows)))
        assert Counter((row["width"], row["height"]) for row in records) == expected
        epochs.append(indices)
    assert epochs[0] != epochs[1]


def test_resume_and_rank_sharding_do_not_consume_training_rng(tmp_path):
    path = tmp_path / "metadata.jsonl"
    write_records(path)
    data = PromptDataset(path, seed=42)
    baseline = [data.sample(i // 4, i % 4, 4) for i in range(24)]
    torch.manual_seed(17)
    state = torch.random.get_rng_state().clone()
    for rank in range(4):
        resumed = PromptDataset(path, seed=42)
        assert [resumed.sample(step, rank, 4) for step in range(3, 6)] == [baseline[step * 4 + rank] for step in range(3, 6)]
    torch.testing.assert_close(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("width,height,frames,latent_frames", [(1344, 768, 124, 37), (768, 1344, 124, 37), (992, 1024, 124, 37), (1536, 672, 107, 32)])
def test_native_geometry_controls_video_packing(width, height, frames, latent_frames):
    condition = {"text_token_tags": torch.ones(3, dtype=torch.long), "prompt_embeds": torch.zeros(1, 3, 5120)}
    sample = AVSample(condition, {"width": width, "height": height, "num_frames": frames}, "cpu")
    assert sample.shapes["video"] == (1, latent_frames * (height // 32) * (width // 32), 96)
    positions = sample.layout.position_ids[sample.layout.video_indices]
    assert len(positions[:, 1].unique()) == height // 32
    assert len(positions[:, 2].unique()) == width // 32

    audio_frames = round(frames / 24 * 40)
    assert sample.shapes["audio"] == (1, 2 * audio_frames, 32)
    assert len(sample.layout.audio_indices) == 2 * audio_frames
    audio_positions = sample.layout.position_ids[sample.layout.audio_indices]
    torch.testing.assert_close(audio_positions[:audio_frames, 0], audio_positions[audio_frames:, 0])
