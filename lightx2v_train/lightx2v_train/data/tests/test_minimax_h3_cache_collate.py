"""Exercise exported Ref conditions through the actual training DataLoader."""

import json
from types import SimpleNamespace

import pytest
import torch
from lightx2v_train.data import minimax_h3_cache_dataset as cache
from lightx2v_train.model_zoo.minimax_h3.minimax_h3_ref2av import MiniMaxH3Ref2AVModel
from torch.utils.data import default_collate


def make_loader(tmp_path, monkeypatch, *, workers, optional_nulls):
    references = []
    for index, (height, width) in enumerate(((2, 2), (4, 6))):
        reference = {
            "kind": "image",
            "normalized": True,
            "num_latent_frames": 1,
            "latent_height": height,
            "latent_width": width,
            "video_latents": torch.full((height * width // 4, 96), float(index + 1)),
            "num_audio_latents": 0,
        }
        if optional_nulls:
            reference["audio_latents"] = None
        references.append(reference)
    positive = {
        "task": "ref2av",
        "target_num_frames": 124,
        "target_height": 64,
        "target_width": 96,
        "prompt_embeds": torch.ones(1, 3, 8),
        "text_token_tags": torch.tensor([0, 1, 1]),
        "references": references,
    }
    for name in ("video.pt", "audio.pt"):
        # Invalid tensor bytes deliberately catch accidental eager target loading.
        (tmp_path / name).write_bytes(b"targets must remain deferred")
    rows = []
    for index in range(2):
        condition = tmp_path / f"condition{index}.pt"
        torch.save({"conditioning": {"positive": positive}}, condition)
        row = {
            "caption": f"prompt-{index}",
            "task": "ref2av",
            "target_num_frames": 124,
            "target_height": 64,
            "target_width": 96,
            "fps": 24,
            "condition_path": str(condition),
            "video_latent_path": str(tmp_path / "video.pt"),
            "audio_latent_path": str(tmp_path / "audio.pt"),
            "reference_image_count": 2,
            "reference_video_count": 0,
            "reference_audio_count": 0,
            "packed_sequence_tokens_124": 1000 + index,
        }
        if optional_nulls:
            row["duration"] = None
        rows.append(row)
    manifest = tmp_path / "metadata.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(cache, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cache, "get_data_parallel_rank", lambda: 0)
    return cache.build_minimax_h3_ref_cache_dataset(
        {
            "data_path": str(manifest),
            "batch_size": 1,
            "num_workers": workers,
            "pin_memory": False,
            "defer_latent_loading": True,
            "reference_cost_sampler": {
                "batch_mode": "count_coverage",
                "image_counts": [2],
                "balance_orientation": False,
                "strict_full_epoch": False,
                "remainder_policy": "rotating_drop",
            },
        }
    )


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("optional_nulls", [False, True])
def test_ref_loader_preserves_optional_audio_and_packed_geometry(tmp_path, monkeypatch, workers, optional_nulls):
    loader = make_loader(tmp_path, monkeypatch, workers=workers, optional_nulls=optional_nulls)
    loader.sampler.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=5)
    if optional_nulls:
        with pytest.raises(TypeError, match="NoneType"):
            default_collate([loader.dataset[0]])
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["inputs"] == {}
    assert batch["meta"]["video_latent_path"] == [str(tmp_path / "video.pt")]
    assert batch["meta"]["audio_latent_path"] == [str(tmp_path / "audio.pt")]
    positive = batch["conditioning"]["positive"]
    assert positive["prompt_embeds"].shape == (1, 3, 8)
    assert positive["text_token_tags"].shape == (1, 3)
    # Preserve the established singleton-stripping + collation shape convention.
    assert positive["references"][0]["video_latents"].shape == (1, 96)
    assert positive["references"][1]["video_latents"].shape == (1, 6, 96)
    if optional_nulls:
        assert batch["meta"]["duration"] is None
        assert all(reference["audio_latents"] is None for reference in positive["references"])
    else:
        assert all("audio_latents" not in reference for reference in positive["references"])
    model = SimpleNamespace(device=torch.device("cpu"), running_dtype=torch.float32, patch_size=(1, 2, 2), video_latent_channels=24, audio_latent_channels=32, text_dim=8)
    prepared = MiniMaxH3Ref2AVModel.prepare_text_condition(model, positive)
    torch.testing.assert_close(prepared["condition_video_latents"], torch.cat((torch.ones(1, 96), torch.full((6, 96), 2.0))))
    assert prepared["condition_audio_latents"] is None
    assert [(ref.kind, ref.latent_height, ref.latent_width) for ref in prepared["references"]] == [("image", 2, 2), ("image", 4, 6)]
    assert (prepared["target_num_frames"], prepared["target_height"], prepared["target_width"]) == (124, 64, 96)


def test_none_free_collation_matches_pytorch():
    samples = [{"tensor": torch.arange(6).view(2, 3), "number": 4, "text": "ref", "ordered": [{"kind": "image"}, {"kind": "video"}]}]
    expected = default_collate(samples)
    actual = cache.collate_latent_samples(samples)
    torch.testing.assert_close(actual["tensor"], expected["tensor"])
    torch.testing.assert_close(actual["number"], expected["number"])
    assert actual["text"] == expected["text"]
    assert actual["ordered"] == expected["ordered"]


@pytest.mark.parametrize("values", [[None, torch.ones(2)], [torch.ones(2), None]])
def test_mixed_missing_and_present_values_are_not_silently_dropped(values):
    with pytest.raises(TypeError):
        cache.collate_latent_samples([{"optional": value} for value in values])
