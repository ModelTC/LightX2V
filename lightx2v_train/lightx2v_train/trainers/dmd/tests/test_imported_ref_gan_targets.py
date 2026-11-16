"""Load imported teacher targets through the actual Ref GAN capability on CPU."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.trainers.dmd.dmd2_gan_data import DMD2GanRealStream, _load_ref_targets

TRAIN_ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("_imported_gan_fixtures", TRAIN_ROOT / "data_process/minimax_h3/test_import_lightx2v_ref_teacher_latents.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture
def teacher():
    fixture = fixtures.TensorImportTests()
    fixture.setUp()
    fixture.final_only()
    fixture.original_base()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def publish(fixture):
    fixture.run_import(check_tensors=True)
    manifest = fixture.directory / "manifest.json"
    receipt = {
        "schema": "h3_teacher_completion_v1",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_id": fixture.manifest["source_id"],
        "cache_fingerprint": fixture.manifest["cache_fingerprint"],
    }
    (fixture.directory / "completion.json").write_text(json.dumps(receipt))
    return DMD2GanRealStream(fixture.output, task="ref2av", geometry=None, seed=42)


def load(stream):
    capability = MiniMaxH3DistributionMatchingCapability(model_wrapper(), {"geometry_from_metadata": True})
    return stream.load(capability, 0, geometry=(124, 64, 96))


def test_final_teacher_without_payload_fingerprint_loads_full_gan_pair(teacher):
    stream = publish(teacher)
    before = {name: (teacher.directory / name).read_bytes() for name in ("condition.pt", "video_latents.pt", "audio_latents.pt", "manifest.json")}
    clean, condition = load(stream)
    assert clean.video.shape == (1, 222, 96)
    assert clean.audio.shape == (1, 414, 32)
    assert condition["condition_audio_latents"] is None
    assert len(condition["references"]) == 2
    assert condition["condition_video_latents"].shape == (12, 96)
    assert not list(teacher.directory.glob("state_*.pt"))
    assert before == {name: (teacher.directory / name).read_bytes() for name in before}


def test_legacy_teacher_uses_final_av_without_loading_trajectory(teacher):
    teacher.manifest.update(schema=fixtures.helper.LEGACY_SCHEMA, export_mode="trajectory", num_states=30, state_paths=[f"state_{index:03d}.pt" for index in range(30)])
    for name in teacher.manifest["state_paths"]:
        (teacher.directory / name).write_bytes(b"not a torch checkpoint")
    teacher.save()
    clean, _ = load(publish(teacher))
    assert clean.video.shape == (1, 222, 96)


@pytest.mark.parametrize("field,value", [("manifest_sha256", "wrong"), ("source_id", "other-sample"), ("cache_fingerprint", "other-condition")])
def test_teacher_completion_mismatch_is_rejected(teacher, field, value):
    stream = publish(teacher)
    receipt_path = teacher.directory / "completion.json"
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="completion|paired metadata"):
        load(stream)


def test_modified_teacher_manifest_is_rejected(teacher):
    stream = publish(teacher)
    teacher.manifest["seed"] += 1
    teacher.save()
    with pytest.raises(ValueError, match="manifest fingerprint"):
        load(stream)


def test_swapped_teacher_video_path_is_rejected(teacher):
    stream = publish(teacher)
    copy = teacher.directory / "another_video.pt"
    copy.write_bytes((teacher.directory / "video_latents.pt").read_bytes())
    stream.rows[0]["video_latent_path"] = str(copy)
    with pytest.raises(ValueError, match="video_latent_path.*finalized manifest"):
        load(stream)


def test_present_teacher_payload_fingerprint_must_still_match(teacher):
    stream = publish(teacher)
    path = teacher.directory / "video_latents.pt"
    value = torch.load(path, weights_only=True)
    value["latent_fingerprint"] = "wrong"
    torch.save(value, path)
    with pytest.raises(ValueError, match="latent_fingerprint"):
        load(stream)


@pytest.mark.parametrize("mutation", ["shape", "nonfinite", "normalized"])
def test_teacher_target_tensor_checks_remain_enforced(teacher, mutation):
    stream = publish(teacher)
    path = teacher.directory / "video_latents.pt"
    value = torch.load(path, weights_only=True)
    if mutation == "shape":
        value["latents"] = value["latents"][:, :-1]
    elif mutation == "nonfinite":
        value["latents"].flatten()[0] = float("nan")
    else:
        value["normalized"] = False
    torch.save(value, path)
    with pytest.raises(ValueError, match="target|normalization"):
        load(stream)


def test_selfgen_still_requires_matching_payload_fingerprint(teacher):
    stream = publish(teacher)
    row = dict(stream.rows[0])
    del row["teacher_export_schema"]
    with pytest.raises(ValueError, match="latent_fingerprint"):
        _load_ref_targets({"inputs": {}}, row)
    path = teacher.directory / "video_latents.pt"
    value = torch.load(path, weights_only=True)
    value["latent_fingerprint"] = row["latent_fingerprint"]
    torch.save(value, path)
    sample = {"inputs": {}}
    _load_ref_targets(sample, row)
    assert sample["inputs"]["audio_latents"].shape == (2, 32, 207)
