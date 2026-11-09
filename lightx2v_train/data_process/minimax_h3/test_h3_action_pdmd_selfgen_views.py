import copy
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import yaml
from h3_prompt_test_fixtures import make_input

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("h3_staged_view_test", ROOT / "scripts/minimax_h3_action_pdmd_selfgen_gan_preflight.py")
views = importlib.util.module_from_spec(spec)
spec.loader.exec_module(views)
PHASE1 = ROOT / "configs/train/dmd/minimax_h3_t2av_action_pdmd8_shift12_3_r128_a128_adaln16_fsdp32.yaml"
PHASE2 = ROOT / "configs/train/dmd/minimax_h3_t2av_selfgen_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml"


def make_selfgen(tmp_path, count=64):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index in range(count):
        row = {"task": "t2av", "caption": f"Self generated {index}", "num_frames": 124, "target_height": 768, "target_width": 1344, "fps": 24}
        for key in ("condition_path", "video_latent_path", "audio_latent_path"):
            row[key] = f"{key}-{index}.pt"
            if key == "condition_path":
                torch.save({"task": "t2av", "prompt_embeds": torch.zeros(1, 2, 8), "text_token_tags": torch.ones(2, dtype=torch.long)}, source / row[key])
            else:
                # Invalid tensor payloads prove the generator only loads text.
                (source / row[key]).write_text("must remain deferred")
        rows.append(row)
    rows.extend(
        [
            {"task": "ref2av", "caption": "excluded reference"},
            {"task": "t2av", "num_frames": 124, "target_height": 1344, "target_width": 768, "fps": 24},
            {"task": "t2av", "num_frames": 124, "target_height": 768, "target_width": 1344, "fps": 30},
        ]
    )
    (source / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source


def make_action(tmp_path):
    prompts = make_input(tmp_path, count=2000)
    rows = views.validate_prompts(prompts)
    cache = tmp_path / "action_cache"
    (cache / "conditions").mkdir(parents=True)
    manifest_rows = []
    for index, row in enumerate(rows):
        relative = f"conditions/condition_{index:08d}.pt"
        (cache / relative).touch()
        manifest_rows.append({"source_index": index, "caption": row["prompt"].strip(), "task": "t2av", "condition_path": relative, "num_frames": 124, "target_height": 768, "target_width": 1344})
    content = "".join(json.dumps(row) + "\n" for row in manifest_rows).encode()
    (cache / "metadata.jsonl").write_bytes(content)
    receipt = {"prompt_count": 2000, "world_size": 32, "manifest_sha256": hashlib.sha256(content).hexdigest(), "preprocess_fingerprint": "fixture"}
    (cache / "prompt_cache.complete.json").write_text(json.dumps(receipt))
    (cache / "preprocess_config.json").write_text(json.dumps({"preprocess_fingerprint": "fixture"}))
    return cache, prompts


def test_selfgen_view_is_canonical_deferred_and_exact_gan_population(tmp_path, monkeypatch):
    source = make_selfgen(tmp_path)
    original = (source / "metadata.jsonl").read_bytes()
    output = tmp_path / "view"
    summary = views.prepare_selfgen_view(source, output)
    assert summary["rows"] == 64
    assert views.prepare_selfgen_view(source, output) == summary
    assert views.validate_view(output, "selfgen") == summary
    assert (source / "metadata.jsonl").read_bytes() == original
    assert not (output / "negative_condition.pt").exists()
    helper = views.load_module("_view_test_real", ROOT / "lightx2v_train/trainers/dmd/dmd2_gan_data.py")
    _, selected, _ = helper.load_gan_real_records(source)
    _, positive, _ = helper.load_gan_real_records(summary["manifest"])
    assert selected == positive

    sys.path.insert(0, str(ROOT))
    from lightx2v_train.data import minimax_h3_cache_dataset as dataset

    monkeypatch.setattr(dataset, "get_data_parallel_world_size", lambda: 32)
    monkeypatch.setattr(dataset, "get_data_parallel_rank", lambda: 0)
    data = yaml.safe_load(PHASE2.read_text())["data"]["train"]
    data.update(data_path=str(output), num_workers=0, pin_memory=False)
    loader = dataset.build_minimax_h3_cache_dataset(data)
    loader.sampler.configure_from_microbatch_offset(start_microbatch=0, gradient_accumulation_iters=1, fake_update_ratio=5)
    assert len(loader.dataset) == 64
    assert [sample["row"] for sample in loader.dataset.samples] == positive
    sample = next(iter(loader))
    assert sample["inputs"] == {}
    assert "positive" in sample["conditioning"]
    assert sample["meta"]["video_latent_path"][0].endswith(".pt")
    assert sample["meta"]["audio_latent_path"][0].endswith(".pt")


def test_action_view_has_absolute_paths_and_actual_loader_discovers_2000(tmp_path, monkeypatch):
    source, prompts = make_action(tmp_path)
    output = tmp_path / "action_view"
    summary = views.prepare_action_view(source, output, prompts)
    assert summary["rows"] == 2000
    assert views.prepare_action_view(source, output, prompts) == summary
    sys.path.insert(0, str(ROOT))
    from lightx2v_train.data import minimax_h3_cache_dataset as dataset

    monkeypatch.setattr(dataset, "get_data_parallel_world_size", lambda: 32)
    monkeypatch.setattr(dataset, "get_data_parallel_rank", lambda: 0)
    data = yaml.safe_load(PHASE1.read_text())["data"]["train"]
    data.update(data_path=str(output), num_workers=0, pin_memory=False)
    loader = dataset.build_minimax_h3_cache_dataset(data)
    assert len(loader.dataset) == 2000
    assert all(Path(sample["row"]["condition_path"]).is_absolute() for sample in loader.dataset.samples)
    assert all(not sample["row"].get("video_latent_path") for sample in loader.dataset.samples)


def test_four_worker_publication_is_atomic_and_idempotent(tmp_path):
    source = make_selfgen(tmp_path)
    output = tmp_path / "view"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: views.prepare_selfgen_view(source, output), range(4)))
    assert all(result == results[0] for result in results)
    assert not list(tmp_path.glob(".view.prepare-*"))


def test_dry_run_does_not_publish(tmp_path):
    source = make_selfgen(tmp_path)
    output = tmp_path / "uncreated/view"
    assert views.prepare_selfgen_view(source, output, dry_run=True)["rows"] == 64
    assert not output.parent.exists()


@pytest.mark.parametrize("change", ["source", "view", "receipt", "extra_manifest", "negative_condition"])
def test_changed_source_or_view_is_refused(tmp_path, change):
    source = make_selfgen(tmp_path)
    output = tmp_path / "view"
    views.prepare_selfgen_view(source, output)
    if change == "source":
        manifest = source / "metadata.jsonl"
        manifest.write_text(manifest.read_text().replace("excluded reference", "changed excluded reference"))
    elif change == "view":
        manifest = output / views.MANIFEST
        manifest.write_text(manifest.read_text().replace("Self generated 0", "Changed caption"))
    elif change == "extra_manifest":
        (output / "t2av/metadata.jsonl").write_text("{}\n")
    elif change == "negative_condition":
        (output / "t2av/landscape/negative_condition.pt").touch()
    else:
        receipt = output / views.RECEIPT
        receipt.write_text(receipt.read_text().replace(views.SCHEMA, "unknown_schema"))
    with pytest.raises(ValueError):
        views.prepare_selfgen_view(source, output)


@pytest.mark.parametrize("failure", ["too_small", "duplicate_condition", "duplicate_fingerprint", "missing_audio", "orientation"])
def test_unusable_selfgen_population_is_rejected_before_training(tmp_path, failure):
    source = make_selfgen(tmp_path, count=63 if failure == "too_small" else 64)
    manifest = source / "metadata.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if failure == "duplicate_condition":
        rows[1]["condition_path"] = rows[0]["condition_path"]
    elif failure == "duplicate_fingerprint":
        rows[0]["cache_fingerprint"] = rows[1]["cache_fingerprint"] = "same"
    elif failure == "missing_audio":
        (source / rows[0]["audio_latent_path"]).unlink()
    elif failure == "orientation":
        rows[0]["aspect_bucket"] = "portrait"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        views.prepare_selfgen_view(source, tmp_path / "view")
    assert not (tmp_path / "view").exists()


def resolved_configs(tmp_path, monkeypatch):
    for name, value in {
        "H3_MODEL_PATH": tmp_path / "model",
        "H3_ACTION_PDMD_VIEW": tmp_path / "action_view",
        "H3_SELFGEN_VIEW": tmp_path / "selfgen_view",
        "H3_GAN_INIT_CHECKPOINT": tmp_path / "prior/checkpoint-000005700",
        "H3_ACTION_PDMD_OUTPUT": tmp_path / "phase1",
        "H3_SELFGEN_GAN_OUTPUT": tmp_path / "phase2",
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.delenv("H3_GAN_MAX_ITERS", raising=False)
    return views.load_config(PHASE1), views.load_config(PHASE2)


def test_staged_recipes_match_requested_update_accounting(tmp_path, monkeypatch):
    phase1, phase2 = resolved_configs(tmp_path, monkeypatch)
    views.validate_recipe(phase1, gan_enabled=False)
    views.validate_recipe(phase2, gan_enabled=True)
    assert phase1["training"]["max_train_iters"] == 600
    assert int(phase2["training"]["max_train_iters"]) == 15000
    for config in (phase1, phase2):
        assert config["training"]["dmd"]["fake_update_ratio"] == 5
    assert phase2["training"]["student"]["checkpoint_path"] == str(tmp_path / "phase1/checkpoint-000000600")
    changed = copy.deepcopy(phase2)
    changed["data"]["train"]["data_path"] = str(tmp_path / "other_captions")
    with pytest.raises(ValueError, match="same filtered manifest"):
        views.validate_recipe(changed)
    phase1["training"]["max_train_iters"] = 100
    with pytest.raises(ValueError, match="100 student updates"):
        views.validate_recipe(phase1)
