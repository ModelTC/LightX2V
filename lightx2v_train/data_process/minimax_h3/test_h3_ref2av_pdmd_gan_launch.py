"""CPU regression checks for the Ref5700 -> self-generated Ref PDMD+GAN launch."""

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.tests.test_ref_gan_data import paired_cache as paired_cache
from lightx2v_train.trainers.dmd.tests.test_ref_gan_data import write_rows

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_h3_action_pdmd_selfgen_gan_stages import weights  # noqa: E402


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


preflight = module("_test_ref_pdmd_gan_preflight", "scripts/minimax_h3_ref2av_pdmd_gan_preflight.py")
CONFIG = ROOT / "configs/train/dmd/minimax_h3_ref2av_pdmd8_gan_from5700_shift12_3_fake5_fsdp32.yaml"
LAUNCHER = ROOT / "scripts/run_minimax_h3_ref2av_pdmd8_gan_fsdp32_32gpu_acp.sh"


@pytest.fixture
def config(tmp_path, monkeypatch):
    for key, value in {
        "H3_MODEL_PATH": tmp_path / "original",
        "H3_SCORE_MODEL_PATH": tmp_path / "premerged-score",
        "H3_REF_GAN_INIT_CHECKPOINT": tmp_path / "prior/train/checkpoint-000005700",
        "H3_REF_GAN_OUTPUT": tmp_path / "new-gan",
        "H3_REF_GAN_REAL_CACHE": tmp_path / "selfgen/ref2av_imageonly/metadata.jsonl",
        "H3_REF_GAN_MAX_ITERS": "15000",
        # Legacy sources must not enter this new self-generated-only recipe.
        "H3_REF2AV_CACHE": tmp_path / "must-not-use-old",
        "H3_NEW_CACHE": tmp_path / "must-not-use-curriculum",
    }.items():
        monkeypatch.setenv(key, str(value))
    return load_config(str(CONFIG))


def test_recipe_keeps_selfgen_only_ref_data_and_the_continuation_hyperparameters(config):
    source, _ = preflight.validate_recipe(config)
    training, model, data = config["training"], config["model"], config["data"]["train"]
    assert source.name == "checkpoint-000005700"
    assert (training["student"]["optimizer"]["learning_rate"], training["fake"]["optimizer"]["learning_rate"]) == (2e-5, 4e-6)
    assert (training["dmd"]["num_inference_steps"], training["dmd"]["fake_update_ratio"]) == (8, 5)
    assert training["dmd"]["generation_shapes"] is None
    assert model["capabilities"]["distribution_matching"]["geometry_from_metadata"]
    assert training["student"]["checkpoint_path"] == training["fake"]["checkpoint_path"]
    assert model["pretrained_model_name_or_path"] != model["teacher"]["pretrained_model_name_or_path"]
    assert model["teacher"]["pretrained_model_name_or_path"] == model["fake"]["pretrained_model_name_or_path"]
    assert all(model[role]["training_adapter"] == {"enabled": False} for role in ("teacher", "fake"))
    assert not training["student"]["ema"]["enabled"]
    assert data["name"] == "minimax_h3_ref_cache_dataset" and data["defer_latent_loading"]
    assert data["data_path"] == training["dmd"]["gan"]["real_cache"]
    assert Path(data["data_path"]).parts[-2:] == ("ref2av_imageonly", "metadata.jsonl")
    assert "sources" not in data and "kind" not in data
    sampler = data["reference_cost_sampler"]
    assert sampler["image_counts"] == list(range(1, 10))
    assert sampler["batch_mode"] == "count_coverage" and sampler["require_image_only"] and sampler["require_compute_cost"]
    assert not any(sampler[key] for key in ("require_all_image_counts", "balance_image_counts", "balance_orientation", "strict_full_epoch"))
    assert sampler["remainder_policy"] == "rotating_drop"


@pytest.mark.parametrize("location", ["run", "checkpoint", "inside_checkpoint", "ancestor"])
def test_recipe_rejects_overwriting_or_containing_the_source_run(config, location):
    source = Path(config["training"]["student"]["checkpoint_path"])
    output = {"run": source.parent, "checkpoint": source, "inside_checkpoint": source / "gan", "ancestor": source.parent.parent}[location]
    config["training"]["output_dir"] = str(output)
    with pytest.raises(ValueError, match="new output directory"):
        preflight.validate_recipe(config)


@pytest.mark.parametrize("key,value", [("rank", 64), ("alpha", 64), ("rank_pattern", {"adaln_proj.linear": 16}), ("alpha_pattern", {"adaln_proj.linear": 16, "norm_out.linear": 32})])
def test_recipe_rejects_changed_lora_architecture(config, key, value):
    config["training"]["student"]["lora"][key] = value
    with pytest.raises(ValueError, match="regular128/128"):
        preflight.validate_recipe(config)


def test_dry_run_prints_only_preflight_and_32_gpu_training_without_executing_python(tmp_path):
    output = tmp_path / "not-created"
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="23456",
        H3_CODE_ROOT=str(ROOT.parent),
        H3_PYTHON="/deliberately/missing/python",
        H3_REF_GAN_OUTPUT=str(output),
        H3_SELFGEN_CACHE=str(tmp_path / "missing-selfgen"),
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    for key in ("LOCAL_KERNELS", "KERNELS_CACHE", "H3_KERNEL_SNAPSHOT", "H3_REF_GAN_CONFIG"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=env, capture_output=True, text=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 2
    assert commands[0] == ["/deliberately/missing/python", "-u", str(ROOT / "scripts/minimax_h3_ref2av_pdmd_gan_preflight.py"), str(CONFIG)]
    assert commands[1][:4] == ["/deliberately/missing/python", "-u", "-m", "torch.distributed.run"]
    assert "--nnodes=4" in commands[1] and "--nproc_per_node=8" in commands[1]
    assert "--rdzv_endpoint=127.0.0.1:23456" in commands[1]
    assert commands[1][-3:] == [str(ROOT / "train.py"), "--config", str(CONFIG)]
    assert not any(word in result.stdout for word in ("prepare_", "build_", "download", "turbo", "--cm"))
    assert not output.exists() and not (tmp_path / "missing-selfgen").exists()


@pytest.fixture
def source_checkpoint(config):
    source, adapter = preflight.validate_recipe(config)
    weights(source, 5700)
    old_recipe = copy.deepcopy(config)
    old_recipe["training"]["dmd"]["fake_update_ratio"] = 10
    old_recipe["training"]["dmd"].pop("gan")
    for rank in range(32):
        torch.save({"iteration": 5700, "world_size": 32, "rank": rank, "recipe": old_recipe}, source / f"official_pdmd.rank{rank:05d}.pt")
    return source, adapter


def source_state(source, mutate):
    path = source / "official_pdmd.rank00000.pt"
    state = torch.load(path, weights_only=False)
    mutate(state)
    torch.save(state, path)


def test_source_warmstart_requires_trained_fake_weights_not_just_student_adapter(config, source_checkpoint):
    source, adapter = source_checkpoint
    with patch("torch.distributed.checkpoint.FileSystemReader") as reader:
        reader.return_value.read_metadata.return_value = SimpleNamespace(state_dict_metadata={"fake_model.transformer_blocks.0.weight": object()})
        result = preflight.validate_source(config, source, adapter)
        assert result == {"checkpoint": str(source), "iteration": 5700, "lora_tensors": 6, "weights_only": ["student", "fake"]}
        reader.assert_called_once_with(str(source / "dist_state"))
        reader.return_value.read_metadata.return_value = SimpleNamespace(state_dict_metadata={"model.lora_A.weight": object()})
        with pytest.raises(ValueError, match="trained fake_model"):
            preflight.validate_source(config, source, adapter)


@pytest.mark.parametrize("problem,match", [("fl2av", "Ref2AV"), ("world", "32-rank"), ("rank", "32-rank"), ("missing_rank", "missing rank"), ("alpha", "LoRA architecture")])
def test_source_rejects_other_tasks_incomplete_ranks_and_incompatible_adapters(config, source_checkpoint, problem, match):
    source, adapter = source_checkpoint
    if problem == "missing_rank":
        (source / "official_pdmd.rank00031.pt").unlink()
    else:

        def mutate(state):
            if problem == "fl2av":
                state["recipe"]["model"]["name"] = "minimax_h3_t2av"
            elif problem == "world":
                state["world_size"] = 16
            elif problem == "rank":
                state["rank"] = 1
            else:
                state["recipe"]["training"]["student"]["lora"]["alpha"] = 64

        source_state(source, mutate)
    with pytest.raises(ValueError, match=match):
        preflight.validate_source(config, source, adapter)


@pytest.fixture
def complete_ref_data(config, paired_cache):
    manifest, examples = paired_cache
    rows = []
    for index in range(32):
        row = dict(
            examples[index % len(examples)], source_id=f"full-AV-{index}", cache_fingerprint=f"condition-{index}", latent_fingerprint=f"paired-AV-{index}", packed_sequence_tokens_124=1000 + index
        )
        for key in ("condition_path", "video_latent_path", "audio_latent_path"):
            row[key] = f"preflight-{index}-{key}.pt"
            # Preflight must validate manifest accounting without loading tensors.
            (manifest.parent / row[key]).write_bytes(b"not a tensor: preflight must remain metadata-only")
        rows.append(row)
    write_rows(manifest, rows)
    receipt = manifest.with_suffix(".complete.json")
    receipt.write_text(
        json.dumps(
            {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "completed_count": 32, "failed_count": 0, "input_total_rows": 32, "preprocess_fingerprint": "complete-ref-AV-cache"}
        )
    )
    config["training"]["dmd"]["gan"]["real_cache"] = str(manifest)
    config["data"]["train"]["data_path"] = str(manifest)
    return manifest, rows, receipt


def test_data_accepts_completed_paired_ref_metadata_without_loading_tensor_payloads(config, complete_ref_data):
    manifest, _, _ = complete_ref_data
    with patch("torch.load", side_effect=AssertionError("preflight must not load model/data tensors")):
        result = preflight.validate_data(config)
    assert result["manifest"] == str(manifest) and result["rows"] == 32
    assert len(result["fingerprint"]) == 64
    assert {tuple(record["geometry"]) for record in result["geometry_rows"]} == {(124, 64, 96), (124, 96, 64)}
    assert sum(record["rows"] for record in result["geometry_rows"]) == 32


@pytest.mark.parametrize("problem,match", [("condition_only", "audio_latent_path"), ("video_reference", "image-only"), ("missing_receipt", "fully merged"), ("stale_receipt", "completion receipt")])
def test_data_rejects_unpaired_nonimage_or_unfinished_selfgen_input(config, complete_ref_data, problem, match):
    manifest, rows, receipt = complete_ref_data
    if problem == "condition_only":
        rows[0].pop("audio_latent_path")
    elif problem == "video_reference":
        rows[0]["reference_video_count"] = 1
    elif problem == "missing_receipt":
        receipt.unlink()
    else:
        rows[0]["caption"] += " changed after completion"
    if problem != "missing_receipt":
        write_rows(manifest, rows)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        preflight.validate_data(config)
