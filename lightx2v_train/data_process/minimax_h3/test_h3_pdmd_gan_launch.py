import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESS = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_PROCESS))
import build_minimax_h3_prompt_latents as builder
import prepare_minimax_h3_gan_prompts as coordinator
from h3_prompt_test_fixtures import make_input


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = load_module("h3_gan_launch_preflight_test", ROOT / "scripts/minimax_h3_pdmd_gan_preflight.py")
CONFIG = ROOT / "configs/train/dmd/minimax_h3_t2av_action_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml"
LAUNCHER = ROOT / "scripts/run_minimax_h3_t2av_action_pdmd8_gan_fsdp32_32gpu_acp.sh"
MODEL = {"text_hidden_size": 8, "video_latent_channels": 24, "root": "/model", "component_files": []}


def encode_args(source, output, rank=0, world=3):
    return [
        str(source),
        "--prompt-column",
        "prompt",
        "--task",
        "t2av",
        "--model-path",
        "/model",
        "--output-dir",
        str(output),
        "--num-shards",
        str(world),
        "--shard-index",
        str(rank),
        "--device",
        "cpu",
    ]


def fake_condition(*args, **kwargs):
    return {"prompt_embeds": torch.zeros(1, 2, 8, dtype=torch.bfloat16), "text_token_tags": torch.ones(2, dtype=torch.long)}


def encode_shards(source, output, world=3):
    with (
        patch.object(builder, "model_identity", return_value=MODEL),
        patch.object(builder, "load_conditioner", return_value=(None, None, None)) as load,
        patch.object(builder, "encode_prompt", side_effect=fake_condition),
        patch.object(builder, "release_device_memory"),
    ):
        for rank in range(world):
            builder.main(encode_args(source, output, rank, world))
        return load.call_count


def merge(source, output, world=3):
    samples = builder.read_samples(source, "prompt", "t2av", 0, None)
    return coordinator.merge_shards(output, samples, builder.parse_args(encode_args(source, output, 0, world)), MODEL, world)


def test_global_shards_are_disjoint_and_publish_complete_cover(tmp_path):
    source, output = make_input(tmp_path, count=5), tmp_path / "cache"
    assert encode_shards(source, output) == 3
    assert not (output / "metadata.jsonl").exists()
    receipt = merge(source, output)
    assert receipt["shard_counts"] == [2, 2, 1]
    assert receipt["prompt_count"] == 5
    rows = builder._read_json_rows(output / "metadata.jsonl")
    assert [row["source_index"] for row in rows] == list(range(5))
    assert all(not row.get("video_latent_path") and not row.get("audio_latent_path") for row in rows)
    assert preflight.validate_cache(output, preflight.validate_prompts(source, expected=5), world_size=3) == receipt
    assert encode_shards(source, output) == 0  # Fingerprinted restart reuses encoded tensors.


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "caption", "fingerprint", "condition_path"])
def test_incomplete_or_stale_shards_never_publish(tmp_path, corruption):
    source, output = make_input(tmp_path, count=5), tmp_path / "cache"
    encode_shards(source, output)
    manifest = output / "metadata.shard-00000-of-00003.jsonl"
    rows = builder._read_json_rows(manifest)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["cache_fingerprint" if corruption == "fingerprint" else corruption] = "wrong"
    builder.atomic_write_jsonl(manifest, rows)
    with pytest.raises(ValueError):
        merge(source, output)
    assert not (output / "metadata.jsonl").exists()
    assert not (output / "prompt_cache.complete.json").exists()


def test_changed_prompt_input_refuses_existing_namespace(tmp_path):
    source, output = make_input(tmp_path, count=5), tmp_path / "cache"
    encode_shards(source, output)
    source.write_text(source.read_text().replace("Action 0", "Changed action"))
    with pytest.raises(RuntimeError, match="different input/model/geometry"):
        encode_shards(source, output)


def test_empty_global_shards_are_valid(tmp_path):
    source, output = make_input(tmp_path, count=2), tmp_path / "cache"
    assert encode_shards(source, output) == 2
    assert merge(source, output)["shard_counts"] == [1, 1, 0]


def test_sharded_overwrite_and_partial_input_are_rejected(tmp_path):
    source = make_input(tmp_path, count=5)
    for extra in (["--overwrite"], ["--replace-manifest"], ["--start-index", "1"], ["--max-samples", "1"]):
        with pytest.raises(SystemExit):
            builder.parse_args(encode_args(source, tmp_path / "cache") + extra)


def test_external_input_and_config_match_action_recipe(tmp_path):
    rows = preflight.validate_prompts(make_input(tmp_path, count=2000))
    assert len({row["id"] for row in rows}) == 2000
    config = yaml.safe_load(CONFIG.read_text())
    dmd = config["training"]["dmd"]
    assert (dmd["num_inference_steps"], dmd["fake_update_ratio"]) == (8, 5)
    assert (dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"]) == (12, 3)
    assert config["training"]["student"]["checkpoint_path"] == config["training"]["fake"]["checkpoint_path"]
    assert config["training"]["save_every_iters"] == 100
    assert config["training"]["save_total_limit"] == 2


def test_launcher_dry_run_has_two_distinct_rendezvous_and_no_execution(tmp_path):
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="23456",
        H3_GAN_REAL_CACHE="/real/metadata.jsonl",
        H3_PYTHON="/deliberately/missing/python",
        H3_GAN_OUTPUT=str(tmp_path / "output"),
        H3_GAN_PROMPT_CACHE=str(tmp_path / "cache"),
    )
    for key in ("LOCAL_KERNELS", "KERNELS_CACHE", "H3_KERNEL_SNAPSHOT", "CUDA_VISIBLE_DEVICES", "H3_GAN_PREPROCESS_PORT"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=env, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    assert len(lines) == 4
    assert "--rdzv_endpoint=127.0.0.1:23457" in lines[1]
    assert "--expected-world-size 32" in lines[1]
    assert "--check-cache" in lines[2]
    assert "--rdzv_endpoint=127.0.0.1:23456" in lines[3]
    assert "--nnodes=4 --nproc_per_node=8" in lines[1]
    assert "--nnodes=4 --nproc_per_node=8" in lines[3]
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "output").exists()


def test_launcher_rejects_missing_real_distribution():
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="23456")
    env.pop("H3_GAN_REAL_CACHE", None)
    result = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "H3_GAN_REAL_CACHE" in result.stderr


def test_full_cpu_preflight_resolves_environment_and_checks_checkpoint(tmp_path, monkeypatch, capsys):
    prompts = make_input(tmp_path, count=2000)
    source = tmp_path / "prior/checkpoint-000005700"
    (source / "dist_state").mkdir(parents=True)
    (source / "official_pdmd.complete").write_text("5700\n")
    (source / "dist_state/.metadata").write_bytes(b"DCP metadata placeholder")
    header, offset = {}, 0
    for module, rank in (("blocks.0.to_q", 128), ("blocks.0.adaln_proj.linear", 16), ("norm_out.linear", 16)):
        for kind, shape in (("A", [rank, 1]), ("B", [1, rank])):
            end = offset + rank * 4
            header[f"{module}.lora_{kind}.weight"] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
            offset = end
    encoded = json.dumps(header).encode()
    (source / "pytorch_lora_weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    model = tmp_path / "model"
    for name in ("transformer", "text_encoder", "vae"):
        (model / name).mkdir(parents=True)
        (model / name / "config.json").write_text("{}")
    real = tmp_path / "real"
    real.mkdir()
    row = {"task": "t2av", "caption": "A real action clip", "num_frames": 124, "target_height": 768, "target_width": 1344, "fps": 24}
    for key in ("condition_path", "video_latent_path", "audio_latent_path"):
        row[key] = key + ".pt"
        (real / row[key]).touch()
    (real / "metadata.jsonl").write_text(json.dumps(row) + "\n")
    for key, value in {
        "H3_MODEL_PATH": model,
        "H3_GAN_PROMPT_CACHE": tmp_path / "cache",
        "H3_GAN_REAL_CACHE": real,
        "H3_GAN_INIT_CHECKPOINT": source,
        "H3_GAN_OUTPUT": tmp_path / "new-stage",
        "H3_GAN_MAX_ITERS": "1100",
    }.items():
        monkeypatch.setenv(key, str(value))
    preflight.main([str(CONFIG), "--prompts", str(prompts)])
    output = capsys.readouterr().out
    assert "eligible_full_AV_rows=1" in output
    assert "optimizer_updates=1100" in output
    (source / "dist_state/.metadata").unlink()
    with pytest.raises(FileNotFoundError, match="dist_state"):
        preflight.main([str(CONFIG), "--prompts", str(prompts)])
    shutil.rmtree(source)
    runtime = preflight.load_module("_gan_resume_test_config", ROOT / "lightx2v_train/runtime/config.py")
    config = runtime.load_config(CONFIG)
    own = write_resume_checkpoint(Path(config["training"]["output_dir"]), config)
    preflight.main([str(CONFIG), "--prompts", str(prompts)])
    assert f"own-stage resume={own}" in capsys.readouterr().out
    # Successful own-stage continuation still checks its real distribution.
    (real / row["condition_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="condition_path"):
        preflight.main([str(CONFIG), "--prompts", str(prompts)])


def write_resume_checkpoint(output, config, iteration=100):
    checkpoint = output / f"checkpoint-{iteration:09d}"
    (checkpoint / "dist_state").mkdir(parents=True)
    (checkpoint / "dist_state/.metadata").touch()
    (checkpoint / "official_pdmd.complete").write_text(f"{iteration}\n")
    for rank in range(32):
        (checkpoint / f"official_pdmd.rank{rank:05d}.pt").touch()
    torch.save(
        {
            "iteration": iteration,
            "world_size": 32,
            "dmd2_gan_enabled": True,
            "dmd2_gan_config": {"options": config["training"]["dmd"]["gan"]},
            "dmd2_gan_head": {},
            "dmd2_gan_optimizer": {},
            "dmd2_gan_lr_scheduler": {},
        },
        checkpoint / "trainer_state.pt",
    )
    return checkpoint


@pytest.mark.parametrize("failure", ["complete", "rank", "dcp", "non_gan", "optimizer", "recipe"])
def test_own_stage_resume_rejects_latest_broken_save_instead_of_falling_back(tmp_path, failure):
    config = {"resume": {"auto_resume": True}, "training": {"output_dir": str(tmp_path), "dmd": {"gan": {"enabled": True}}}}
    previous = write_resume_checkpoint(tmp_path, config, 100)
    latest = write_resume_checkpoint(tmp_path, config, 200)
    assert preflight.own_stage_resume(config) == latest
    if failure in ("complete", "rank", "dcp"):
        relative = {"complete": "official_pdmd.complete", "rank": "official_pdmd.rank00031.pt", "dcp": "dist_state/.metadata"}[failure]
        (latest / relative).unlink()
    else:
        state = torch.load(latest / "trainer_state.pt", weights_only=False)
        if failure == "non_gan":
            state["dmd2_gan_enabled"] = False
        elif failure == "optimizer":
            state.pop("dmd2_gan_optimizer")
        else:
            state["dmd2_gan_config"]["options"] = {"enabled": True, "changed": True}
        torch.save(state, latest / "trainer_state.pt")
    with pytest.raises(ValueError, match="Latest own-stage"):
        preflight.own_stage_resume(config)
    assert previous.is_dir()
    config["resume"]["auto_resume"] = False
    assert preflight.own_stage_resume(config) is None


def test_rank_encoder_logs_capture_output_and_traceback(tmp_path, capsys):
    def fail(_):
        print("encoding sample 17")
        print("encoder diagnostic", file=sys.stderr)
        raise RuntimeError("encoding failed")

    path = tmp_path / "logs/rank-00017.test.log"
    with patch.object(builder, "main", side_effect=fail), pytest.raises(RuntimeError, match="encoding failed"):
        coordinator.run_encoder_logged([], path)
    contents = path.read_text()
    assert "encoding sample 17" in contents
    assert "encoder diagnostic" in contents
    assert "Traceback" in contents
    assert "RuntimeError: encoding failed" in contents
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_node_launch_log_is_optional_without_starting_python_or_gpus(tmp_path, enabled):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text('#!/bin/sh\necho fake-python-invocation "$@"\n')
    fake_python.chmod(0o755)
    output = tmp_path / "output"
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="23456",
        HOSTNAME="test-node",
        H3_GAN_REAL_CACHE="/real/metadata.jsonl",
        H3_PYTHON=str(fake_python),
        H3_GAN_OUTPUT=str(output),
        H3_GAN_LAUNCH_LOG=enabled,
    )
    for key in ("LOCAL_KERNELS", "KERNELS_CACHE", "H3_KERNEL_SNAPSHOT", "CUDA_VISIBLE_DEVICES", "H3_GAN_PREPROCESS_PORT"):
        env.pop(key, None)
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    assert result.stdout.count("fake-python-invocation") == 4
    logs = list((output / "logs").glob("launch.test-node.*.log"))
    if enabled == "1":
        assert len(logs) == 1
        assert logs[0].read_text().count("fake-python-invocation") == 4
    else:
        assert logs == []
        assert not output.exists()
