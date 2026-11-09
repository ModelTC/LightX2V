"""CPU-only transition tests: no model creation and no torchrun/GPU execution."""

import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from h3_prompt_test_fixtures import make_input

ROOT = Path(__file__).resolve().parents[2]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


stages = module("_test_action_selfgen_stages", "run_minimax_h3_action_pdmd_selfgen_gan_stages.py")
helper = module("_test_action_selfgen_preflight", "minimax_h3_action_pdmd_selfgen_gan_preflight.py")
audit = module("_test_action_selfgen_audit", "run_minimax_h3_cm_pdmd_stages.py")
PHASE1 = ROOT / "configs/train/dmd/minimax_h3_t2av_action_pdmd8_shift12_3_r128_a128_adaln16_fsdp32.yaml"
PHASE2 = ROOT / "configs/train/dmd/minimax_h3_t2av_selfgen_pdmd8_gan_shift12_3_r128_a128_adaln16_fsdp32.yaml"
LAUNCHER = ROOT / "scripts/run_minimax_h3_action_pdmd_selfgen_gan_fsdp32_32gpu_acp.sh"


def weights(checkpoint, iteration):
    (checkpoint / "dist_state").mkdir(parents=True, exist_ok=True)
    (checkpoint / "dist_state/.metadata").write_bytes(b"DCP metadata placeholder")
    (checkpoint / "official_pdmd.complete").write_text(f"{iteration}\n")
    header, offset = {}, 0
    for target, rank in (("blocks.0.to_q", 128), ("blocks.0.adaln_proj.linear", 16), ("norm_out.linear", 16)):
        for kind, shape in (("A", [rank, 1]), ("B", [1, rank])):
            end = offset + rank * 4
            header[f"{target}.lora_{kind}.weight"] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
            offset = end
    encoded = json.dumps(header).encode()
    (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def save_checkpoint(config, iteration):
    checkpoint = Path(config["training"]["output_dir"]) / f"checkpoint-{iteration:09d}"
    weights(checkpoint, iteration)
    gan = bool(config["training"]["dmd"].get("gan", {}).get("enabled", False))
    state = {"iteration": iteration, "world_size": 32, "dmd2_gan_enabled": gan}
    if gan:
        state.update(dmd2_gan_config={"options": config["training"]["dmd"]["gan"]}, dmd2_gan_head={}, dmd2_gan_optimizer={}, dmd2_gan_lr_scheduler={})
    torch.save(state, checkpoint / "trainer_state.pt")
    for rank in range(32):
        torch.save(
            {
                "version": 1,
                "iteration": iteration,
                "rank": rank,
                "world_size": 32,
                "recipe": stages.official_recipe(config),
                "consumed_microbatches": iteration // 6 * 5 + min(iteration % 6, 5),
                "cached": [],
            },
            checkpoint / f"official_pdmd.rank{rank:05d}.pt",
        )
    return checkpoint


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    output = tmp_path / "staged"
    model = tmp_path / "model"
    for component in ("transformer", "text_encoder", "vae"):
        (model / component).mkdir(parents=True)
        (model / component / "config.json").write_text("{}")
    initial = tmp_path / "prior/checkpoint-000005700"
    weights(initial, 5700)
    prompts = make_input(tmp_path, count=2000)
    environment = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "23456",
        "H3_MODEL_PATH": str(model),
        "H3_GAN_INIT_CHECKPOINT": str(initial),
        "H3_STAGED_OUTPUT": str(output),
        "H3_ACTION_PDMD_OUTPUT": str(output / "phase1_action_pdmd"),
        "H3_SELFGEN_GAN_OUTPUT": str(output / "phase2_selfgen_gan"),
        "H3_ACTION_PDMD_VIEW": str(output / "data/action_prompts"),
        "H3_SELFGEN_VIEW": str(output / "data/self_generated_t2av"),
        "H3_GAN_MAX_ITERS": "1200",
        "H3_GAN_PROMPTS": str(prompts),
        "H3_GAN_PROMPT_CACHE": str(tmp_path / "prompt_cache"),
        "H3_SELFGEN_CACHE": str(tmp_path / "selfgen/metadata.jsonl"),
        "H3_RDZV_ID": "cpu-test",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in ("H3_GAN_PREPROCESS_PORT", "H3_GAN_PHASE2_PORT", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    phase1, phase2 = helper.load_config(PHASE1), helper.load_config(PHASE2)
    events = []
    actual_warm_start = helper.validate_warm_start

    def warm_start(config):
        events.append(("warm_start", config["training"]["student"]["checkpoint_path"]))
        return actual_warm_start(config)

    def run(arguments, **kwargs):
        assert kwargs["check"] is True
        if "prepare_minimax_h3_gan_prompts.py" in " ".join(arguments):
            events.append(("run", "prompts"))
        elif str(PHASE1) in arguments:
            events.append(("run", "phase1"))
            save_checkpoint(phase1, 600)
        else:
            assert str(PHASE2) in arguments
            events.append(("run", "phase2"))
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(helper, "validate_warm_start", warm_start)
    monkeypatch.setattr(helper, "prepare_action_view", lambda *args: events.append(("prepare", "action")))
    monkeypatch.setattr(helper, "prepare_selfgen_view", lambda *args: events.append(("prepare", "selfgen")))
    monkeypatch.setattr(helper, "validate_view", lambda *args, kind=None: events.append(("validate", kind)))
    monkeypatch.setattr(stages, "load_module", lambda name, path: helper if "preflight" in str(path) else audit)
    monkeypatch.setattr(stages.subprocess, "run", run)

    def execute(dry_run=False):
        return stages.execute(PHASE1, PHASE2, prompts=prompts, prompt_cache=environment["H3_GAN_PROMPT_CACHE"], real_cache=environment["H3_SELFGEN_CACHE"], dry_run=dry_run)

    return {"execute": execute, "phase1": phase1, "phase2": phase2, "events": events, "root": output, "initial": initial, "prompts": prompts, "env": environment}


def launched(workflow):
    return [event[1] for event in workflow["events"] if event[0] == "run"]


def test_fresh_run_orders_encoding_plain_pdmd_and_weights_only_gan(workflow):
    workflow["execute"]()
    assert launched(workflow) == ["prompts", "phase1", "phase2"]
    sources = [event[1] for event in workflow["events"] if event[0] == "warm_start"]
    assert sources == [str(workflow["initial"]), str(workflow["root"] / "phase1_action_pdmd/checkpoint-000000600")]
    receipt = json.loads((workflow["root"] / "stages/phase1_handoff.complete.json").read_text())
    assert receipt["student_updates"] == 100
    assert receipt["fake_updates"] == 500
    assert len(receipt["rank_state_sha256"]) == 32
    assert receipt["initialization"] == "student_and_fake_weights_only"
    assert (workflow["root"] / "stages/phase2.started.json").is_file()


def test_phase1_auto_resume_never_requires_original5700(workflow):
    save_checkpoint(workflow["phase1"], 300)
    shutil.rmtree(workflow["initial"])
    workflow["execute"]()
    assert launched(workflow) == ["prompts", "phase1", "phase2"]
    assert all(event[1] != str(workflow["initial"]) for event in workflow["events"] if event[0] == "warm_start")


def test_completed600_skips_prompt_encoding_and_phase1(workflow):
    save_checkpoint(workflow["phase1"], 600)
    shutil.rmtree(workflow["initial"])
    workflow["prompts"].unlink()
    workflow["execute"]()
    assert launched(workflow) == ["phase2"]
    assert ("prepare", "action") not in workflow["events"]


def test_phase2_auto_resume_never_reads_phase1_or_initial_or_prompt_sources(workflow):
    checkpoint = save_checkpoint(workflow["phase2"], 100)
    shutil.rmtree(workflow["initial"])
    workflow["prompts"].unlink()
    result = workflow["execute"]()
    assert result["resumed_phase2"] == str(checkpoint)
    assert workflow["events"] == [("validate", "selfgen"), ("run", "phase2")]
    assert not (workflow["root"] / "phase1_action_pdmd").exists()


def test_phase2_broken_latest_does_not_fallback_or_repeat_phase1(workflow):
    save_checkpoint(workflow["phase2"], 100)
    latest = save_checkpoint(workflow["phase2"], 200)
    (latest / "official_pdmd.rank00031.pt").unlink()
    with pytest.raises(ValueError, match="Latest own-stage checkpoint is incomplete"):
        workflow["execute"]()
    assert workflow["events"] == []


def test_failed_phase1_never_starts_gan_or_publishes_handoff(workflow, monkeypatch):
    def fail(arguments, **kwargs):
        if str(PHASE1) in arguments:
            raise subprocess.CalledProcessError(1, arguments)
        workflow["events"].append(("run", "prompts"))

    monkeypatch.setattr(stages.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        workflow["execute"]()
    assert launched(workflow) == ["prompts"]
    assert not (workflow["root"] / "stages/phase1_handoff.complete.json").exists()
    assert not (workflow["root"] / "stages/phase2.started.json").exists()


@pytest.mark.parametrize("corruption", ["cursor", "recipe", "rank", "cached", "gan", "complete", "adapter"])
def test_handoff_rejects_wrong_or_incomplete_phase1_state(workflow, corruption):
    checkpoint = save_checkpoint(workflow["phase1"], 600)
    if corruption in {"cursor", "recipe", "rank", "cached"}:
        path = checkpoint / "official_pdmd.rank00031.pt"
        state = torch.load(path, weights_only=False)
        if corruption == "cursor":
            state["consumed_microbatches"] = 600
        elif corruption == "recipe":
            state["recipe"]["training"]["dmd"]["fake_update_ratio"] = 10
        elif corruption == "rank":
            state["rank"] = 0
        else:
            state["cached"] = ["unfinished trajectory"]
        torch.save(state, path)
    elif corruption == "gan":
        path = checkpoint / "trainer_state.pt"
        state = torch.load(path, weights_only=False)
        state["dmd2_gan_enabled"] = True
        torch.save(state, path)
    elif corruption == "complete":
        (checkpoint / "official_pdmd.complete").unlink()
    else:
        (checkpoint / "pytorch_lora_weights.safetensors").write_bytes(b"broken")
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow["execute"]()
    assert launched(workflow) == []
    assert not (workflow["root"] / "stages/phase1_handoff.complete.json").exists()


def test_started_phase2_without_recovery_state_never_repeats_phase1(workflow):
    folder = workflow["root"] / "stages"
    folder.mkdir(parents=True)
    (folder / "phase2.started.json").write_text("{}")
    with pytest.raises(ValueError, match="Refusing to repeat phase 1"):
        workflow["execute"]()
    assert launched(workflow) == []


def test_dry_run_prints_three_distinct_32_gpu_commands_without_writes(workflow, capsys):
    shutil.rmtree(workflow["initial"])
    workflow["prompts"].unlink()
    workflow["execute"](dry_run=True)
    output = capsys.readouterr().out
    assert output.count("--nnodes=4 --nproc_per_node=8") == 3
    for port in (23456, 23457, 23458):
        assert f"--rdzv_endpoint=127.0.0.1:{port}" in output
    assert "--expected-samples 2000 --expected-world-size 32" in output
    assert not workflow["root"].exists()
    assert workflow["events"] == []


@pytest.mark.parametrize("extra", [{"H3_GAN_PHASE2_PORT": "23456"}, {"H3_GAN_PREPROCESS_PORT": "23458"}, {"MASTER_PORT": "65534"}, {"MASTER_PORT": "bad"}, {"CUDA_VISIBLE_DEVICES": "0,1"}])
def test_invalid_rendezvous_stops_before_any_work(workflow, monkeypatch, extra):
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        workflow["execute"](dry_run=True)
    assert not workflow["root"].exists()
    assert workflow["events"] == []


def test_shell_dry_run_expands_default_source_and_all_three_commands(tmp_path):
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT="23456", H3_PYTHON=sys.executable, H3_STAGED_OUTPUT=str(tmp_path / "output"))
    for key in (
        "LOCAL_KERNELS",
        "KERNELS_CACHE",
        "H3_KERNEL_SNAPSHOT",
        "CUDA_VISIBLE_DEVICES",
        "H3_GAN_PREPROCESS_PORT",
        "H3_GAN_PHASE2_PORT",
        "H3_ACTION_PDMD_OUTPUT",
        "H3_SELFGEN_GAN_OUTPUT",
        "H3_ACTION_PDMD_VIEW",
        "H3_SELFGEN_VIEW",
        "H3_SELFGEN_CACHE",
        "H3_GAN_REAL_CACHE",
    ):
        env.pop(key, None)
    result = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=env, check=True, capture_output=True, text=True)
    assert result.stdout.count("--nnodes=4 --nproc_per_node=8") == 3
    assert '"phase1_student_updates": 100' in result.stdout
    assert not (tmp_path / "output").exists()
