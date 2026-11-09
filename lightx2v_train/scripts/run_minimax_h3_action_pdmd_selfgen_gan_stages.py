#!/usr/bin/env python3
"""Four-node ACP: encode action prompts -> 100 PDMD student updates -> self-generated GAN."""

import argparse
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
PHASE1_ITERS = 600
WORLD_SIZE = 32


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rendezvous():
    address = os.environ.get("MASTER_ADDR", "").strip()
    require(address, "ACP must provide the same MASTER_ADDR on all four workers")
    try:
        phase1 = int(os.environ["MASTER_PORT"])
        prompts = int(os.environ.get("H3_GAN_PREPROCESS_PORT", str(phase1 + 1)))
        phase2 = int(os.environ.get("H3_GAN_PHASE2_PORT", str(phase1 + 2)))
    except (KeyError, ValueError) as error:
        raise ValueError("MASTER_PORT, H3_GAN_PREPROCESS_PORT and H3_GAN_PHASE2_PORT must be integer TCP ports") from error
    ports = {"prompts": prompts, "phase1": phase1, "phase2": phase2}
    require(len(set(ports.values())) == 3 and all(1 <= value <= 65535 for value in ports.values()), "Prompt encoding, phase 1 and phase 2 require three distinct TCP ports in 1..65535")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")]
    require(len(visible) == 8 and len(set(visible)) == 8 and all(visible), "Each ACP worker must expose eight distinct GPUs")
    run_id = os.environ.get("H3_RDZV_ID", "h3_action_pdmd_selfgen_gan_from5800_32gpu_20261109").strip()
    require(run_id, "H3_RDZV_ID must be nonempty and identical on all four workers")
    host = f"[{address}]" if ":" in address and not address.startswith("[") else address
    return {"run_id": run_id, **{key: f"{host}:{port}" for key, port in ports.items()}}


def distributed_command(phase, settings, script, arguments):
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--nnodes=4",
        "--nproc_per_node=8",
        f"--rdzv_id={settings['run_id']}-{phase}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={settings[phase]}",
        "--max_restarts=0",
        str(script),
        *map(str, arguments),
    ]


def commands(phase1_path, phase2_path, phase1, prompts, prompt_cache, settings):
    return {
        "prompts": distributed_command(
            "prompts",
            settings,
            SCRIPT_ROOT.parent / "data_process/minimax_h3/prepare_minimax_h3_gan_prompts.py",
            [
                "--prompts",
                prompts,
                "--model-path",
                phase1["model"]["pretrained_model_name_or_path"],
                "--output-dir",
                prompt_cache,
                "--expected-samples",
                2000,
                "--expected-world-size",
                WORLD_SIZE,
            ],
        ),
        "phase1": distributed_command("phase1", settings, SCRIPT_ROOT.parent / "train.py", ["--config", Path(phase1_path).resolve()]),
        "phase2": distributed_command("phase2", settings, SCRIPT_ROOT.parent / "train.py", ["--config", Path(phase2_path).resolve()]),
    }


def validate_pair(helper, phase1, phase2):
    helper.validate_recipe(phase1, gan_enabled=False)
    helper.validate_recipe(phase2, gan_enabled=True)
    first, second = phase1["training"], phase2["training"]
    require(int(first["max_train_iters"]) == PHASE1_ITERS, "Phase 1 must run exactly 600 optimizer-role iterations = 500 fake + 100 student updates")
    require(first["gradient_accumulation_iters"] == second["gradient_accumulation_iters"] == 1, "Both phases require gradient_accumulation_iters=1")
    require(first["dmd"]["fake_update_ratio"] == second["dmd"]["fake_update_ratio"] == 5, "Both phases require five fake updates per student update")
    require(phase1.get("resume", {}).get("auto_resume") is True and phase2.get("resume", {}).get("auto_resume") is True, "Both phases must enable their own auto_resume")
    output1, output2 = (Path(config["training"]["output_dir"]).expanduser().resolve() for config in (phase1, phase2))
    require(output1 != output2 and output1.parent == output2.parent, "Phase outputs must be separate sibling directories under H3_STAGED_OUTPUT")
    handoff = output1 / f"checkpoint-{PHASE1_ITERS:09d}"
    require(all(Path(second[role]["checkpoint_path"]).expanduser().resolve() == handoff for role in ("student", "fake")), "Phase 2 must warm-start student and fake from phase 1 checkpoint-000000600")
    require(phase1["model"] == phase2["model"], "Both phases must use the same H3 model and adapter capabilities")
    for role in ("student", "fake"):
        expected = {key: value for key, value in first[role].items() if key != "checkpoint_path"}
        actual = {key: value for key, value in second[role].items() if key != "checkpoint_path"}
        require(expected == actual, f"Phase 2 must preserve the phase 1 {role} architecture and optimizer recipe")
    action_data = Path(phase1["data"]["train"]["data_path"]).expanduser().resolve()
    selfgen_data = Path(phase2["data"]["train"]["data_path"]).expanduser().resolve()
    require(action_data != selfgen_data, "The action prompt and self-generated training views must be separate")
    require(
        Path(second["dmd"]["gan"]["real_cache"]).expanduser().resolve() == selfgen_data / "t2av/landscape/metadata.jsonl",
        "Phase 2 conditioning and GAN positives must use the same filtered self-generated manifest",
    )
    return output1.parent


def official_recipe(config):
    """Mirror OfficialH3PdmdTraining.recipe for non-GAN H3 without importing CUDA code."""
    excluded = {"max_train_iters", "output_dir", "save_every_iters", "save_total_limit", "save_consolidated_weights"}
    return {
        "seed": int(config.get("seed", 42)),
        "model": config["model"],
        "training": {key: value for key, value in config["training"].items() if key not in excluded},
        "data": {key: value for key, value in config["data"]["train"].items() if key not in {"num_workers", "pin_memory"}},
        "scheduler": config["scheduler"],
        "distributed": config.get("distributed", {}),
    }


def verify_handoff(helper, audit, phase1, phase2):
    """Accept only the final complete non-GAN 32-rank phase-1 save, never a partial save."""
    import torch

    checkpoint = Path(phase1["training"]["output_dir"]).expanduser().resolve() / f"checkpoint-{PHASE1_ITERS:09d}"
    candidate = helper.own_stage_resume(phase1, gan_enabled=False)
    require(candidate is not None and Path(candidate).resolve() == checkpoint, f"Phase 1 did not produce its complete final checkpoint: {checkpoint}")
    recipe = official_recipe(phase1)
    hashes = {}
    for rank in range(WORLD_SIZE):
        path = checkpoint / f"official_pdmd.rank{rank:05d}.pt"
        state = torch.load(path, map_location="cpu", weights_only=False)
        expected = {"version": 1, "iteration": PHASE1_ITERS, "rank": rank, "world_size": WORLD_SIZE, "recipe": recipe, "consumed_microbatches": 500}
        require(isinstance(state, dict) and all(state.get(key) == value for key, value in expected.items()), f"Phase 1 handoff rank, recipe or data cursor mismatch: {path}")
        require(isinstance(state.get("cached"), list) and not state["cached"], f"Phase 1 handoff must be at a complete fake5/student1 cycle boundary: {path}")
        hashes[path.name] = audit.digest_file(path)
    source = helper.validate_warm_start(phase2)
    return {
        "checkpoint": str(checkpoint),
        "iteration": PHASE1_ITERS,
        "student_updates": 100,
        "fake_updates": 500,
        "world_size": WORLD_SIZE,
        "recipe_sha256": audit.digest_value(recipe),
        "rank_state_sha256": hashes,
        "student_weights_sha256": audit.digest_file(checkpoint / "pytorch_lora_weights.safetensors"),
        "dist_metadata_sha256": audit.digest_file(checkpoint / "dist_state/.metadata"),
        "initialization": "student_and_fake_weights_only",
        "source": source,
    }


def immutable_plan(phase1, phase2, prompts, prompt_cache, real_cache):
    configs = []
    for config in (phase1, phase2):
        copied = {key: value for key, value in config.items() if key != "config_path"}
        copied["training"] = {key: value for key, value in config["training"].items() if key != "max_train_iters"}
        configs.append(copied)
    return {
        "schema_version": 1,
        "phase1": configs[0],
        "phase2": configs[1],
        "phase1_optimizer_iterations": PHASE1_ITERS,
        "prompts": str(Path(prompts).expanduser().resolve()),
        "prompt_cache": str(Path(prompt_cache).expanduser().resolve()),
        "self_generated_source": str(Path(real_cache).expanduser().resolve()),
    }


def execute(phase1_path, phase2_path, *, prompts, prompt_cache, real_cache, dry_run=False):
    helper = load_module("_h3_action_selfgen_preflight", SCRIPT_ROOT / "minimax_h3_action_pdmd_selfgen_gan_preflight.py")
    audit = load_module("_h3_action_selfgen_audit", SCRIPT_ROOT / "run_minimax_h3_cm_pdmd_stages.py")
    phase1, phase2 = helper.load_config(phase1_path), helper.load_config(phase2_path)
    root = validate_pair(helper, phase1, phase2)
    settings = rendezvous()
    invocation = commands(phase1_path, phase2_path, phase1, prompts, prompt_cache, settings)
    summary = {
        "phase1_optimizer_iterations": PHASE1_ITERS,
        "phase1_student_updates": 100,
        "phase1_fake_updates": 500,
        "phase2_optimizer_iterations": int(phase2["training"]["max_train_iters"]),
        "phase2_data": phase2["data"]["train"]["data_path"],
        "rendezvous": settings,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        for phase, arguments in invocation.items():
            print(f"[h3-stages][{phase}] {shlex.join(arguments)}", flush=True)
        print("[h3-stages] Read-only command preview; complete phase-2 resume skips prompt encoding and phase 1. Input/checkpoint validation occurs on normal launch.", flush=True)
        return summary

    # Check phase 2 FIRST: its own resume has no dependency on the initial 5800
    # checkpoint, action prompts/cache, or retained phase-1 checkpoint weights.
    resume2 = helper.own_stage_resume(phase2, gan_enabled=True)
    helper.validate_model(phase2)
    plan = immutable_plan(phase1, phase2, prompts, prompt_cache, real_cache)
    plan_id = audit.digest_value(plan)
    stage_directory = root / "stages"
    audit.publish_once(stage_directory / "plan.json", {"plan_sha256": plan_id, **plan})
    if resume2 is not None:
        # The published view fixes this run's distribution. Later additions to
        # the raw source manifest must not change or invalidate a saved run.
        helper.validate_view(phase2["data"]["train"]["data_path"], kind="selfgen")
        print(f"[h3-stages] Phase 2 own-stage resume={resume2}; restoring student/fake/GAN optimizers and data cursors", flush=True)
        subprocess.run(invocation["phase2"], cwd=SCRIPT_ROOT.parent, check=True)
        return {**summary, "resumed_phase2": str(resume2)}

    helper.prepare_selfgen_view(real_cache, phase2["data"]["train"]["data_path"])
    receipt_path = stage_directory / "phase1_handoff.complete.json"
    started_path = stage_directory / "phase2.started.json"
    resume1 = helper.own_stage_resume(phase1, gan_enabled=False)
    final_checkpoint = Path(phase1["training"]["output_dir"]).expanduser().resolve() / f"checkpoint-{PHASE1_ITERS:09d}"
    complete1 = resume1 is not None and Path(resume1).resolve() == final_checkpoint
    if receipt_path.exists() or started_path.exists():
        require(
            complete1,
            "Phase 2 was already handed off but has no usable own-stage checkpoint, and phase 1 checkpoint-000000600 is unavailable. Refusing to repeat phase 1; restore either checkpoint to continue.",
        )
    if not complete1:
        if resume1 is not None:
            require(int(Path(resume1).name.split("-")[-1]) < PHASE1_ITERS, "Phase 1 resume exceeds its fixed 600-iteration boundary")
        else:
            helper.validate_warm_start(phase1)
        helper.validate_model(phase1)
        helper.validate_prompts(prompts)
        print("[h3-stages] Encoding/reusing 2,000 action conditions across 32 GPU shards", flush=True)
        subprocess.run(invocation["prompts"], cwd=SCRIPT_ROOT.parent, check=True)
        helper.prepare_action_view(prompt_cache, phase1["data"]["train"]["data_path"], prompts)
        print(
            f"[h3-stages] Phase 1 {'resume=' + str(resume1) if resume1 is not None else 'weights-only warm-start'}; plain official PDMD, fake5/student1, 600 role iterations = 100 student updates",
            flush=True,
        )
        subprocess.run(invocation["phase1"], cwd=SCRIPT_ROOT.parent, check=True)
    else:
        print("[h3-stages] Phase 1 checkpoint-000000600 is present; checking handoff without rerunning phase 1", flush=True)

    require(helper.load_config(phase1_path) == phase1 and helper.load_config(phase2_path) == phase2, "Stage configs changed while phase 1 ran; relaunch with a consistent recipe")
    handoff = verify_handoff(helper, audit, phase1, phase2)
    audit.publish_once(receipt_path, {"schema_version": 1, "plan_sha256": plan_id, **handoff})
    helper.validate_view(phase2["data"]["train"]["data_path"], kind="selfgen")
    audit.publish_once(started_path, {"schema_version": 1, "plan_sha256": plan_id, "phase1_checkpoint": str(final_checkpoint)})
    print("[h3-stages] Phase 2: self-generated T2AV conditions and positive AV pairs; PDMD + GAN, fake5/student1; student/fake weights only, fresh optimizers, GAN head and cursors", flush=True)
    subprocess.run(invocation["phase2"], cwd=SCRIPT_ROOT.parent, check=True)
    print(f"[h3-stages] worker={socket.gethostname()} finished both phases", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-config", type=Path, required=True)
    parser.add_argument("--phase2-config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-cache", type=Path, required=True)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    execute(args.phase1_config, args.phase2_config, prompts=args.prompts, prompt_cache=args.prompt_cache, real_cache=args.real_cache, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        raise SystemExit(f"H3 action PDMD -> self-generated GAN launch failed: {error}") from error
