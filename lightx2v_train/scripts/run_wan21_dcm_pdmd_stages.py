#!/usr/bin/env python3
"""Run the two Wan experiments sequentially; never resume Turbo state as PDMD."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.dmd.official_pdmd_core import critic_updates_before
from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
from lightx2v_train.trainers.turbo.config import TurboConfig


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(dcm, pdmd):
    curriculum = TurboConfig.from_mapping(dcm)
    validate_official_pdmd_config(pdmd)
    require((curriculum.dcm_steps, curriculum.scm_steps, curriculum.joint_steps) == (100, 0, 0), "Stage 1 must contain exactly 100 dCM updates and no sCM/DMD")
    require(
        type(pdmd["training"]["max_train_iters"]) is int and pdmd["training"]["max_train_iters"] > 0 and pdmd["training"]["dmd"]["fake_update_ratio"] == 5,
        "Stage 2 must use a positive official optimizer-update budget and fake/student ratio 5:1",
    )
    require(dcm["training"]["output_dir"] != pdmd["training"]["output_dir"], "Stage outputs must differ")
    for config in (dcm, pdmd):
        require(config["model"]["name"] == "wan_t2v", "This experiment requires non-causal Wan T2V")
        require(config["distributed"]["fsdp2"]["enabled"] and config["distributed"]["fsdp2"]["size"] == 2, "This experiment requires FSDP on two GPUs")
        require(not config["training"].get("allow_tf32", True), "TF32 must remain disabled")
        require(not config["model"].get("t5_cpu", True), "T5 must remain on the GPU")
        for key in ("running_dtype", "transformer_param_dtype", "vae_dtype", "t5_dtype"):
            require(config["model"][key] == "fp32", f"model.{key} must be fp32")
        for role in ("fake", "teacher"):
            require(config["model"][role]["running_dtype"] == "fp32" and config["model"][role]["transformer_param_dtype"] == "fp32", f"{role} must be fp32")
        require(Path(config["model"]["pretrained_model_name_or_path"]).is_dir(), "Missing Wan model directory")
        for split in ("train", "val"):
            for path in config["data"][split]["data_path"]:
                require(Path(path).is_file(), f"Missing {split} manifest: {path}")
    require(dcm["data"]["train"]["name"] == "video_dataset", "dCM must encode raw videos online")
    require(pdmd["data"]["train"]["name"] == "prompt_dataset", "PDMD must generate its own training latents from prompts")
    for key in ("lora", "train_type"):
        require(dcm["training"]["student"][key] == pdmd["training"]["student"][key], "Student adapter architectures must match across stages")
    checkpoint = Path(dcm["training"]["output_dir"]) / "checkpoint-000000100"
    require(Path(pdmd["training"]["student"]["checkpoint_path"]) == checkpoint / "student_ema", "PDMD must load the dCM-100 EMA export explicitly")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    require(len(gpus) == 2 and all(value.strip() for value in gpus) and gpus[0] != gpus[1], "Select exactly two distinct visible GPUs")
    return checkpoint


def command(config_path):
    runner = Path(__file__).with_name("run_wan_turbo_fp32.py")
    return [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(runner), "--config", str(config_path)]


def budget_summary(pdmd):
    updates = pdmd["training"]["max_train_iters"]
    fake_updates = critic_updates_before(updates, pdmd["training"]["dmd"]["fake_update_ratio"])
    student_updates = updates - fake_updates
    return {
        "stages": ["100 dCM student updates", f"{updates} official PDMD optimizer updates = {student_updates} student + {fake_updates} fake"],
        "pdmd_optimizer_updates": updates,
        "pdmd_student_updates": student_updates,
        "pdmd_fake_updates": fake_updates,
        "pdmd_infer_every_role_updates": pdmd["inference"]["infer_every_iters"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcm-config", type=Path, required=True)
    parser.add_argument("--pdmd-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dcm, pdmd = load_config(str(args.dcm_config)), load_config(str(args.pdmd_config))
    checkpoint = preflight(dcm, pdmd)
    summary = {
        **budget_summary(pdmd),
        "fake_to_student": "5:1",
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        "checkpoint": str(checkpoint),
        "pdmd_sampler": "8-step Euler, shift 5, guidance 1; conditional/unconditional teacher CFG 5",
        "precision": "all components resident FP32; TF32 disabled",
        "dcm_config": str(args.dcm_config.resolve()),
        "pdmd_config": str(args.pdmd_config.resolve()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    for path in (args.dcm_config, args.pdmd_config):
        print(shlex.join(command(path)), flush=True)
    if args.dry_run:
        return
    # Stage 1 owns its strict Turbo resume cursor. A failure never starts PDMD.
    subprocess.run(command(args.dcm_config), check=True)
    # Read a revised stage-2 budget after a long-running dCM stage. Both the
    # subprocess and transition receipt must describe the effective recipe.
    require(load_config(str(args.dcm_config)) == dcm, "The dCM recipe changed while it was running")
    pdmd = load_config(str(args.pdmd_config))
    require(preflight(dcm, pdmd) == checkpoint, "The stage-transition checkpoint changed")
    summary.update(budget_summary(pdmd))
    marker = checkpoint / "turbo.complete.json"
    adapter = checkpoint / "student_ema/pytorch_lora_weights.safetensors"
    require(marker.is_file() and adapter.is_file(), "dCM completed without its strict completion receipt or EMA adapter")
    require(json.loads(marker.read_text())["iteration"] == 100, "dCM receipt has an unexpected iteration")
    transition = {
        **summary,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "student_adapter": str(adapter),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "warm_start": "student EMA weights only; fresh fake, optimizers, schedulers and cursors",
    }
    destination = checkpoint.parent.parent.parent / "stage_transition.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(transition, indent=2) + "\n")
    temporary.replace(destination)
    print("[stages] dCM complete; starting official PDMD, fake5 -> student1", flush=True)
    subprocess.run(command(args.pdmd_config), check=True)


if __name__ == "__main__":
    main()
