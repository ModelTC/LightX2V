#!/usr/bin/env python3
"""Start only official PDMD from a completed H3 CM EMA and premerged score bases."""

import argparse
import json
import math
import os
import shlex
import subprocess
from pathlib import Path

from run_minimax_h3_cm_pdmd_stages import digest_file, leaf, publish_once, require

TRAIN_ROOT = Path(__file__).resolve().parents[1]


def verify_cm_export(config, *, task):
    """Use the recorded CM recipe, without requiring the current CM YAML/cache."""
    export = Path(config["training"]["student"]["checkpoint_path"]).expanduser().resolve()
    require(export.name == "student_ema", "H3_STUDENT_INIT must point explicitly to checkpoint-000001500/student_ema")
    checkpoint = export.parent
    marker = {"version": 1, "iteration": 1500, "world_size": 32}
    require(json.loads((checkpoint / "turbo.complete.json").read_text()) == marker, f"CM1500 is not complete on all 32 ranks: {checkpoint}")
    recipe = None
    hashes = {}
    for rank in range(32):
        path = checkpoint / f"turbo.rank{rank:05d}.json"
        state = json.loads(path.read_text())
        require(all(state.get(key) == value for key, value in {**marker, "rank": rank, "fake_updates": 0, "consumed_microbatches": 1500}.items()), f"Invalid completed CM rank/cursor: {path}")
        if recipe is None:
            recipe = state["recipe"]
        require(state.get("recipe") == recipe, f"CM rank recipes differ: {path}")
        hashes[path.name] = digest_file(path)
    model_name = "minimax_h3_t2av" if task == "fl2av" else "minimax_h3_ref2av"
    require(recipe["model"]["name"] == config["model"]["name"] == model_name, "CM export task differs from requested PDMD task")
    require(
        Path(recipe["model"]["pretrained_model_name_or_path"]).resolve() == Path(config["model"]["pretrained_model_name_or_path"]).resolve(),
        "Student must use the original CM frozen base, not a premerged DeCFG base",
    )
    require(recipe["training"]["student"]["lora"] == config["training"]["student"]["lora"], "CM and PDMD student LoRA targets/ranks/alpha differ")
    for name in ("pytorch_lora_weights.safetensors", "adapter_config.json", "training_adapter.json"):
        path = export / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing CM EMA export component: {path}")
        hashes[name] = digest_file(path)
    sidecar = json.loads((export / "training_adapter.json").read_text())
    require(
        sidecar.get("enabled") is False
        and sidecar.get("schema_version") == 1
        and sidecar.get("export_role") == "student"
        and sidecar.get("model_name") == model_name
        and sidecar.get("inference_requires_preset") is False
        and sidecar.get("export_semantics") == "learned_lora_only",
        "CM EMA must be a learned student LoRA on the ORIGINAL base, without DeCFG dependency",
    )
    score_adapter = sidecar.get("score_training_adapter")
    require(isinstance(score_adapter, dict) and score_adapter.get("enabled") is True and score_adapter.get("roles") == ["fake", "teacher"], "CM export must record the score-only DeCFG identity")
    return {"student_init": str(export), "source_recipe": recipe, "source_hashes": hashes}, score_adapter


def validate(config, task):
    preflight = leaf(TRAIN_ROOT / "scripts/minimax_h3_cm_pdmd_preflight.py", "_h3_premerged_curriculum")
    merged = leaf(TRAIN_ROOT / "lightx2v_train/model_zoo/minimax_h3/premerged_decfg.py", "_h3_premerged_validation")
    handoff, adapter = verify_cm_export(config, task=task)
    model, training = config["model"], config["training"]
    component = "transformer" if task == "fl2av" else "transformer_ref"
    require(merged.validate_premerged_score_base(model["pretrained_model_name_or_path"], component_name=component) is None, "Student must use an original base without merged DeCFG")
    require(model["training_adapter"] == {"enabled": False}, "Online DeCFG must be disabled for this premerged recipe")
    score_records = {}
    for role in ("fake", "teacher"):
        override = model[role]
        require(override["training_adapter"] == {"enabled": False}, f"Disable online DeCFG for the premerged {role}; otherwise it is applied twice")
        record = merged.validate_premerged_score_base(override["pretrained_model_name_or_path"], component_name=component, model_name=model["name"], expected_adapter=adapter, required=True)
        require(Path(record["manifest"]["source_model_path"]).resolve() == Path(model["pretrained_model_name_or_path"]).resolve(), "Score base was merged from a different original H3 root")
        score_records[role] = {"manifest_path": str(record["manifest_path"]), "manifest_sha256": digest_file(record["manifest_path"]), "component": record["component"]}
    require(model["fake"]["pretrained_model_name_or_path"] == model["teacher"]["pretrained_model_name_or_path"], "Fake and teacher must initialize from the same premerged score base")
    require(
        model["fake"]["transformer_param_dtype"] == "fp32" and model["teacher"]["transformer_param_dtype"] == model["transformer_param_dtype"] == "bf16",
        "Preserve original FP32 fake masters and BF16 student/teacher bases",
    )
    dmd = training["dmd"]
    require(
        training["method"] == "dmd" and (dmd["official_pdmd"], dmd["num_inference_steps"], dmd["update_order"]) == (True, 8, "fake_first"),
        "Use official PDMD8 with fake-first updates",
    )
    require(type(dmd["fake_update_ratio"]) is int and dmd["fake_update_ratio"] > 0, "fake_update_ratio must be a positive integer")
    require(
        model["capabilities"]["distribution_matching"]["official_pdmd"] is True and model["capabilities"]["distribution_matching"]["projected_dmd"] is True, "Use the official projected DMD objective"
    )
    rollout_shifts = (dmd["official_rollout_video_shift"], dmd["official_rollout_audio_shift"])
    require(all(type(shift) in (int, float) and math.isfinite(shift) and shift > 0 for shift in rollout_shifts), "Rollout shifts must be finite and positive")
    metadata = model["capabilities"]["distribution_matching"]
    require((metadata["video_flow_shift"], metadata["audio_flow_shift"]) == rollout_shifts, "Inference metadata must match rollout shifts")
    require(training["max_train_iters"] == 1500000 and training["save_total_limit"] == 2, "PDMD uses 1500000 role updates and retains2 checkpoints")
    require(type(training["save_every_iters"]) is int and training["save_every_iters"] > 0, "save_every_iters must be a positive integer")
    require(
        training["student"]["train_type"] == "lora" and training["fake"]["train_type"] == "full" and training["student"]["ema"] == {"enabled": False},
        "Continue the CM LoRA as the live student and initialize a fresh full fake",
    )
    require(float(training["student"]["optimizer"]["learning_rate"]) == 2e-5 and float(training["fake"]["optimizer"]["learning_rate"]) == 4e-6, "Preserve student/fake learning rates2e-5/4e-6")
    distributed = config["distributed"]
    require(
        distributed["fsdp2"]["enabled"] and distributed["fsdp2"]["size"] == 32 and distributed["fsdp2"]["stream_load_pretrained"] and distributed["sequence_parallel"] == {"enabled": False, "size": 1},
        "This launcher requires streamed FSDP32, SP1",
    )
    data = config["data"]["train"]
    require(data["name"] == "minimax_h3_curriculum_cache_dataset" and data["kind"] == task and set(data["sources"]) == {"new", "old"}, "Use task-specific NEW -> OLD -> repeat data")
    new, _, new_paths = preflight._condition_source(data["sources"]["new"], task, "new")
    old, _, old_paths = preflight._condition_source(data["sources"]["old"], task, "old")
    require(not new_paths.intersection(old_paths), "NEW and OLD condition paths overlap")
    output = Path(training["output_dir"]).resolve()
    checkpoint = Path(handoff["student_init"]).parent
    require(
        output != checkpoint.parent and checkpoint.parent not in output.parents and output not in checkpoint.parents, "PDMD output must be separate from the source CM training/checkpoint directory"
    )
    return {"schema_version": 1, "task": task, **handoff, "score_bases": score_records, "new": new, "old": old, "student_uses_decfg": False, "online_decfg_merge": False}


def execute(task, config_path, *, dry_run=False):
    helper = leaf(TRAIN_ROOT / "lightx2v_train/runtime/config.py", "_h3_premerged_config")
    config = helper.load_config(str(config_path))
    plan = {"config": config, **validate(config, task)}
    address = os.environ.get("MASTER_ADDR", "").strip()
    port = int(os.environ.get("MASTER_PORT", "0"))
    run_id = os.environ.get("H3_RDZV_ID", "").strip()
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    require(address and 1 <= port <= 65535 and run_id, "ACP must provide MASTER_ADDR/MASTER_PORT; set identical H3_RDZV_ID on all four workers")
    require(len(visible) == len(set(visible)) == 8 and all(visible), "Expose eight distinct GPUs on each of the four workers")
    endpoint = f"[{address}]:{port}" if ":" in address and not address.startswith("[") else f"{address}:{port}"
    args = [
        os.environ.get("H3_PYTHON", os.sys.executable),
        "-u",
        "-m",
        "torch.distributed.run",
        "--nnodes=4",
        "--nproc_per_node=8",
        f"--rdzv_id={run_id}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={endpoint}",
        "--max_restarts=0",
        "train.py",
        "--config",
        str(Path(config_path).resolve()),
    ]
    print(
        json.dumps(
            {
                "task": task,
                "student_init": plan["student_init"],
                "score_base": config["model"]["teacher"]["pretrained_model_name_or_path"],
                "pdmd_output": config["training"]["output_dir"],
                "new_rows": plan["new"]["rows"],
                "old_rows": plan["old"]["rows"],
                "fake_update_ratio": config["training"]["dmd"]["fake_update_ratio"],
                "video_shift": config["training"]["dmd"]["official_rollout_video_shift"],
                "audio_shift": config["training"]["dmd"]["official_rollout_audio_shift"],
                "pdmd_total_iters": config["training"]["max_train_iters"],
                "pdmd_student_updates": config["training"]["max_train_iters"] // (config["training"]["dmd"]["fake_update_ratio"] + 1),
                "pdmd_fake_updates": config["training"]["max_train_iters"] - config["training"]["max_train_iters"] // (config["training"]["dmd"]["fake_update_ratio"] + 1),
                "save_every_role_updates": config["training"]["save_every_iters"],
                "save_total_limit": config["training"]["save_total_limit"],
                "student_uses_decfg": False,
                "online_decfg_merge": False,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"[h3-pdmd-from-cm] {shlex.join(args)}", flush=True)
    if not dry_run:
        # Bind the source adapter and premerged score identities across resumes.
        output = Path(config["training"]["output_dir"])
        receipt = output / "cm_premerged_handoff.json"
        require(receipt.is_file() or not any(output.glob("checkpoint-*")), "Existing PDMD checkpoints have no premerged CM handoff identity; select a new H3_PDMD_OUTPUT")
        publish_once(receipt, plan)
        subprocess.run(args, cwd=TRAIN_ROOT, check=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("fl2av", "ref2av"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    execute(args.task, args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
