#!/usr/bin/env python3
"""Four-node ACP dCM -> official PDMD, preserving each stage's own resume state."""

import argparse
import errno
import hashlib
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent


def leaf(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def publish_once(path, value):
    """Publish identical shared metadata atomically from all four workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reservation = path.with_name(f".{path.name}.publish-lock")
    deadline = time.monotonic() + 30
    # All writers share this reservation, including hardlink-capable workers.
    # This prevents a fallback rename from replacing a racing link publication.
    while True:
        if path.is_file():
            require(json.loads(path.read_text()) == value, f"Existing staged recipe differs: {path}; use a fresh output directory")
            return
        try:
            reservation.mkdir()
            break
        except FileExistsError:
            remaining = deadline - time.monotonic()
            require(remaining > 0, f"Timed out waiting for staged metadata publication: {reservation}")
            time.sleep(min(0.05, remaining))
    temporary = None
    try:
        if path.is_file():
            require(json.loads(path.read_text()) == value, f"Existing staged recipe differs: {path}; use a fresh output directory")
            return
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        with os.fdopen(descriptor, "w") as handle:
            os.fchmod(handle.fileno(), 0o644)
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(json.loads(path.read_text()) == value, f"Existing staged recipe differs: {path}; use a fresh output directory")
        except OSError as error:
            if error.errno not in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.EXDEV}:
                raise
            # Some AFS/FUSE mounts support atomic mkdir/rename but not links.
            # The shared reservation makes this conditional rename safe.
            if path.is_file():
                require(json.loads(path.read_text()) == value, f"Existing staged recipe differs: {path}; use a fresh output directory")
            else:
                os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        reservation.rmdir()


def rendezvous():
    address = os.environ.get("MASTER_ADDR", "").strip()
    require(address, "ACP must provide a common MASTER_ADDR")
    try:
        cm_port = int(os.environ["MASTER_PORT"])
        pdmd_port = int(os.environ.get("H3_PDMD_MASTER_PORT", str(cm_port + 1)))
    except (KeyError, ValueError) as error:
        raise RuntimeError("MASTER_PORT and H3_PDMD_MASTER_PORT must be integer TCP ports") from error
    require(1 <= cm_port <= 65535 and 1 <= pdmd_port <= 65535 and cm_port != pdmd_port, "dCM and PDMD need two distinct TCP ports in 1..65535; set H3_PDMD_MASTER_PORT if necessary")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")]
    require(len(visible) == 8 and len(set(visible)) == 8 and all(visible), "Each ACP worker must expose eight distinct GPUs")
    run_id = os.environ.get("H3_RDZV_ID", "").strip()
    require(run_id, "Set the same H3_RDZV_ID on all four workers")
    host = f"[{address}]" if ":" in address and not address.startswith("[") else address
    return {"run_id": run_id, "cm": f"{host}:{cm_port}", "pdmd": f"{host}:{pdmd_port}"}


def command(config_path, phase, settings):
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--nnodes=4",
        "--nproc_per_node=8",
        f"--rdzv_id={settings['run_id']}_{phase}",
        "--rdzv_backend=c10d",
        f"--rdzv_endpoint={settings[phase]}",
        "--max_restarts=0",
        "train.py",
        "--config",
        str(Path(config_path).resolve()),
    ]


def expected_cm_recipe(config):
    from lightx2v_train.trainers.turbo.core import UPSTREAM_REVISION

    paths = leaf(SCRIPT_ROOT.parent / "lightx2v_train/data/minimax_h3_turbo_paths.py", "_h3_stages_cm_paths")
    data = config["data"]["train"]
    manifest, rows, _ = paths.load_cm_records(data["data_path"], path_mapping=data.get("path_mapping"), tasks=data["tasks"], native_geometry=True)
    digest = hashlib.sha256()
    for row in rows:
        record = {"type": "metadata", "row": row, "base_dir": str(manifest.parent)}
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode())
        digest.update(b"\n")
    return {
        "upstream": UPSTREAM_REVISION,
        "seed": config.get("seed", 42),
        "model": config["model"],
        "training": {key: value for key, value in config["training"].items() if key not in {"output_dir", "save_every_iters", "save_total_limit"}},
        "data": {key: value for key, value in data.items() if key not in {"num_workers", "pin_memory"}},
        "scheduler": config["scheduler"],
        "distributed": config.get("distributed", {}),
        "dataset": {"rows": len(rows), "sha256": digest.hexdigest()},
    }


def preset_metadata(config):
    # Only the real launch imports torch-backed modules. The preflight/dry-run
    # reads metadata paths and receipts without loading tensors or CUDA.
    from lightx2v_train.model_zoo.minimax_h3.training_adapter import AdapterOptions

    return AdapterOptions.from_mapping(config["model"]["training_adapter"]).metadata()


def verify_handoff(config, summary, preset):
    from lightx2v_train.trainers.dmd.student_warmstart import validate_student_adapter_preset

    iteration = summary["cm_total_steps"]
    checkpoint = Path(summary["cm_output"]) / f"checkpoint-{iteration:09d}"
    expected = {"version": 1, "iteration": iteration, "world_size": 32}
    marker = checkpoint / "turbo.complete.json"
    require(marker.is_file() and json.loads(marker.read_text()) == expected, f"dCM checkpoint is not complete on all 32 ranks: {checkpoint}")
    recipe = expected_cm_recipe(config)
    for rank in range(32):
        path = checkpoint / f"turbo.rank{rank:05d}.json"
        state = json.loads(path.read_text())
        require(
            all(state.get(key) == value for key, value in {**expected, "rank": rank, "recipe": recipe, "fake_updates": 0, "consumed_microbatches": iteration}.items()),
            f"dCM recipe, rank or cursor mismatch: {path}",
        )
    export = checkpoint / "student_ema"
    for name in ("pytorch_lora_weights.safetensors", "adapter_config.json", "training_adapter.json"):
        require((export / name).is_file(), f"Missing dCM EMA export component: {export / name}")
    validate_student_adapter_preset(export, expected_preset=preset, model_name=config["model"]["name"])
    source = json.loads((export / "training_adapter.json").read_text())
    require(
        source.get("enabled") is False and source.get("inference_requires_preset") is False and source.get("export_semantics") == "learned_lora_only",
        "The staged student must have no DeCFG dependency; old student-DeCFG exports require a fresh run",
    )
    return {
        "student_init": str(export),
        "student_uses_decfg": False,
        "inference_requires_preset": False,
        "student_adapter_sha256": digest_file(export / "pytorch_lora_weights.safetensors"),
        "adapter_config_sha256": digest_file(export / "adapter_config.json"),
        "training_adapter_sha256": digest_file(export / "training_adapter.json"),
    }


def execute(task, cm_path, pdmd_path, *, dry_run=False):
    helper = leaf(SCRIPT_ROOT / "minimax_h3_cm_pdmd_preflight.py", "_h3_stages_preflight")
    cm, pdmd = helper.load_config(cm_path), helper.load_config(pdmd_path)
    summary = helper.validate(task, cm, pdmd)
    settings = rendezvous()
    commands = {phase: command(path, phase, settings) for phase, path in (("cm", cm_path), ("pdmd", pdmd_path))}
    print(json.dumps({**summary, "rendezvous": settings}, indent=2), flush=True)
    for phase, args in commands.items():
        print(f"[h3-stages][{phase}] {shlex.join(args)}", flush=True)
    if dry_run:
        return summary
    preset = preset_metadata(cm)
    plan = {
        "schema_version": 1,
        "task": task,
        "cm": {key: value for key, value in cm.items() if key != "config_path"},
        "pdmd": {key: value for key, value in pdmd.items() if key != "config_path"},
        "data": summary,
        "preset": preset,
    }
    plan_id = digest_value(plan)
    root = Path(cm["training"]["output_dir"]).resolve().parent.parent
    require(Path(pdmd["training"]["output_dir"]).resolve().parent.parent == root, "Both stage outputs must share one H3_STAGED_OUTPUT root")
    publish_once(root / "stages/plan.json", {"plan_sha256": plan_id, **plan})
    receipt = root / "stages/cm_handoff.complete.json"
    if receipt.is_file():
        print("[h3-stages] verifying completed dCM EMA before resuming PDMD", flush=True)
    else:
        print(f"[h3-stages] worker={socket.gethostname()} starting dCM1500 on NEW full AV latents", flush=True)
        subprocess.run(commands["cm"], cwd=SCRIPT_ROOT.parent, check=True)
    # Never pass a changed recipe into a stage after spending hours training.
    require(helper.load_config(cm_path) == cm and helper.load_config(pdmd_path) == pdmd, "A stage config changed while dCM was running; relaunch with a consistent recipe")
    handoff = verify_handoff(cm, summary, preset)
    publish_once(receipt, {"schema_version": 1, "plan_sha256": plan_id, "task": task, **handoff})
    helper.validate(task, cm, pdmd, check_student_init=True)
    print("[h3-stages] dCM1500 EMA verified; starting official PDMD1500000, fake5 -> student1, NEW -> OLD -> repeat, DeCFG only on teacher/fake; student rollout/inference without DeCFG", flush=True)
    subprocess.run(commands["pdmd"], cwd=SCRIPT_ROOT.parent, check=True)
    print(f"[h3-stages] worker={socket.gethostname()} finished both stages", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("fl2av", "ref2av"))
    parser.add_argument("--cm-config", type=Path, required=True)
    parser.add_argument("--pdmd-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    execute(args.task, args.cm_config, args.pdmd_config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
