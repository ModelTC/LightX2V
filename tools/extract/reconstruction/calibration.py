"""Select calibration requests and invoke the existing native trace collector."""

import importlib.util
import json
import os
import random
import subprocess
import sys
import uuid
from datetime import timedelta
from functools import cache
from pathlib import Path

import torch.distributed as dist

SELECTION_FILE = "calibration_selection.json"
PREVIEW_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/qwen_image_21/run_validation_preview.py"


@cache
def preview_module():
    # `scripts` is not a package in this repository and can collide with an
    # installed package of that name. Load this specific local script instead.
    spec = importlib.util.spec_from_file_location("qwen_image21_calibration_preview", PREVIEW_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_indices(manifest, counts, seed):
    preview = preview_module()
    pools = {task: [] for task in preview.TASKS}
    with Path(manifest).open() as file:
        records = [json.loads(line) for line in file if line.strip()]
    for index, record in enumerate(records):
        pools[preview.task_group(record)].append(index)
    generator = random.Random(seed)
    selected = []
    for task in preview.TASKS:
        if len(pools[task]) < counts[task]:
            raise ValueError(f"Calibration manifest has {len(pools[task])} {task} requests, but {counts[task]} were requested")
        selected.extend(generator.sample(pools[task], counts[task]))
    # Preserve manifest indices for trace filenames and reproducible seeds.
    return sorted(selected)


class CollectionGroup:
    """CPU-only coordination; each inference subprocess owns just one GPU."""

    def __init__(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.owns_group = int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized()
        if self.owns_group:
            # Workers may spend hours collecting before joining the final
            # collective; no accelerator allocation is needed for coordination.
            dist.init_process_group("gloo", timeout=timedelta(hours=24))
        if dist.is_initialized() and dist.get_backend() != "gloo":
            raise ValueError("Calibration collection requires a separate Gloo process group; launch it separately from reconstruction")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1

    def on_main(self, function):
        if self.world == 1:
            return function()
        message = [None]
        if self.rank == 0:
            try:
                message[0] = {"result": function()}
            except Exception as exc:  # noqa: BLE001 - propagate any coordinator failure to every worker
                message[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(message, src=0)
        if "error" in message[0]:
            raise RuntimeError(message[0]["error"])
        return message[0]["result"]

    def check_errors(self, error):
        errors = [error]
        if self.world > 1:
            errors = [None] * self.world
            dist.all_gather_object(errors, error)
        if any(errors):
            detail = "; ".join(f"rank {rank}: {value}" for rank, value in enumerate(errors) if value)
            raise RuntimeError(f"Calibration collection failed; selection remains incomplete. {detail}")

    def close(self):
        if self.owns_group:
            dist.destroy_process_group()


def worker_environment(config, local_rank, distributed):
    env = dict(os.environ)
    if distributed:
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            device = str(local_rank)
        else:
            devices = [value.strip() for value in visible.split(",") if value.strip()]
            if local_rank >= len(devices) or devices[local_rank] == "-1":
                raise ValueError(f"LOCAL_RANK={local_rank} has no device in CUDA_VISIBLE_DEVICES={visible!r}")
            device = devices[local_rank]
        # The preview resets LOCAL_RANK to zero. Restrict visibility first so
        # that zero refers to this worker's GPU, not the launcher's first GPU.
        env["CUDA_VISIBLE_DEVICES"] = device
    for name in list(env):
        if name.startswith("TORCHELASTIC_") or name in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK"):
            env.pop(name)
    env.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1")
    env["DTYPE"] = "BF16" if config.compute_dtype in ("bf16", "bfloat16") else "FP32"
    return env


def _collection_plan(config, world, dry_run):
    if not config.calibration_manifest or config.calibration_counts is None:
        raise ValueError("Collection requires calibration_manifest and calibration_counts in the reconstruction config")
    manifest = Path(config.calibration_manifest).resolve()
    output = Path(config.calibration).resolve()
    indices = select_indices(manifest, config.calibration_counts, config.seed)
    # Validate image paths/canvas sizes without loading a model or writing files.
    jobs = preview_module().build_requests(manifest, output, per_task=1, indices=indices, seed=config.seed)
    inference = Path(config.calibration_inference_config).resolve()
    inference_config = json.loads(inference.read_text())
    if inference_config.get("enable_cfg", False) or inference_config.get("parallel", False):
        raise ValueError("Trace collection requires enable_cfg=false and parallel=false")
    model = Path(config.target_model).resolve()
    for component in ("transformer", "text_encoder", "processor", "vae", "scheduler"):
        if not (model / component).is_dir():
            raise ValueError(f"Collection needs a full target inference pipeline; missing {model / component}")
    selection = {
        "format_version": 1,
        "complete": False,
        "manifest": str(manifest),
        "target_model": str(model),
        "counts": config.calibration_counts,
        "seed": config.seed,
        "indices": indices,
        "files": [Path(job["request"]["save_result_path"]).with_suffix(".pt").name for job in jobs],
        "world_size": world,
        "inference_config": str(inference),
    }
    if world > 1:
        run_dir = f"collection_{uuid.uuid4().hex}"
        selection["files"] = [f"{run_dir}/rank_{position % world:05d}/{name}" for position, name in enumerate(selection["files"])]
    summary = {
        "target_model": str(model),
        "manifest": str(manifest),
        "counts": config.calibration_counts,
        "requests": len(jobs),
        "inference_steps": inference_config["infer_steps"],
        "reconstruction_samples": len(jobs) * inference_config["infer_steps"],
        "output": str(output),
        "dry_run": dry_run,
        "world_size": world,
        "requests_per_rank": [len(indices[rank::world]) for rank in range(world)],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return selection


def _write_selection(output, selection):
    output.mkdir(parents=True, exist_ok=True)
    path = output / SELECTION_FILE
    temporary = output / f".{SELECTION_FILE}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(selection, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_shard(config, selection, group):
    indices = selection["indices"][group.rank :: group.world]
    files = selection["files"][group.rank :: group.world]
    if not indices:
        return
    output = Path(config.calibration).resolve()
    shard_output = (output / files[0]).parent
    env = worker_environment(config, group.local_rank, group.world > 1)
    print(f"Collection rank {group.rank}/{group.world}: {len(indices)} requests, CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'default')}, output={shard_output}", flush=True)
    command = [
        sys.executable,
        str(PREVIEW_SCRIPT),
        "--model-path",
        selection["target_model"],
        "--config-json",
        selection["inference_config"],
        "--manifest",
        selection["manifest"],
        "--output-dir",
        str(shard_output),
        "--seed",
        str(config.seed),
        "--dump-trace",
        "--indices",
        *map(str, indices),
    ]
    subprocess.run(command, check=True, env=env)
    missing = [name for name in files if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Collection did not produce all requested traces: {missing[:5]}")


def collect_calibration(config, dry_run=False):
    group = CollectionGroup()
    try:
        # Only rank zero selects/validates requests. Broadcasting the plan keeps
        # filenames and seeds independent of worker scheduling and GPU count.
        selection = group.on_main(lambda: _collection_plan(config, group.world, dry_run))
        if dry_run:
            return selection
        output = Path(config.calibration).resolve()
        group.on_main(lambda: _write_selection(output, selection))
        error = None
        try:
            _collect_shard(config, selection, group)
        except Exception as exc:  # noqa: BLE001 - keep failures collective and the inventory incomplete
            error = f"{type(exc).__name__}: {exc}"
        group.check_errors(error)

        def finish():
            missing = [name for name in selection["files"] if not (output / name).is_file()]
            if missing:
                raise RuntimeError(f"Missing traces on the coordinator: {missing[:5]}; all ranks must share the calibration directory")
            completed = {**selection, "complete": True}
            _write_selection(output, completed)
            return completed

        return group.on_main(finish)
    finally:
        group.close()


def calibration_files(config):
    root = Path(config.calibration)
    selection_path = root / SELECTION_FILE
    if root.is_dir() and selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        if not selection.get("complete"):
            raise ValueError("Calibration collection is incomplete; rerun --collect-calibration before reconstruction")
        if config.calibration_counts is not None and selection["counts"] != config.calibration_counts:
            raise ValueError("Collected request counts differ from calibration_counts; collect data with the current config")
        names = selection["files"]
        if not names or len(set(names)) != len(names) or any(Path(name).is_absolute() or ".." in Path(name).parts or not name.endswith(".pt") for name in names):
            raise ValueError("Invalid calibration selection file list")
        # Use the exact selected set, even if older traces exist alongside it.
        files = [root / name for name in names]
    else:
        files = sorted(root.rglob("*.pt")) if root.is_dir() else [root]
    if not files or any(not path.is_file() for path in files):
        raise ValueError("No calibration .pt files found, or selected traces are missing")
    return files
