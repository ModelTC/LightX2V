"""Select calibration requests and invoke the existing native trace collector."""

import hashlib
import json
import os
import random
import subprocess
import sys
import uuid
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch.distributed as dist

from .collect_calibration import TASKS, build_requests, resolve_inference_config, task_group

SELECTION_FILE = "calibration_selection.json"
COLLECTION_SCRIPT = Path(__file__).with_name("collect_calibration.py")
PIPELINE_COMPONENTS = ("transformer", "text_encoder", "processor", "vae", "scheduler")


def select_indices(manifest, counts, seed):
    pools = {task: [] for task in TASKS}
    with Path(manifest).open() as file:
        records = [json.loads(line) for line in file if line.strip()]
    for index, record in enumerate(records):
        pools[task_group(record)].append(index)
    generator = random.Random(seed)
    selected = []
    for task in TASKS:
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
        # The collector resets LOCAL_RANK to zero. Restrict visibility first so
        # that zero refers to this worker's GPU, not the launcher's first GPU.
        env["CUDA_VISIBLE_DEVICES"] = device
    for name in list(env):
        if name.startswith("TORCHELASTIC_") or name in ("MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "NODE_RANK"):
            env.pop(name)
    env.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1")
    # Native inference accepts BF16/FP16 only. Reconstruction precision is
    # independent of the BF16 target trajectories used for calibration.
    env["DTYPE"] = "BF16"
    return env


def collection_signature(config, inference=None):
    """Record sampling inputs without reading multi-GB weight tensors."""
    if not config.calibration_manifest or config.calibration_counts is None:
        raise ValueError("Collection/reuse requires calibration_manifest and calibration_counts")
    model = Path(config.target_model).resolve(strict=True)
    if inference is None:
        inference = resolve_inference_config(json.loads(Path(config.calibration_inference_config).read_text()), model)
    if inference.get("enable_cfg", False) or inference.get("parallel", False) or inference.get("lora_configs"):
        raise ValueError("Target collection requires CFG disabled, parallel=false and no additional LoRA")
    files = set()
    for name in PIPELINE_COMPONENTS:
        component = (model / name).resolve(strict=True)
        if not component.is_dir():
            raise ValueError(f"Missing pipeline component directory: {component}")
        files.update(path.resolve() for path in component.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(component).parts))
    if (model / "model_index.json").is_file():
        files.add((model / "model_index.json").resolve())
    inventory = []
    for path in sorted(files):
        stat = path.stat()
        item = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix == ".json":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append(item)
    manifest = Path(config.calibration_manifest).resolve(strict=True)
    return {
        "format_version": 2,
        "target_model": str(model),
        "model_files": inventory,
        "inference": inference,
        "collection_dtype": "bf16",
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "seed": config.seed,
        "counts": config.calibration_counts,
    }


def _collection_plan(config, world, dry_run, indices=None):
    if not config.calibration_manifest or config.calibration_counts is None:
        raise ValueError("Collection requires calibration_manifest and calibration_counts in the reconstruction config")
    manifest = Path(config.calibration_manifest).resolve()
    output = Path(config.calibration).resolve()
    indices = select_indices(manifest, config.calibration_counts, config.seed) if indices is None else indices
    # Validate image paths/canvas sizes without loading a model or writing files.
    jobs = build_requests(manifest, output, per_task=1, indices=indices, seed=config.seed)
    inference = Path(config.calibration_inference_config).resolve()
    inference_config = json.loads(inference.read_text())
    if inference_config.get("enable_cfg", False) or inference_config.get("parallel", False):
        raise ValueError("Trace collection requires enable_cfg=false and parallel=false")
    model = Path(config.target_model).resolve()
    for component in PIPELINE_COMPONENTS:
        if not (model / component).is_dir():
            raise ValueError(f"Collection needs a full target inference pipeline; missing {model / component}")
    inference_config = resolve_inference_config(inference_config, model)
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
        "provenance": collection_signature(config, inference_config),
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
        "sigmas": inference_config.get("sigmas"),
        "do_time_shift": inference_config.get("do_time_shift", True),
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
        str(COLLECTION_SCRIPT),
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
        "--indices",
        *map(str, indices),
    ]
    subprocess.run(command, check=True, env=env)
    missing = [name for name in files if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Collection did not produce all requested traces: {missing[:5]}")


def collect_calibration(config, dry_run=False, indices=None):
    group = CollectionGroup()
    try:
        # Only rank zero selects/validates requests. Broadcasting the plan keeps
        # filenames and seeds independent of worker scheduling and GPU count.
        selection = group.on_main(lambda: _collection_plan(config, group.world, dry_run, indices))
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


def link_file(destination, source):
    source = source.resolve(strict=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise ValueError(f"Existing pipeline entry points elsewhere: {destination}")
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_pipeline(config):
    """Use a full pipeline or attach exported transformer files to base assets."""
    target = Path(config.target_model).resolve(strict=True)
    if target.is_dir() and (target / "transformer").is_dir():
        for name in PIPELINE_COMPONENTS:
            if not (target / name).is_dir():
                raise ValueError(f"Missing target pipeline component: {target / name}")
        return str(target), str(Path(config.calibration_inference_config).resolve()), None

    root = target.parent if target.is_file() else target
    if target.is_file():
        if target.suffix != ".safetensors":
            raise ValueError("Collection requires safetensors weights; export the training checkpoint first")
        weights = {"diffusion_pytorch_model.safetensors": target}
    else:
        indices = sorted(root.glob("*.safetensors.index.json"))
        if len(indices) > 1:
            raise ValueError(f"Multiple weight indices; specify a single transformer directory: {root}")
        if indices:
            names = set(json.loads(indices[0].read_text())["weight_map"].values())
            if not names or any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
                raise ValueError("Transformer index must reference safetensors shards in the same directory")
            weights = {name: root / name for name in sorted(names)}
            weights[indices[0].name] = indices[0]
        else:
            weights = {path.name: path for path in sorted(root.glob("*.safetensors"))}
        if not weights:
            raise ValueError(f"No safetensors weights found in transformer directory: {root}")
    for path in weights.values():
        path.resolve(strict=True)

    base = Path(config.source_model).resolve(strict=True)
    inference = json.loads(Path(config.calibration_inference_config).read_text())
    release_path = root / "inference_config.json"
    if release_path.is_file():
        release = json.loads(release_path.read_text())
        if release.get("true_cfg_scale", 1.0) != 1.0:
            raise ValueError("Checkpoint collection currently requires CFG=1")
        if "sample_sigmas" in release:
            if not release["sample_sigmas"]:
                raise ValueError("Checkpoint sample_sigmas must not be empty")
            inference.update(infer_steps=len(release["sample_sigmas"]), sigmas=release["sample_sigmas"], do_time_shift=False)
        if "scheduler" in release:
            inference["scheduler_config"] = release["scheduler"]
    if inference.get("enable_cfg", False) or inference.get("parallel", False) or inference.get("lora_configs"):
        raise ValueError("Target collection requires CFG disabled, parallel=false and no additional LoRA")
    inference.update(enable_cfg=False, parallel=False)
    pipeline = Path(config.work_dir).resolve() / "collection_pipeline"
    transformer = pipeline / "transformer"
    transformer.mkdir(parents=True, exist_ok=True)
    existing = {path.name for pattern in ("*.safetensors", "*.safetensors.index.json") for path in transformer.glob(pattern)}
    if existing - weights.keys():
        raise ValueError(f"Collection pipeline contains stale weight files; use a separate work_dir: {transformer}")
    for name, path in weights.items():
        link_file(transformer / name, path)
    model_config = Path(config.model_config) if config.model_config else root / "config.json"
    if not config.model_config and not model_config.is_file():
        model_config = base / "transformer/config.json"
    link_file(transformer / "config.json", model_config)
    for component in PIPELINE_COMPONENTS[1:]:
        link_file(pipeline / component, base / component)
    link_file(pipeline / "model_index.json", base / "model_index.json")
    inference_file = pipeline / "collection_config.json"
    inference_file.write_text(json.dumps(inference, indent=2) + "\n")

    # Older single-file collections saved a separate, narrower signature.
    legacy = None
    if target.is_file() and config.calibration_manifest:
        stat = target.stat()
        legacy = {
            "checkpoint": str(target),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "base_model": str(base),
            "inference": inference,
            "seed": config.seed,
            "counts": config.calibration_counts,
            "manifest_sha256": hashlib.sha256(Path(config.calibration_manifest).read_bytes()).hexdigest(),
        }
    return str(pipeline), str(inference_file), legacy


def reuse_collection(config, signature, dry_run=False, legacy_signature=None):
    root = Path(config.calibration)
    selection_file = root / SELECTION_FILE
    if not selection_file.is_file():
        if root.is_dir() and any(root.rglob("*.pt")):
            raise ValueError("Existing calibration lacks a selection/provenance record; recollect in a separate calibration directory")
        return None
    selection = json.loads(selection_file.read_text())
    if not selection.get("complete"):
        return None
    saved = selection.get("provenance")
    expected = signature
    if saved is None:
        provenance = next((root / name for name in ("checkpoint_collection.json", "rl_collection.json") if (root / name).is_file()), None)
        if legacy_signature is None or provenance is None:
            raise ValueError("Existing calibration lacks verifiable provenance; recollect in a separate calibration directory")
        saved = json.loads(provenance.read_text())
        expected = legacy_signature
    if {k: v for k, v in saved.items() if k != "counts"} != {k: v for k, v in expected.items() if k != "counts"}:
        raise ValueError("Existing calibration belongs to a different checkpoint/config; use a separate calibration directory")
    counts = config.calibration_counts
    if (
        selection["counts"] != saved["counts"]
        or selection["seed"] != config.seed
        or Path(selection["target_model"]).resolve() != Path(config.target_model).resolve()
        or Path(selection["manifest"]).resolve() != Path(config.calibration_manifest).resolve()
    ):
        raise ValueError("Collection selection and provenance disagree")
    files = calibration_files(replace(config, calibration_counts=saved["counts"]))
    if len(files) != len(selection["indices"]) or len(set(selection["indices"])) != len(files) or len(files) != sum(saved["counts"].values()):
        raise ValueError("Collection inventory is inconsistent")
    if saved["counts"] == counts:
        print(f"Reusing target trajectories: {root}", flush=True)
        return str(root)
    records = [json.loads(line) for line in Path(config.calibration_manifest).read_text().splitlines() if line.strip()]
    if any(type(index) is not int or index < 0 or index >= len(records) for index in selection["indices"]):
        raise ValueError("Collection selection contains invalid manifest indices")
    pools = {task: [] for task in counts}
    for index, path in zip(selection["indices"], files, strict=True):
        pools[task_group(records[index])].append((index, path))
    rng = random.Random(config.seed)
    chosen = []
    for task, count in counts.items():
        if len(pools[task]) < count:
            return None
        chosen.extend(rng.sample(pools[task], count))
    chosen.sort(key=lambda item: item[0])
    suffix = "_".join(str(counts[k]) for k in ("t2i", "i2i_single", "i2i_multi"))
    subset = root.with_name(f"{root.name}_subset_{suffix}_seed{config.seed}")
    print(f"Reuse {len(chosen)} calibration traces from {root}; subset inventory: {subset}", flush=True)
    if not dry_run:
        subset.mkdir(parents=True, exist_ok=True)
        names = []
        for index, path in chosen:
            name = f"{index:08d}.pt"
            link_file(subset / name, path)
            names.append(name)
        _write_selection(subset, {**selection, "counts": counts, "indices": [i for i, _ in chosen], "files": names, "derived_from": str(root.resolve()), "provenance": signature})
    return str(subset)


def prepare_calibration_workflow(config, dry_run=False, reuse_only=False):
    """Prepare and validate trajectories for full pipelines and exported weights."""
    group = CollectionGroup()
    try:
        pipeline, inference, legacy = group.on_main(lambda: prepare_pipeline(config))
        collection = replace(config, target_model=pipeline, calibration_inference_config=inference)
        signature = group.on_main(lambda: collection_signature(collection))
        reused = group.on_main(lambda: reuse_collection(collection, signature, dry_run, legacy))
        if reused:
            return replace(config, calibration=reused)
        if reuse_only:
            raise ValueError("No complete matching calibration with enough samples is available")
        collect_calibration(collection, dry_run=dry_run)
        return config
    finally:
        group.close()
