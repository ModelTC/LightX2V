"""Configure and schedule isolated LIBERO / LIBERO-plus / RoboTwin workers."""

import hashlib
import importlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[5]


def get_benchmark_backend(name):
    if name not in {"libero", "libero_plus", "robotwin"}:
        raise ValueError(f"Unknown benchmark: {name}")
    node = "libero" if name.startswith("libero") else "robotwin"
    return importlib.import_module(f"simulator.{node}_node.benchmark")


def resolve_policy_config(cfg, root, backend):
    """Adapt benchmark inputs to LV's existing FastWAM/RealtimeWAM policy."""
    name = cfg.model.name
    if name not in {"fastwam", "realtimewam"}:
        raise ValueError("model must be fastwam or realtimewam")
    if not cfg.config_json:
        path = root / "configs" / name / f"{backend.POLICY_PROFILE}_i2va.json"
        if not path.exists():
            path = root / "configs/fastwam" / f"{backend.POLICY_PROFILE}_i2va.json"
        cfg.config_json = str(path)
        profile = json.loads(path.read_text())
        if name == "realtimewam":
            profile.update(action_infer_steps=1, triton_ops=True, layer_norm_type="Triton", cuda_graph=True)
    else:
        profile = json.loads(Path(cfg.config_json).read_text())
    if profile["policy_profile"] != backend.POLICY_PROFILE:
        raise ValueError(f"Profile {profile['policy_profile']} does not match benchmark {cfg.benchmark}")
    if profile.get("backbone", "fastwam") != "fastwam" or profile.get("lora_path"):
        raise ValueError("Use LV's dense/merged FastWAM checkpoint; PR #1562's FasterWAM/LoRA loader is not included")
    for key, native_key in {
        "model.action_horizon": "action_chunk_size",
        "EVALUATION.num_inference_steps": "action_infer_steps",
        "EVALUATION.sigma_shift": "action_sample_shift",
        "EVALUATION.replan_steps": "actions_per_plan",
    }.items():
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, profile[native_key])
        profile[native_key] = OmegaConf.select(cfg, key)
    if cfg.EVALUATION.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    # Full chunks come from the policy; settling and action execution belong to the evaluator.
    profile.pop("num_steps_wait", None)
    profile.update(
        model_cls=name,
        task="i2va",
        device="cuda:0",
        warmup=False,
        model_path=cfg.model.model_path,
        adapter_model_path=cfg.ckpt,
        dataset_stats_path=cfg.EVALUATION.dataset_stats_path,
        seed=cfg.seed,
        t5_cpu_offload=cfg.model.t5_cpu_offload,
        vae_cpu_offload=cfg.model.vae_cpu_offload,
    )
    cfg.model.native_config = profile


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def result_path(output, task):
    digest = hashlib.sha256(task["key"].encode()).hexdigest()[:16]
    return Path(output) / "tasks" / f"{digest}.json"


def score(episodes):
    successes = sum(bool(e["success"]) for e in episodes)
    return {"successes": successes, "trials": len(episodes), "success_rate": successes / len(episodes) if episodes else None}


def summarize(output, tasks, trials):
    episodes, groups, errors = [], {"categories": {}, "phases": {}, "suites": {}}, []
    completed_tasks = 0
    for task in tasks:
        path = result_path(output, task)
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows = result["episodes"]
        episodes.extend(rows)
        completed_tasks += result["status"] == "complete"
        if result.get("error"):
            errors.append({"task": task["key"], "error": result["error"]})
        for name, key in (("categories", "category"), ("phases", "phase"), ("suites", "suite")):
            if task.get(key):
                groups[name].setdefault(task[key], []).extend(rows)
    return {
        "aggregation": "micro: total successes / completed trials; not mean of category percentages",
        "complete": completed_tasks == len(tasks) and not errors,
        "planned_tasks": len(tasks),
        "completed_tasks": completed_tasks,
        "planned_trials": len(tasks) * trials,
        "completed_trials": len(episodes),
        "overall": score(episodes),
        **{name: {k: score(v) for k, v in group.items()} for name, group in groups.items()},
        "errors": errors,
    }


def load_config(benchmark, argv):
    backend = get_benchmark_backend(benchmark)
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "configs/bench/robotics/eval.yaml"), backend.defaults(benchmark))
    cfg.benchmark = benchmark
    cfg.ckpt = os.environ.get("CKPT_PATH")
    cfg.model.model_path = os.environ.get("WAN_MODEL_PATH")
    cfg.EVALUATION.dataset_stats_path = os.environ.get("DATASET_STATS_PATH")
    cfg.EVALUATION.output_dir = os.environ.get("OUT")
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.model.options, False)
    for argument in argv:
        if "=" not in argument or argument.startswith("--"):
            raise ValueError(f"Expected key=value override (not --key=value), got {argument!r}")
        key, value = argument.split("=", 1)
        if key == "model":
            key = "model.name"
        if key == "task":
            if value != backend.LEGACY_TASK:
                raise ValueError(f"Unsupported legacy task alias: {value}")
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={value}"]))

    backend.validate(cfg)
    # Resolve paths before any simulator changes the worker's working directory.
    for key in (
        "config_json",
        "ckpt",
        "model.model_path",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        *backend.PATH_FIELDS,
    ):
        value = OmegaConf.select(cfg, key)
        if value:
            OmegaConf.update(cfg, key, str(Path(os.path.expandvars(value)).expanduser().resolve()))
    resolve_policy_config(cfg, ROOT, backend)
    for field in ("num_gpus", "max_tasks_per_gpu", "chunk_size"):
        if cfg.MULTIRUN[field] < 1:
            raise ValueError(f"MULTIRUN.{field} must be positive")
    if cfg.EVALUATION.replan_steps < 1 or cfg.EVALUATION.replan_steps > cfg.model.action_horizon:
        raise ValueError("replan_steps must be positive and not exceed model.action_horizon")
    if cfg.EVALUATION.max_steps is not None and cfg.EVALUATION.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not 0 < cfg.MULTIRUN.task_sample_ratio <= 1:
        raise ValueError("task_sample_ratio must be in (0,1]")
    return OmegaConf.to_container(cfg, resolve=True)


def gpu_slots(cfg):
    count = cfg["MULTIRUN"]["num_gpus"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(count)]
    devices = [d.strip() for d in devices if d.strip()]
    if len(devices) < count or len(set(devices)) != len(devices) or any(d == "-1" for d in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain enough distinct GPUs")
    return [device for device in devices[:count] for _ in range(cfg["MULTIRUN"]["max_tasks_per_gpu"])]


def validate_paths(cfg):
    if not cfg["EVALUATION"]["output_dir"]:
        raise ValueError("Set OUT or EVALUATION.output_dir")
    required = get_benchmark_backend(cfg["benchmark"]).required_paths(cfg)
    if not cfg["model"]["factory"]:
        required += [cfg["ckpt"], cfg["model"]["model_path"], cfg["EVALUATION"]["dataset_stats_path"]]
    for value in required:
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"Required checkpoint/model/stats/benchmark path missing: {value}")


def discover(cfg):
    tasks = get_benchmark_backend(cfg["benchmark"]).discover_tasks(cfg)
    ratio = cfg["MULTIRUN"]["task_sample_ratio"]
    if ratio < 1:
        tasks = random.Random(cfg["seed"]).sample(tasks, max(1, int(len(tasks) * ratio)))
    if not tasks or len({t["key"] for t in tasks}) != len(tasks):
        raise ValueError("Task selection must be nonempty and unique")
    return tasks


def run(cfg):
    import fcntl

    validate_paths(cfg)
    slots = gpu_slots(cfg)
    output = Path(cfg["EVALUATION"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".manager.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f"Another manager owns {output}")
    try:
        return _run_locked(cfg, slots, output)
    finally:
        lock.close()


def _run_locked(cfg, slots, output):
    backend = get_benchmark_backend(cfg["benchmark"])
    tasks = discover(cfg)
    effective = json.loads(json.dumps(cfg))
    effective["EVALUATION"].pop("resume")
    effective.pop("dry_run")
    assets = []
    for path in (cfg["ckpt"], cfg["EVALUATION"]["dataset_stats_path"]):
        if not path:
            continue
        root = Path(path)
        for file in [root] if root.is_file() else sorted(root.rglob("*")):
            if file.is_file():
                info = file.stat()
                assets.append([str(file), info.st_size, info.st_mtime_ns])
    source_files = sorted((ROOT / "scripts/bench/robotics").rglob("*.py"))
    source_files += sorted((ROOT / "lightx2v").rglob("*.py"))
    # Include shared adapters but not vendored simulator submodules/assets.
    simulator = ROOT / "lightx2v_ros/src/simulator/simulator"
    for directory in ("libero_node", "robotwin_node", "sim"):
        source_files += sorted((simulator / directory).glob("*.py"))
    source_files += sorted((ROOT / "lightx2v_ros/src/common/common").glob("*.py"))
    source_digest = hashlib.sha256()
    for file in source_files:
        source_digest.update(str(file.relative_to(ROOT)).encode())
        source_digest.update(file.read_bytes())
    source_hash = source_digest.hexdigest()
    fingerprint = hashlib.sha256(json.dumps([effective, assets, tasks, source_hash], sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if not cfg["EVALUATION"]["resume"] or old["fingerprint"] != fingerprint:
            raise ValueError("Output already exists: use a new OUT, or resume=true with identical config/assets")
    atomic_json(manifest, {"fingerprint": fingerprint, "source_hash": source_hash, "config": cfg, "assets": assets, "tasks": tasks})
    print(f"benchmark={cfg['benchmark']} tasks={len(tasks)} GPU slots={slots} output={output}", flush=True)
    if cfg["dry_run"]:
        print("Dry run: manifest generated; no model or rollout launched", flush=True)
        return
    remaining = []
    for task in tasks:
        path = result_path(output, task)
        if path.exists() and json.loads(path.read_text())["status"] == "complete":
            continue
        remaining.append(task)
    size = cfg["MULTIRUN"]["chunk_size"]
    queue = deque(remaining[i : i + size] for i in range(0, len(remaining), size))
    free_slots = deque(slots)
    active, failures = [], []
    trials = backend.num_trials(cfg)
    job_index = 0
    old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while queue or active:
            while queue and free_slots and not failures:
                group, gpu = queue.popleft(), free_slots.popleft()
                job = output / "jobs" / f"{job_index:05d}.json"
                atomic_json(job, {"config": cfg, "tasks": group})
                log_path = job.with_suffix(".log")
                log = log_path.open("a")
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"), OPENBLAS_NUM_THREADS="1")
                backend.worker_environment(env)
                # Workers import the same repository-local simulator and policy code.
                python_paths = [str(ROOT), str(ROOT / "lightx2v_ros/src/simulator"), str(ROOT / "lightx2v_ros/src/common")]
                env["PYTHONPATH"] = os.pathsep.join(python_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
                proc = subprocess.Popen([sys.executable, "-u", "-m", "simulator.sim.evaluator", str(job)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((proc, gpu, log, log_path))
                print(f"launch pid={proc.pid} gpu={gpu} tasks={[t['key'] for t in group]} log={log_path}", flush=True)
                job_index += 1
            for entry in active[:]:
                proc, gpu, log, log_path = entry
                code = proc.poll()
                if code is not None:
                    active.remove(entry)
                    log.close()
                    free_slots.append(gpu)
                    if code:
                        failures.append({"returncode": code, "log": str(log_path)})
            summary = summarize(output, tasks, trials)
            summary["worker_failures"] = failures
            summary["complete"] = summary["complete"] and not failures
            summary["max_steps_override"] = cfg["EVALUATION"]["max_steps"]
            atomic_json(output / "summary.json", summary)
            if failures:
                raise RuntimeError(f"Worker failed; inspect original traceback: {failures}")
            if active:
                time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        for proc, _, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc, _, log, _ in active:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        summary = summarize(output, tasks, trials)
        summary["worker_failures"] = failures
        summary["complete"] = summary["complete"] and not failures
        summary["max_steps_override"] = cfg["EVALUATION"]["max_steps"]
        atomic_json(output / "summary.json", summary)
