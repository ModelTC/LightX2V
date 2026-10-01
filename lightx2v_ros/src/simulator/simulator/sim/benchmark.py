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


def resolve_policy_config(cfg, root, backend):
    """Adapt benchmark inputs to LV's existing FastWAM/RealtimeWAM policy."""
    name = cfg.model.name
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
    for key, native_key in {
        "model.action_horizon": "action_chunk_size",
        "EVALUATION.num_inference_steps": "action_infer_steps",
        "EVALUATION.sigma_shift": "action_sample_shift",
        "EVALUATION.replan_steps": "actions_per_plan",
    }.items():
        if OmegaConf.select(cfg, key) is None:
            OmegaConf.update(cfg, key, profile[native_key])
        profile[native_key] = OmegaConf.select(cfg, key)
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


def load_config(benchmark, backend, argv):
    # Shared runtime defaults; benchmark-specific settings live with its backend.
    cfg = OmegaConf.merge(
        {
            "benchmark": benchmark,
            "backend": backend.__name__,
            "config_json": None,
            "ckpt": os.environ.get("CKPT_PATH"),
            "seed": 42,
            "model": {
                "name": "realtimewam",
                "model_path": os.environ.get("WAN_MODEL_PATH"),
                "action_horizon": None,
                "native_config": None,
                "t5_cpu_offload": False,
                "vae_cpu_offload": False,
                "factory": None,
                "options": {},
            },
            "EVALUATION": {
                "output_dir": os.environ.get("OUT"),
                "dataset_stats_path": os.environ.get("DATASET_STATS_PATH"),
                "num_inference_steps": None,
                "sigma_shift": None,
                "replan_steps": None,
                "max_steps": None,
                "resume": False,
            },
            "MULTIRUN": {"num_gpus": 8, "max_tasks_per_gpu": 2, "chunk_size": 20, "task_sample_ratio": 1.0},
            "dry_run": False,
        },
        backend.defaults(benchmark),
    )
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_struct(cfg.model.options, False)
    for argument in argv:
        key, value = argument.split("=", 1)
        if key == "model":
            key = "model.name"
        if key == "task":
            continue
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{key}={value}"]))

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
    return OmegaConf.to_container(cfg, resolve=True)


def run(cfg):
    backend = importlib.import_module(cfg["backend"])
    output = Path(cfg["EVALUATION"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    count = cfg["MULTIRUN"]["num_gpus"]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(count)))).split(",")
    slots = [device.strip() for device in devices[:count] for _ in range(cfg["MULTIRUN"]["max_tasks_per_gpu"])]
    tasks = backend.discover_tasks(cfg)
    ratio = cfg["MULTIRUN"]["task_sample_ratio"]
    if ratio < 1:
        tasks = random.Random(cfg["seed"]).sample(tasks, max(1, int(len(tasks) * ratio)))
    atomic_json(output / "manifest.json", {"config": cfg, "tasks": tasks})
    print(f"benchmark={cfg['benchmark']} tasks={len(tasks)} GPU slots={slots} output={output}", flush=True)
    if cfg["dry_run"]:
        print("Dry run: manifest generated; no model or rollout launched", flush=True)
        return
    remaining = []
    for task in tasks:
        path = result_path(output, task)
        if cfg["EVALUATION"]["resume"] and path.exists() and json.loads(path.read_text())["status"] == "complete":
            continue
        path.unlink(missing_ok=True)
        remaining.append(task)
    size = cfg["MULTIRUN"]["chunk_size"]
    queue = deque(remaining[i : i + size] for i in range(0, len(remaining), size))
    free_slots = deque(slots)
    active, failures = [], []
    trials = cfg["EVALUATION"][backend.TRIALS_FIELD]
    job_index = len(list((output / "jobs").glob("*.json")))

    def stop_workers(*_):
        raise KeyboardInterrupt

    old_term = signal.signal(signal.SIGTERM, stop_workers)
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
