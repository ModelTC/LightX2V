import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from simulator.sim.bench.backends import get_benchmark_backend
from simulator.sim.bench.config import ROOT, load_config
from simulator.sim.bench.results import atomic_json, result_path, summarize


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
        required += [cfg["base_ckpt"] or cfg["ckpt"], cfg["model"]["model_path"], cfg["EVALUATION"]["dataset_stats_path"]]
    if cfg["lora_path"]:
        required.append(cfg["lora_path"])
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
    for path in (cfg["base_ckpt"] or cfg["ckpt"], cfg["lora_path"], cfg["EVALUATION"]["dataset_stats_path"]):
        if not path:
            continue
        root = Path(path)
        for file in [root] if root.is_file() else sorted(root.rglob("*")):
            if file.is_file():
                info = file.stat()
                assets.append([str(file), info.st_size, info.st_mtime_ns])
    source_files = sorted((ROOT / "scripts/bench/robotics").rglob("*.py"))
    source_files += sorted((ROOT / "lightx2v/models").rglob("*wam*.py"))
    source_files += sorted((ROOT / "lightx2v/models/networks/wan/weights/realtimewam").glob("*.py"))
    # Include shared adapters but not vendored simulator submodules/assets.
    simulator = ROOT / "lightx2v_ros/src/simulator/simulator"
    for directory in ("libero_node", "robotwin_node", "sim"):
        source_files += sorted((simulator / directory).glob("*.py"))
    for directory in ("sim/bench", "libero_node/bench", "robotwin_node/bench"):
        source_files += sorted((simulator / directory).rglob("*.py"))
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
                proc = subprocess.Popen([sys.executable, "-u", "-m", "simulator.sim.bench.worker", str(job)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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


def main(benchmark):
    if "--help" in sys.argv[1:]:
        print(
            "Usage: python scripts/bench/robotics/run_<benchmark>.py config_json=/path/to/model_profile.json base_ckpt=/base.pt lora_path=/lora model.model_path=/Wan EVALUATION.dataset_stats_path=/stats.json EVALUATION.output_dir=/results [key=value ...]\nDefaults: configs/bench/robotics/eval.yaml and the benchmark's bench/config.py. Model profiles: configs/fastwam and configs/realtimewam. dry_run=true lists tasks without model inference."
        )
        return
    run(load_config(benchmark, sys.argv[1:]))
