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

from experiments.common.config import ROOT, load_config
from experiments.common.results import atomic_json, result_path, summarize


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
    required = [cfg["EVALUATION"]["robotwin_root" if cfg["benchmark"] == "robotwin" else "libero_root"]]
    if not cfg["model"]["factory"]:
        required += [cfg["base_ckpt"] or cfg["ckpt"], cfg["model"]["model_path"], cfg["EVALUATION"]["dataset_stats_path"]]
    if cfg["lora_path"]:
        required.append(cfg["lora_path"])
    for value in required:
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"Required checkpoint/model/stats/benchmark path missing: {value}")


def discover(cfg):
    if cfg["benchmark"] == "robotwin":
        from experiments.robotwin.adapter import discover_tasks
    else:
        from experiments.libero.adapter import discover_tasks
    tasks = discover_tasks(cfg)
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
    source_files = sorted((ROOT / "experiments").rglob("*.py"))
    source_files += sorted((ROOT / "lightx2v/models").rglob("*wam*.py"))
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
    trials = cfg["EVALUATION"]["eval_num_episodes"] if cfg["benchmark"] == "robotwin" else cfg["EVALUATION"]["num_trials"]
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
                if cfg["benchmark"] == "robotwin" and env.get("ROBOTWIN_NVIDIA_GL_ROOT"):
                    gl_root = Path(env["ROBOTWIN_NVIDIA_GL_ROOT"])
                    libgl = next((p for p in (gl_root / "libGL.so.1", gl_root / "libGL.so.1.7.0") if p.is_file()), None)
                    if libgl:
                        env["LD_PRELOAD"] = ":".join(filter(None, (str(libgl), env.get("LD_PRELOAD"))))
                    icd = gl_root / "nvidia_icd_abs.json"
                    if icd.is_file():
                        env["VK_ICD_FILENAMES"] = str(icd)
                if cfg["benchmark"] != "robotwin":
                    env.setdefault("MUJOCO_GL", "egl")
                    env.setdefault("PYOPENGL_PLATFORM", env["MUJOCO_GL"])
                    # Let robosuite select from CUDA_VISIBLE_DEVICES. Forcing
                    # EGL=0 breaks its physical-device assertion on GPUs 1+.
                    env.pop("MUJOCO_EGL_DEVICE_ID", None)
                proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "experiments/common/worker.py"), str(job)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
            "Usage: python experiments/<benchmark>/run_<benchmark>_manager.py model=realtimewam base_ckpt=/base.pt lora_path=/lora model.model_path=/Wan EVALUATION.dataset_stats_path=/stats.json EVALUATION.output_dir=/results [key=value ...]\nSee experiments/README.md and experiments/configs/eval.yaml. dry_run=true lists tasks without model inference."
        )
        return
    run(load_config(benchmark, sys.argv[1:]))
