#!/usr/bin/env python3
"""Supervise three fresh Wan experiments with an explicit precision profile.

Run the accompanying shell script. Environment: WAN_DMD_PYTHON, WAN_DMD_MODEL,
WAN_DMD_TEACHER, WAN_DMD_PROMPTS, WAN_DMD_RUN_ROOT; DMD_GPUS/PDMD_GPUS/HEAD_GPUS.
WAN_DMD_GRAD_ACCUM defaults to 16 and controls student/fake and head fitting.
WAN_DMD_PRECISION_PROFILE defaults to fp32_sdpa; bf16_fa3 uses FP32 master
weights and BF16 mixed-precision compute for all DiTs, requiring FA3. Only the
FP32 profile retries all groups with a BF16 teacher after actual CUDA OOM.
Old attempts always remain; the mixed-precision profile never changes backend.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

TRAIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG = TRAIN_ROOT / "configs/train/dmd/wan2_1_t2v_1_3b_teacher14b_full_acc16_fsdp2.yaml"
GROUPS = ("dmd", "pdmd", "head")
OOM_PATTERN = re.compile(r"CUDA out of memory|CUDA error: out of memory", re.I)
STOP_REQUESTED = False


def gradient_accumulation():
    value = os.getenv("WAN_DMD_GRAD_ACCUM", "16")
    if not re.fullmatch(r"[1-9]\d*", value):
        raise ValueError("WAN_DMD_GRAD_ACCUM must be a positive integer")
    return int(value)


def precision_profile(paths=None):
    value = (paths or {}).get("WAN_DMD_PRECISION_PROFILE", os.getenv("WAN_DMD_PRECISION_PROFILE", "fp32_sdpa"))
    if value not in {"fp32_sdpa", "bf16_fa3"}:
        raise ValueError("WAN_DMD_PRECISION_PROFILE must be fp32_sdpa or bf16_fa3")
    return value


def attempt_name(dtype, paths):
    return "mixed_bf16_fa3" if precision_profile(paths) == "bf16_fa3" else f"teacher_{dtype}"


def precision_description(dtype, paths):
    mixed = precision_profile(paths) == "bf16_fa3"
    return {
        "precision_profile": precision_profile(paths),
        "student_fake_master_dtype": "fp32",
        "teacher_master_dtype": "fp32" if mixed else dtype,
        "student_fake_compute_dtype": "bf16" if mixed else "fp32",
        "teacher_compute_dtype": "bf16" if mixed else dtype,
        "attention_backend": "flash_attention_3" if mixed else "sdpa",
        "allow_tf32": mixed,
        "attempt": attempt_name(dtype, paths),
    }


def resolved_config(group, dtype, output, paths):
    from lightx2v_train.runtime.config import load_config

    with patch.dict(os.environ, {**paths, "WAN_DMD_GROUP": group, "WAN_DMD_TEACHER_DTYPE": dtype, "WAN_DMD_OUTPUT": str(output)}):
        accumulation = gradient_accumulation()
        result = load_config(str(CONFIG))
    profile = precision_profile(paths)
    training, model = result["training"], result["model"]
    if profile == "bf16_fa3":
        if dtype != "bf16":
            raise ValueError("bf16_fa3 requires BF16 compute; it has no FP32 attempt or precision fallback")
        model["attention_backend"] = "flash_attention_3"
        for role in (model, model["fake"], model["teacher"]):
            role["running_dtype"] = "bf16"
            role["transformer_param_dtype"] = "fp32"
        result["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"] = "bf16"
        model["teacher"]["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"] = "bf16"
        training["allow_tf32"] = result["inference"]["allow_tf32"] = True
    assert training["max_train_iters"] == 10000
    assert training["gradient_accumulation_iters"] == accumulation
    assert training["dmd"]["residual_head"]["fit_steps"] == accumulation
    assert training["dmd"]["residual_head"]["fit_grad_accum_steps"] == accumulation
    assert training["student"]["train_type"] == training["fake"]["train_type"] == "full"
    assert training["teacher"]["guidance_scale"] == 5
    if profile == "fp32_sdpa":
        assert model["running_dtype"] == model["transformer_param_dtype"] == "fp32"
        assert model["attention_backend"] == "sdpa" and not training["allow_tf32"]
        assert model["teacher"]["running_dtype"] == model["teacher"]["transformer_param_dtype"] == dtype
    else:
        assert model["teacher"]["transformer_param_dtype"] == "fp32"
        assert model["attention_backend"] == "flash_attention_3" and training["allow_tf32"]
    assert training["dmd"]["update_order"] == ("student_first" if group == "pdmd" else "fake_first")
    assert training["dmd"]["fake_update_ratio"] == (1 if group == "pdmd" else 5)
    assert not result["resume"]["auto_resume"]
    return result


def check_fa3(torch):
    """Fail before model loading if the explicitly requested FA3 is unavailable."""
    from lightx2v_train.model_zoo.native.wan.modules import attention as wan_attention

    capabilities = [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]
    if not wan_attention.FLASH_ATTN_3_AVAILABLE or any(major != 9 for major, _ in capabilities):
        raise ValueError(f"bf16_fa3 requires installed flash_attn_interface and Hopper GPUs; capabilities={capabilities}")
    interface = wan_attention.flash_attn_interface
    if not callable(getattr(interface, "flash_attn_varlen_func", None)):
        raise ValueError("Installed FA3 interface lacks flash_attn_varlen_func; no FA2/SDPA fallback allowed")
    return {"module": str(getattr(interface, "__file__", "unknown")), "compute_capabilities": capabilities}


def check_model(path, text_encoder=False):
    required = ["config.json"]
    index = path / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        required.extend(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        required.append("diffusion_pytorch_model.safetensors")
    if text_encoder:
        required.extend(("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"))
    for name in required:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Missing model file: {path / name}")


def gpu_query(fields, kind="gpu"):
    output = subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], text=True, timeout=30)
    return [[item.strip() for item in row] for row in csv.reader(io.StringIO(output)) if row]


def preflight(root, gpu_groups, paths):
    selected = {index for group in gpu_groups.values() for index in group.split(",")}
    devices = {row[0]: row for row in gpu_query("index,uuid,name,memory.used,utilization.gpu")}
    if not selected <= devices.keys():
        raise ValueError(f"Missing selected GPUs: {selected - devices.keys()}")
    uuids = {devices[index][1] for index in selected}
    busy = [row for row in gpu_query("gpu_uuid,pid,process_name", "compute-apps") if row[0] in uuids]
    busy.extend(devices[index] for index in selected if float(devices[index][3]) > 1024 or float(devices[index][4]) > 5)
    if busy:
        raise ValueError(f"GPUs busy; no processes killed, no training started: {busy}")
    check_model(Path(paths["WAN_DMD_MODEL"]), text_encoder=True)
    check_model(Path(paths["WAN_DMD_TEACHER"]))
    prompt_digest, prompt_count = hashlib.sha256(), 0
    with Path(paths["WAN_DMD_PROMPTS"]).open("rb") as handle:
        for line in handle:
            prompt_digest.update(line)
            prompt_count += bool(line.strip())
    if prompt_count < 8:
        raise ValueError("At least eight nonempty prompts required")
    parent = root.parent
    while not parent.exists():
        parent = parent.parent
    free_gib = shutil.disk_usage(parent).free / 2**30
    # Three 37-GiB full optimizer/EMA checkpoint sets + preserved OOM attempt.
    if free_gib < 240:
        raise ValueError(f"Need 240 GiB free disk incl. OOM-attempt reserve; {parent}: {free_gib:.1f} GiB")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(selected, key=int))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
        raise ValueError("Training Python cannot see all six selected CUDA GPUs")
    mixed = precision_profile(paths) == "bf16_fa3"
    dtype = "bf16" if mixed else "fp32"
    fa3 = check_fa3(torch) if mixed else None
    configs = {group: resolved_config(group, dtype, root / attempt_name(dtype, paths) / group, paths) for group in GROUPS}
    if not mixed:
        for group in GROUPS:
            resolved_config(group, "bf16", root / "teacher_bf16" / group, paths)
    source_paths = [CONFIG, Path(__file__), TRAIN_ROOT / "train.py"]
    source_paths.extend(TRAIN_ROOT / "lightx2v_train/trainers/dmd" / name for name in ("runtime.py", "residual_head.py", "residual_head_training.py", "checkpoint.py"))
    source_paths.extend(TRAIN_ROOT / "lightx2v_train/model_zoo" / name for name in ("native/wan/modules/attention.py", "native/wan/modules/model.py", "wan/wan_t2v.py"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "torch": torch.__version__,
        "paths": paths,
        "gpu_groups": gpu_groups,
        "gpu_snapshot": [devices[i] for i in sorted(selected, key=int)],
        "prompt_count": prompt_count,
        "prompt_sha256": prompt_digest.hexdigest(),
        "free_disk_gib": free_gib,
        "configs": configs,
        "precision": precision_description(dtype, paths),
        "flash_attention_3": fa3,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }


def stop_jobs(jobs):
    for job in jobs.values():
        if job["process"].poll() is None:
            try:
                os.killpg(job["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    # Elastic's rank processes have their own sessions and its cleanup waits
    # 30 seconds before escalating. Leave time for that cleanup to finish.
    deadline = time.monotonic() + 90
    for job in jobs.values():
        process = job["process"]
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def log_has_oom(path):
    with path.open(errors="replace") as handle:
        return any(OOM_PATTERN.search(line) for line in handle)


def run_attempt(root, dtype, gpu_groups, paths):
    attempt = root / attempt_name(dtype, paths)
    attempt.mkdir()
    jobs = {}
    try:
        for group in GROUPS:
            if STOP_REQUESTED:
                raise KeyboardInterrupt
            output = attempt / group
            output.mkdir()
            config_path = output / "config.json"
            config_path.write_text(json.dumps(resolved_config(group, dtype, output, paths), indent=2) + "\n")
            log_path = output / "launch.log"
            with log_path.open("w") as log:
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_groups[group], "NVIDIA_TF32_OVERRIDE": "1" if precision_profile(paths) == "bf16_fa3" else "0"}
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "torch.distributed.run",
                        "--nnodes=1",
                        "--nproc_per_node=2",
                        "--rdzv_backend=c10d",
                        "--rdzv_endpoint=localhost:0",
                        "--max_restarts=0",
                        f"--rdzv_id=wan14b_{group}_{os.getpid()}_{dtype}",
                        "train.py",
                        "--config",
                        str(config_path),
                    ],
                    cwd=TRAIN_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            jobs[group] = {"process": process, "log": log_path}
            (output / "launcher.pid").write_text(f"{process.pid}\n")
            print(f"START group={group} profile={precision_profile(paths)} teacher_compute={dtype} pid={process.pid} GPUs={gpu_groups[group]} log={log_path}", flush=True)
        reported = set()
        while True:
            if STOP_REQUESTED:
                raise KeyboardInterrupt
            for group, job in jobs.items():
                code = job["process"].poll()
                if code is not None and group not in reported:
                    reported.add(group)
                    (attempt / group / "exit_code").write_text(f"{code}\n")
                    print(f"END group={group} teacher={dtype} rc={code}", flush=True)
                    if code:
                        return "oom" if log_has_oom(job["log"]) else "failed"
            if len(reported) == len(GROUPS):
                return "complete"
            time.sleep(2)
    finally:
        stop_jobs(jobs)
        for group, job in jobs.items():
            (attempt / group / "exit_code").write_text(f"{job['process'].returncode}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        accumulation = gradient_accumulation()
        profile = precision_profile()
    except ValueError as error:
        parser.error(str(error))
    paths = {
        "WAN_DMD_MODEL": os.getenv("WAN_DMD_MODEL", "/data/nvme0/gushiqiao/models/official_models/Wan2.1-T2V-1.3B"),
        "WAN_DMD_TEACHER": os.getenv("WAN_DMD_TEACHER", "/data/nvme1/wq/proj/sd/models/Wan2.1-T2V-14B"),
        "WAN_DMD_PROMPTS": os.getenv("WAN_DMD_PROMPTS", "/data/nvme4/gushiqiao/new/Causal-Forcing/prompts/vidprom_filtered_extended.txt"),
        "WAN_DMD_GRAD_ACCUM": str(accumulation),
        "WAN_DMD_PRECISION_PROFILE": profile,
    }
    dtype = "bf16" if profile == "bf16_fa3" else "fp32"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_bf16_fa3" if profile == "bf16_fa3" else ""
    root = Path(os.getenv("WAN_DMD_RUN_ROOT", str(TRAIN_ROOT.parent / "outputs" / f"wan21_14b_full_acc{accumulation}{suffix}_{stamp}_{os.getpid()}"))).resolve()
    gpu_groups = dict(zip(GROUPS, (os.getenv("DMD_GPUS", "0,1"), os.getenv("PDMD_GPUS", "2,3"), os.getenv("HEAD_GPUS", "5,6"))))
    selected = []
    for value in gpu_groups.values():
        if not re.fullmatch(r"(0|[1-9]\d*),(0|[1-9]\d*)", value):
            parser.error("Each GPU group must have exactly two numeric GPU indices")
        selected.extend(value.split(","))
    if len(set(selected)) != 6:
        parser.error("GPU groups must be disjoint")
    if root.exists():
        parser.error(f"Fresh output required, refusing existing run root: {root}")
    print(f"Run root: {root}\n10000 iterations; full student/fake; accumulation {accumulation}; teacher CFG5.", flush=True)
    print("Precision: " + json.dumps(precision_description(dtype, paths)), flush=True)
    print(f"GPU groups: {gpu_groups}; PDMD student→critic 1:1; DMD/head critic→student 5:1", flush=True)
    if args.dry_run:
        for group in GROUPS:
            resolved_config(group, dtype, root / attempt_name(dtype, paths) / group, paths)
        print("Dry run: config checked, no GPU checks/writes/training.")
        return 0
    manifest = preflight(root, gpu_groups, paths)
    if args.preflight:
        print(json.dumps(manifest, indent=2))
        print("Preflight passed; no output directory created or training started.")
        return 0
    root.mkdir(parents=True, exist_ok=False)
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "launcher.pid").write_text(f"{os.getpid()}\n")

    def interrupted(signum, frame):
        # Do not interrupt Popen registration or the cleanup of owned groups.
        # The loop observes this flag before the next launch/poll.
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    status = "running"
    (root / "status.json").write_text(json.dumps({"status": status, **precision_description(dtype, paths)}) + "\n")
    try:
        status = run_attempt(root, dtype, gpu_groups, paths)
        if status == "oom" and profile == "fp32_sdpa":
            print("Actual CUDA OOM observed. Preserving FP32 attempt; restarting ALL groups with ONLY teacher BF16.", flush=True)
            dtype = "bf16"
            (root / "status.json").write_text(json.dumps({"status": "running", **precision_description(dtype, paths), "fallback_reason": "CUDA OOM"}) + "\n")
            status = run_attempt(root, dtype, gpu_groups, paths)
    except KeyboardInterrupt:
        status = "interrupted"
        print("Stopped only this launcher's process groups.", flush=True)
    except Exception:
        status = "failed"
        raise
    finally:
        (root / "status.json").write_text(json.dumps({"status": status, **precision_description(dtype, paths), "updated_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
