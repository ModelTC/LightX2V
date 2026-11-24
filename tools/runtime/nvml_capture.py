"""Optional bounded NVML acquisition for the existing gpu_watch capture/check CLI.

The supplied policy module retains PID ancestry, task windows, idle checks and
coverage rules. This module only supplies the same required device inventory.
"""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BINDING_VERSION = "12.550.52"


def trace_stage(stage):
    destination = os.environ.get("LIGHTX2V_NVML_QUERY_STAGE")
    if destination:
        Path(destination).write_text(stage)


def text_string(value):
    return value.decode() if isinstance(value, bytes) else value


def sample_nvml(devices, nvml=None):
    if nvml is None:
        trace_stage("binding metadata")
        if importlib.metadata.version("nvidia-ml-py") != BINDING_VERSION:
            raise ValueError("Install the pinned nvidia-ml-py binding")
        trace_stage("import pynvml")
        import pynvml as nvml
    trace_stage("nvmlInit")
    nvml.nvmlInit()
    try:
        result = []
        for device in devices:
            trace_stage("nvmlDeviceGetHandleByUUID")
            handle = nvml.nvmlDeviceGetHandleByUUID(device)
            trace_stage("nvmlDeviceGetUUID")
            actual = text_string(nvml.nvmlDeviceGetUUID(handle))
            if actual != device:
                raise ValueError("NVML device UUID mismatch")
            # Missing/unsupported inventory APIs fail closed. A100 whole-device
            # operation is validated; MIG and MPS are explicitly unsupported.
            trace_stage("nvmlDeviceGetMigMode")
            mig = nvml.nvmlDeviceGetMigMode(handle)
            if len(mig) != 2 or any(mig):
                raise ValueError("MIG unsupported or unavailable")
            trace_stage("nvmlDeviceGetMPSComputeRunningProcesses")
            mps = nvml.nvmlDeviceGetMPSComputeRunningProcesses(handle)
            if not isinstance(mps, list) or mps:
                raise ValueError("MPS unsupported or unavailable")
            processes = {}
            trace_stage("nvmlDeviceGetComputeRunningProcesses")
            compute = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            trace_stage("nvmlDeviceGetGraphicsRunningProcesses")
            graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            for kind, entries in (("C", compute), ("G", graphics)):
                for entry in entries:
                    pid = int(entry.pid)
                    if pid <= 0:
                        raise ValueError("GPU PID unavailable")
                    trace_stage("nvmlSystemGetProcessName")
                    name = text_string(nvml.nvmlSystemGetProcessName(pid))
                    if "nvidia-cuda-mps" in name:
                        raise ValueError("MPS server unsupported")
                    processes.setdefault(pid, set()).add(kind)
            # v1 includes driver-reserved framebuffer pages. v2.used excludes
            # reserved pages and matches the existing nvidia-smi used field.
            trace_stage("nvmlDeviceGetMemoryInfo_v2")
            memory = int(nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2).used)
            trace_stage("nvmlDeviceGetUtilizationRates")
            utilization = int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            if memory < 0 or not 0 <= utilization <= 100:
                raise ValueError("Invalid NVML telemetry")
            result.append(
                {
                    "uuid": actual,
                    "memory_mib": (memory + 1024 * 1024 - 1) // (1024 * 1024),
                    "utilization_pct": utilization,
                    "processes": [{"pid": pid, "type": "+".join(sorted(kinds))} for pid, kinds in sorted(processes.items())],
                }
            )
        return result
    finally:
        trace_stage("nvmlShutdown")
        nvml.nvmlShutdown()


def bounded_sample(devices, timeout, policy):
    # A fresh CPU-only query process preserves the existing hard query deadline,
    # including a native NVML call that fails to return. It creates no CUDA context.
    with tempfile.TemporaryDirectory(prefix="lightx2v-nvml-query-") as directory:
        stage = Path(directory) / "stage"
        stage.write_text("Python startup/imports")
        env = dict(os.environ, LIGHTX2V_NVML_QUERY_STAGE=str(stage))
        try:
            raw = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "sample", *devices], timeout=timeout, env=env)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"NVML acquisition exceeded {timeout}s; last stage: {stage.read_text()}") from error
    gpus = json.loads(raw)
    if sorted(gpu["uuid"] for gpu in gpus) != sorted(devices):
        raise ValueError("Incomplete NVML device inventory")
    for gpu in gpus:
        for process in gpu["processes"]:
            process["ancestors"] = policy.lineage(process["pid"])
    return gpus


def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "sample":
        trace_stage("Python module loaded")
        print(json.dumps(sample_nvml(arguments[1:]), allow_nan=False))
        return 0
    if len(arguments) < 3 or arguments[0] != "--policy":
        raise ValueError("Use --policy /path/to/gpu_watch.py capture|check ...")
    path = Path(arguments[1]).resolve()
    spec = importlib.util.spec_from_file_location("lightx2v_gpu_watch_policy", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)

    def acquire(devices, timeout):
        return bounded_sample(devices, timeout, policy)

    policy.sample = acquire
    sys.argv = [str(path), *arguments[2:]]
    return policy.main()


if __name__ == "__main__":
    raise SystemExit(main())
