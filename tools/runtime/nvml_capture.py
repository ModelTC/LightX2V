"""Optional bounded NVML acquisition for the existing gpu_watch capture/check CLI.

The supplied policy module retains PID ancestry, task windows, idle checks and
coverage rules. This module only supplies the same required device inventory.
"""

import importlib.metadata
import importlib.util
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BINDING_VERSION = "12.550.52"


def trace_stage(stage):
    destination = os.environ.get("LIGHTX2V_NVML_QUERY_STAGE")
    if destination:
        Path(destination).write_text(stage)


def text_string(value):
    return value.decode() if isinstance(value, bytes) else value


def load_binding():
    trace_stage("binding metadata")
    if importlib.metadata.version("nvidia-ml-py") != BINDING_VERSION:
        raise ValueError("Install the pinned nvidia-ml-py binding")
    trace_stage("import pynvml")
    import pynvml

    return pynvml


def sample_nvml(devices, nvml=None, initialized=False):
    if nvml is None:
        nvml = load_binding()
    if not initialized:
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
        if not initialized:
            trace_stage("nvmlShutdown")
            nvml.nvmlShutdown()


class NVMLQueryTimeout(subprocess.TimeoutExpired):
    """Keep the original gpu_watch error contract and include acquisition stage."""

    def __init__(self, command, timeout, stage):
        super().__init__(command, timeout)
        self.stage = stage

    def __str__(self):
        return f"{super().__str__()}; last stage: {self.stage}"


class NVMLQueryProcess:
    """One CPU-only NVML session; every request still has a hard deadline."""

    def __init__(self, helper=None):
        self.helper = Path(helper or __file__).resolve()
        self.process = None
        self.directory = None
        self.buffer = b""

    def _start(self, devices):
        self.directory = tempfile.TemporaryDirectory(prefix="lightx2v-nvml-query-", dir=os.environ.get("LIGHTX2V_EVAL_RESOURCE_SCRATCH", "/tmp"))
        self.stage = Path(self.directory.name) / "stage"
        self.stage.write_text("Python startup/imports")
        self.devices = list(devices)
        self.command = [sys.executable, str(self.helper), "serve", *devices]
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=dict(os.environ, LIGHTX2V_NVML_QUERY_STAGE=str(self.stage)),
            bufsize=0,
        )
        os.set_blocking(self.process.stdout.fileno(), False)

    def query(self, devices, timeout):
        deadline = time.monotonic() + timeout
        try:
            if self.process is None:
                self._start(devices)
            if list(devices) != self.devices:
                raise ValueError("Cannot change NVML capture devices")
            self.process.stdin.write(b"sample\n")
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in self.buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise NVMLQueryTimeout(self.command, timeout, self.stage.read_text())
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block:
                        raise ValueError(f"NVML query process exited; last stage: {self.stage.read_text()}")
                    self.buffer += block
            line, self.buffer = self.buffer.split(b"\n", 1)
            response = json.loads(line)
            if "error" in response:
                raise ValueError(f"NVML inventory unavailable: {response['error']}")
            gpus = response["gpus"]
            if sorted(gpu["uuid"] for gpu in gpus) != sorted(devices):
                raise ValueError("Incomplete NVML device inventory")
            return gpus
        except BaseException:
            # This CPU observer is our own subprocess, never a GPU workload PID.
            self.close()
            raise

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
            self.process.stdout.close()
            self.process = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
        self.buffer = b""


def serve(devices):
    nvml = load_binding()
    trace_stage("nvmlInit")
    nvml.nvmlInit()
    try:
        for request in sys.stdin:
            try:
                if request.strip() != "sample":
                    raise ValueError("Unknown NVML query request")
                response = {"gpus": sample_nvml(devices, nvml, initialized=True)}
            except Exception as error:
                response = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response, allow_nan=False), flush=True)
            trace_stage("awaiting query")
    finally:
        trace_stage("nvmlShutdown")
        nvml.nvmlShutdown()


def sample_with_policy(client, devices, timeout, policy):
    gpus = client.query(devices, timeout)
    for gpu in gpus:
        for process in gpu["processes"]:
            process["ancestors"] = policy.lineage(process["pid"])
    return gpus


def main():
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "sample":
        print(json.dumps(sample_nvml(arguments[1:]), allow_nan=False))
        return 0
    if arguments and arguments[0] == "serve":
        serve(arguments[1:])
        return 0
    if len(arguments) < 3 or arguments[0] != "--policy":
        raise ValueError("Use --policy /path/to/gpu_watch.py capture|check ...")
    path = Path(arguments[1]).resolve()
    spec = importlib.util.spec_from_file_location("lightx2v_gpu_watch_policy", path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    client = NVMLQueryProcess()

    def acquire(devices, timeout):
        return sample_with_policy(client, devices, timeout, policy)

    policy.sample = acquire
    sys.argv = [str(path), *arguments[2:]]
    try:
        return policy.main()
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
