"""Resolve device aliases without importing torch or creating a CUDA context."""

import csv
import io
import math
import subprocess
import time
from functools import lru_cache


def gpu_status() -> dict:
    """Read a fresh NVML snapshot; indices follow normalize_devices' PCI order."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is unavailable; GPU status could not be read") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GPU status query timed out; automatic refresh will retry") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi could not read GPU status; check the NVIDIA driver") from exc

    def metric(value):
        try:
            number = float(value)
            return number if math.isfinite(number) and number >= 0 else None
        except ValueError:
            return None  # NVIDIA reports N/A or [Not Supported] on some devices.

    devices = []
    try:
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if not row:
                continue
            physical, uuid, name, pci, total, used, utilization = (field.strip() for field in row)
            if not uuid.startswith("GPU-") or not pci:
                raise ValueError("Missing GPU identity")
            devices.append(
                {
                    "nvidia_index": int(physical),
                    "uuid": uuid,
                    "name": name,
                    "pci_bus_id": pci,
                    "memory_total_mib": metric(total),
                    "memory_used_mib": metric(used),
                    "utilization_gpu_percent": metric(utilization),
                }
            )
    except (csv.Error, ValueError) as exc:
        raise RuntimeError("nvidia-smi returned an invalid GPU inventory") from exc
    devices.sort(key=lambda device: device["pci_bus_id"])
    for index, device in enumerate(devices):
        device["index"] = index
    return {"devices": devices, "sampled_at": time.time()}


@lru_cache(maxsize=1)
def inventory():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return ()  # CPU-only configuration checks and test backends.
    rows = [tuple(part.strip() for part in line.split(",")) for line in result.stdout.splitlines() if line.strip()]
    return tuple(row[0] for row in sorted(rows, key=lambda row: row[1]))


def normalize_devices(value: str) -> str:
    devices = [part.strip() for part in value.split(",")]
    if not devices or any(not part or not (part.isdecimal() or part.startswith("GPU-")) for part in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES requires comma-separated device indices or GPU UUIDs (MIG is not supported)")
    available = inventory()
    resolved = []
    for device in devices:
        if available:
            matches = [available[int(device)]] if device.isdecimal() and int(device) < len(available) else [gpu for gpu in available if gpu.startswith(device)] if not device.isdecimal() else []
            if len(matches) != 1:
                raise ValueError("GPU is unavailable or ambiguous in this runtime: " + device)
            device = matches[0]
        elif device.isdecimal():
            device = str(int(device))
        resolved.append(device)
    if len(set(resolved)) != len(resolved):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPUs")
    return ",".join(resolved)
