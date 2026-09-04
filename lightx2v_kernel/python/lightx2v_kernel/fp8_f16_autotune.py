"""Automatic SM120 FP8 GEMM autotuning and persistent dispatch cache."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

import torch

_SCHEMA_VERSION = 1
_KERNEL_NAME = "sm120_fp8_f16_accum"
_DTYPE_NAMES = {
    0: "bfloat16",
    1: "float16",
}


def _require_ops() -> None:
    required = (
        "fp8_f16_accum_autotune_cache_abi_sm120",
        "fp8_f16_accum_autotune_configs_sm120",
        "set_fp8_f16_accum_autotune_config_sm120",
        "set_fp8_f16_accum_autotune_enabled_sm120",
        "get_fp8_f16_accum_autotune_cache_sm120",
        "clear_fp8_f16_accum_autotune_cache_sm120",
    )
    missing = [name for name in required if not hasattr(torch.ops.lightx2v_kernel, name)]
    if missing:
        raise ImportError(f"lightx2v-kernel was built without FP8-F16 autotune ops: {missing}")


def _device_index(device: torch.device | str | int | None) -> int:
    if isinstance(device, int):
        return device
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError(f"FP8-F16 autotune requires a CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


def _runtime_identity(device_index: int) -> dict:
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cache_abi": torch.ops.lightx2v_kernel.fp8_f16_accum_autotune_cache_abi_sm120(),
    }


def _default_cache_path(device_index: int) -> Path:
    properties = torch.cuda.get_device_properties(device_index)
    device_name = re.sub(r"[^a-z0-9]+", "-", properties.name.lower()).strip("-")
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "lightx2v" / "autotune" / "fp8_f16_accum" / f"sm{properties.major}{properties.minor}-{device_name}.json"


def _entry_key(entry: dict) -> tuple:
    return (
        int(entry["m"]),
        int(entry["n"]),
        int(entry["k"]),
        entry["out_dtype"],
        bool(entry["has_bias"]),
    )


def _validate_cache(cache: dict, device_index: int) -> list[dict]:
    if cache.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"Unsupported FP8-F16 autotune schema: {cache.get('schema_version')}")
    if cache.get("kernel") != _KERNEL_NAME:
        raise ValueError(f"Unexpected autotune kernel: {cache.get('kernel')!r}")

    expected = _runtime_identity(device_index)
    actual = cache.get("runtime")
    if actual != expected:
        raise ValueError(f"FP8-F16 autotune cache runtime mismatch: expected {expected}, got {actual}")

    config_names = torch.ops.lightx2v_kernel.fp8_f16_accum_autotune_configs_sm120()
    entries = cache.get("entries")
    if not isinstance(entries, list):
        raise ValueError("FP8-F16 autotune cache entries must be a list")
    seen = set()
    for entry in entries:
        key = _entry_key(entry)
        if min(key[:3]) <= 0:
            raise ValueError(f"GEMM dimensions must be positive, got {key[:3]}")
        if key[3] not in _DTYPE_NAMES.values():
            raise ValueError(f"Unsupported FP8-F16 output dtype: {key[3]!r}")
        if not isinstance(entry["has_bias"], bool):
            raise ValueError(f"has_bias must be a bool, got {entry['has_bias']!r}")
        config_id = int(entry["config_id"])
        if not 0 <= config_id < len(config_names):
            raise ValueError(f"Invalid FP8-F16 autotune config_id: {config_id}")
        if entry.get("config") != config_names[config_id]:
            raise ValueError(f"FP8-F16 autotune config name does not match config_id {config_id}")
        if key in seen:
            raise ValueError(f"Duplicate FP8-F16 autotune cache entry: {key}")
        seen.add(key)
    return entries


def _load_entries(cache_path: Path, device_index: int) -> list[dict]:
    if not cache_path.is_file():
        return []
    return _validate_cache(json.loads(cache_path.read_text()), device_index)


def _register_entries(entries: list[dict], device_index: int) -> None:
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    for entry in entries:
        torch.ops.lightx2v_kernel.set_fp8_f16_accum_autotune_config_sm120(
            device_index,
            int(entry["m"]),
            int(entry["n"]),
            int(entry["k"]),
            dtype_by_name[entry["out_dtype"]],
            bool(entry["has_bias"]),
            int(entry["config_id"]),
        )


def _current_entries(device_index: int) -> list[dict]:
    config_names = torch.ops.lightx2v_kernel.fp8_f16_accum_autotune_configs_sm120()
    cache = torch.ops.lightx2v_kernel.get_fp8_f16_accum_autotune_cache_sm120(device_index)
    entries = []
    for m, n, k, dtype_code, has_bias, config_id in cache.tolist():
        entries.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "out_dtype": _DTYPE_NAMES[dtype_code],
                "has_bias": bool(has_bias),
                "config_id": config_id,
                "config": config_names[config_id],
            }
        )
    return entries


class Fp8F16AccumAutotuner:
    """Manage automatic first-use tuning and its process-independent cache."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        device: torch.device | str | int | None = None,
    ):
        _require_ops()
        self.device_index = _device_index(device)
        if torch.cuda.get_device_capability(self.device_index) != (12, 0):
            raise ValueError("FP8-F16 autotune requires an SM120 device")
        self.cache_path = Path(cache_path).expanduser() if cache_path else _default_cache_path(self.device_index)
        self._saved_configs = {}

    def start(self) -> int:
        """Load compatible winners and enable exact-shape tuning on cache misses."""
        torch.ops.lightx2v_kernel.clear_fp8_f16_accum_autotune_cache_sm120(self.device_index)
        try:
            entries = _load_entries(self.cache_path, self.device_index)
        except (KeyError, OSError, TypeError, ValueError) as error:
            warnings.warn(f"Ignoring FP8-F16 autotune cache {self.cache_path}: {error}", stacklevel=2)
            entries = []
        _register_entries(entries, self.device_index)
        self._saved_configs = {_entry_key(entry): int(entry["config_id"]) for entry in entries}
        torch.ops.lightx2v_kernel.set_fp8_f16_accum_autotune_enabled_sm120(True)
        return len(entries)

    def save(self) -> int:
        """Merge newly tuned winners and atomically persist the cache."""
        current_entries = _current_entries(self.device_index)
        current_configs = {_entry_key(entry): int(entry["config_id"]) for entry in current_entries}
        if current_configs == self._saved_configs:
            return 0

        lock_path = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    disk_entries = _load_entries(self.cache_path, self.device_index)
                except (KeyError, OSError, TypeError, ValueError):
                    disk_entries = []
                merged = {_entry_key(entry): entry for entry in disk_entries}
                merged.update({_entry_key(entry): entry for entry in current_entries})
                entries = [merged[key] for key in sorted(merged)]
                payload = {
                    "schema_version": _SCHEMA_VERSION,
                    "kernel": _KERNEL_NAME,
                    "runtime": _runtime_identity(self.device_index),
                    "entries": entries,
                }
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        "w",
                        dir=self.cache_path.parent,
                        prefix=self.cache_path.name + ".",
                        delete=False,
                    ) as temporary:
                        temporary_path = Path(temporary.name)
                        json.dump(payload, temporary, indent=2)
                        temporary.write("\n")
                    os.replace(temporary_path, self.cache_path)
                finally:
                    if temporary_path is not None:
                        temporary_path.unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f"Could not persist FP8-F16 autotune cache {self.cache_path}: {error}", stacklevel=2)
            return 0

        self._saved_configs = current_configs
        return len(current_configs)
