"""Workflow settings use LightX2V's startup names and Comfy-local paths."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .files import MAX_CONFIG_BYTES, local_path, validate_config
from .gpus import normalize_devices
from .models import inferred_variant, model_catalog, validate_selection

SERVER_WIDGETS = ("lightx2v_path", "model_path", "model_cls", "config_json", "task", "CUDA_VISIBLE_DEVICES")
RUNTIME_KEYS = {"model_variant", "nproc_per_node", "startup_timeout", "task_timeout", "http_timeout", "poll_interval", "cancel_grace", "stop_grace", "idle_ttl"}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        result = os.path.expandvars(os.path.expanduser(value))
        if "${" in result:
            raise ValueError("Unresolved environment variable: " + value)
        return result
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def idle_ttl() -> float:
    try:
        value = float(os.environ.get("LIGHTX2V_COMFY_IDLE_TTL", "600"))
    except ValueError as exc:
        raise ValueError("LIGHTX2V_COMFY_IDLE_TTL must be a nonnegative number of seconds") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("LIGHTX2V_COMFY_IDLE_TTL must be finite and nonnegative")
    return value


def same_gpu_policy() -> str:
    value = os.environ.get("LIGHTX2V_COMFY_SAME_GPU_POLICY", "exclusive")
    if value not in {"exclusive", "coexist"}:
        raise ValueError("LIGHTX2V_COMFY_SAME_GPU_POLICY must be exclusive or coexist")
    return value


def cache_fingerprint(inputs: dict):
    dependencies = {"python": os.environ.get("LIGHTX2V_COMFY_PYTHON") or sys.executable, "idle_ttl": idle_ttl(), "same_gpu_policy": same_gpu_policy()}
    # Comfy passes None for linked inputs; their upstream cache signatures cover them.
    if inputs.get("config_json"):
        value = _expand(inputs["config_json"])
        base = None
        if not Path(value).is_absolute():
            if not inputs.get("lightx2v_path"):
                return float("nan")
            base = local_path(inputs["lightx2v_path"])
        with local_path(value, base).open("rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
        validate_config(data)
        dependencies["config_json"] = hashlib.sha256(data).hexdigest()
    return _json(dependencies)


@dataclass(frozen=True)
class ServerRef:
    inputs_json: str

    def resolve(self) -> ServerSettings:
        return resolve_server(json.loads(self.inputs_json))


@dataclass(frozen=True)
class ServerSettings:
    name: str
    settings_json: str
    config_json: str
    fingerprint: str

    @property
    def service_key(self) -> str:
        s = self.settings
        fields = ("python", "lightx2v_path", "model_path", "model_cls", "config_json", "model_variant", "CUDA_VISIBLE_DEVICES", "nproc_per_node")
        identity = {key: s[key] for key in fields}
        if s["model_cls"] not in {"minimax_h3", "taomate_h3"}:
            identity["task"] = s["task"]
        return hashlib.sha256(_json(identity).encode()).hexdigest()

    @property
    def settings(self) -> dict:
        return json.loads(self.settings_json)

    def command(self, config_path: Path, port: int, metric_port: int) -> list[str]:
        s = self.settings
        argv = [s["python"], "-m"]
        if s["nproc_per_node"] > 1:
            argv += ["torch.distributed.run", "--standalone", "--nproc_per_node=" + str(s["nproc_per_node"]), "-m"]
        argv += [
            "lightx2v.server",
            "--model_path",
            s["model_path"],
            "--model_cls",
            s["model_cls"],
            "--config_json",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--metric_port",
            str(metric_port),
            "--max_queue_size",
            "1",
        ]
        for field, flag in (("model_variant", "--model-variant"), ("task", "--task")):
            # H3 chooses startup weights via model_variant; task is a request input.
            if field == "task" and s["model_cls"] in {"minimax_h3", "taomate_h3"}:
                continue
            if s.get(field):
                argv += [flag, s[field]]
        return argv


def server_reference(lightx2v_path, model_path, model_cls, config_json, task="t2av", model_variant="", CUDA_VISIBLE_DEVICES="0", nproc_per_node=1) -> ServerRef:
    raw = {
        "lightx2v_path": lightx2v_path.strip(),
        "model_path": model_path.strip(),
        "model_cls": model_cls.strip(),
        "config_json": config_json.strip(),
        "task": task.strip(),
        "model_variant": model_variant.strip(),
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES.strip(),
        "nproc_per_node": nproc_per_node,
    }
    return ServerRef(_json(raw))


def resolve_server(raw: dict) -> ServerSettings:
    allowed = RUNTIME_KEYS | set(SERVER_WIDGETS)
    if set(raw) - allowed:
        raise ValueError("Unknown server settings: " + str(sorted(set(raw) - allowed)))
    s = {
        "python": os.environ.get("LIGHTX2V_COMFY_PYTHON") or sys.executable,
        "nproc_per_node": 1,
        "CUDA_VISIBLE_DEVICES": "0",
        "model_variant": "",
        "task": "t2av",
        "startup_timeout": 900,
        "task_timeout": 7200,
        "http_timeout": 5,
        "poll_interval": 0.5,
        "cancel_grace": 3,
        "stop_grace": 10,
        "idle_ttl": idle_ttl(),
    }
    s.update(_expand(raw))
    for key in ("lightx2v_path", "model_path", "config_json"):
        if not s.get(key):
            raise ValueError("Set " + key + " on the Service node")
        path = local_path(s[key], Path(s["lightx2v_path"]) if key == "config_json" else None)
        if not path.exists():
            raise ValueError(f"{key} does not exist in the ComfyUI filesystem: {path}")
        s[key] = str(path)
    python = Path(_expand(s["python"]))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("LIGHTX2V_COMFY_PYTHON must be an absolute executable path in the runtime environment")
    s["python"] = str(python)  # Keep venv symlinks intact.
    if not Path(s["lightx2v_path"]).is_dir() or not Path(s["model_path"]).is_dir():
        raise ValueError("lightx2v_path and model_path must be directories")
    if not isinstance(s.get("model_cls"), str) or not s["model_cls"]:
        raise ValueError("model_cls is required")
    if not isinstance(s["task"], str) or not s["task"]:
        raise ValueError("Set task on the Service node")
    validate_selection(model_catalog(s["lightx2v_path"]), s["model_cls"], s["task"])
    variant = inferred_variant(s["model_cls"], s["task"])
    if variant:
        if s["model_variant"] and s["model_variant"] != variant:
            raise ValueError("H3 task does not match model_variant")
        s["model_variant"] = variant
    if type(s["nproc_per_node"]) is not int or s["nproc_per_node"] < 1:
        raise ValueError("nproc_per_node must be a positive integer")
    if s["nproc_per_node"] > 1 and not sys.platform.startswith("linux"):
        raise ValueError("Multi-GPU process-tree cleanup requires Linux child subreaper support")
    if not isinstance(s["CUDA_VISIBLE_DEVICES"], str) or not s["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must be a nonempty string")
    s["CUDA_VISIBLE_DEVICES"] = normalize_devices(s["CUDA_VISIBLE_DEVICES"])
    if len(s["CUDA_VISIBLE_DEVICES"].split(",")) < s["nproc_per_node"]:
        raise ValueError("nproc_per_node exceeds the configured GPU count")
    for key in ("startup_timeout", "task_timeout", "http_timeout", "poll_interval", "cancel_grace", "stop_grace"):
        value = s[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")
    if isinstance(s["idle_ttl"], bool) or not isinstance(s["idle_ttl"], (int, float)) or not math.isfinite(s["idle_ttl"]) or s["idle_ttl"] < 0:
        raise ValueError("idle_ttl must be finite and nonnegative")
    with Path(s["config_json"]).open("rb") as handle:
        config = _expand(validate_config(handle.read(MAX_CONFIG_BYTES + 1)))
    for lora in config.get("lora_configs") or []:
        lora_path = local_path(lora["path"], Path(s["lightx2v_path"]))
        if not lora_path.is_file():
            raise ValueError("Configured LoRA does not exist: " + str(lora_path))
    config_json, settings_json = _json(config), _json(s)
    identity = dict(s)
    if s["model_cls"] in {"minimax_h3", "taomate_h3"}:
        identity.pop("task")  # Same H3 weights can handle multiple request tasks.
    fingerprint = hashlib.sha256((_json(identity) + "\n" + config_json).encode()).hexdigest()
    return ServerSettings(s["model_cls"], settings_json, config_json, fingerprint)
