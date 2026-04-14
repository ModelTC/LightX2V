from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from locust import LoadTestShape, User, events, task
except ModuleNotFoundError:
    class _EventHook:
        def add_listener(self, fn):
            return fn

        def fire(self, **kwargs):
            return None

    class _Events:
        def __init__(self):
            self.test_start = _EventHook()
            self.test_stop = _EventHook()
            self.request = _EventHook()

    class LoadTestShape:  # type: ignore[no-redef]
        pass

    class User:  # type: ignore[no-redef]
        pass

    def task(fn):  # type: ignore[no-redef]
        return fn

    events = _Events()  # type: ignore[no-redef]

from lightx2v.disagg.conn import REQUEST_POLLING_PORT, ReqManager


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG_JSON = REPO_ROOT / "configs" / "disagg" / "wan22_i2v_distill_controller.json"
DEFAULT_STAGE_DEFINITIONS_JSON = REPO_ROOT / "configs" / "disagg" / "wan22_i2v_workload_stages.json"

_TEST_START_MONOTONIC: Optional[float] = None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_base_config() -> dict[str, Any]:
    config_path = os.getenv("DISAGG_BASE_CONFIG_JSON")
    if config_path:
        path = Path(config_path)
    else:
        path = DEFAULT_BASE_CONFIG_JSON

    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "task": "i2v",
        "model_cls": "wan2.2_moe",
        "seed": 42,
        "prompt": "A cinematic cat scene with detailed lighting and motion.",
        "negative_prompt": "blurry, low quality, artifacts",
        "save_path": str(REPO_ROOT / "save_results" / "locust_disagg.mp4"),
    }


def _load_stage_definitions() -> list[dict[str, Any]]:
    stage_file = Path(os.getenv("DISAGG_WORKLOAD_STAGES_JSON", str(DEFAULT_STAGE_DEFINITIONS_JSON)))
    if not stage_file.is_file():
        raise FileNotFoundError(f"workload stage config not found: {stage_file}")

    with stage_file.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, list) or not loaded:
        raise ValueError(f"{stage_file} must contain a non-empty JSON list")

    return loaded


@dataclass(frozen=True)
class StageSpec:
    name: str
    duration_s: float
    user_count: int
    spawn_rate: float
    wait_time_s: float = 0.0
    config_variants: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "StageSpec":
        name = str(raw.get("name", "stage"))
        duration_s = float(raw.get("duration_s", 0.0))
        user_count = int(raw.get("user_count", 1))
        spawn_rate = float(raw.get("spawn_rate", max(1, user_count)))
        wait_time_s = float(raw.get("wait_time_s", 0.0))
        config_variants = raw.get("config_variants", []) or []
        if not isinstance(config_variants, list):
            raise ValueError(f"stage {name}: config_variants must be a list")
        return StageSpec(
            name=name,
            duration_s=max(duration_s, 0.0),
            user_count=max(user_count, 1),
            spawn_rate=max(spawn_rate, 0.1),
            wait_time_s=max(wait_time_s, 0.0),
            config_variants=[variant for variant in config_variants if isinstance(variant, dict)],
        )


def _load_stage_specs() -> list[StageSpec]:
    return [StageSpec.from_dict(stage) for stage in _load_stage_definitions()]


def load_base_config() -> dict[str, Any]:
    return _load_base_config()


def load_stage_specs() -> list[StageSpec]:
    return _load_stage_specs()


def _elapsed_since_start() -> float:
    if _TEST_START_MONOTONIC is None:
        return 0.0
    return max(0.0, time.monotonic() - _TEST_START_MONOTONIC)


def _stage_index_for_elapsed(stages: list[StageSpec], elapsed_s: float) -> int:
    if not stages:
        return 0

    accumulated = 0.0
    for index, stage in enumerate(stages):
        accumulated += stage.duration_s
        if elapsed_s < accumulated:
            return index
    return len(stages) - 1


def _current_stage(stages: list[StageSpec]) -> StageSpec:
    return stages[_stage_index_for_elapsed(stages, _elapsed_since_start())]


def _build_request_payload(base_config: dict[str, Any], stage: StageSpec, request_index: int) -> dict[str, Any]:
    payload = copy.deepcopy(base_config)
    variant = stage.config_variants[request_index % len(stage.config_variants)] if stage.config_variants else {}
    payload = _deep_merge(payload, variant)

    payload.setdefault("request_metrics", {})
    payload["request_metrics"]["request_id"] = request_index
    payload["request_metrics"]["client_send_ts"] = time.time()
    payload["request_metrics"]["stage_name"] = stage.name
    payload["request_metrics"]["load_stage"] = stage.name

    if "data_bootstrap_room" not in payload:
        payload["data_bootstrap_room"] = request_index

    save_path_prefix = os.getenv("DISAGG_WORKLOAD_SAVE_PREFIX")
    if save_path_prefix:
        save_root = Path(save_path_prefix)
        save_root.parent.mkdir(parents=True, exist_ok=True)
        payload["save_path"] = str(save_root.with_name(f"{save_root.stem}_{stage.name}_{request_index}{save_root.suffix}"))

    return payload


def _get_controller_target() -> tuple[str, int]:
    host = os.getenv("DISAGG_CONTROLLER_HOST", "127.0.0.1")
    port = int(os.getenv("DISAGG_CONTROLLER_REQUEST_PORT", str(REQUEST_POLLING_PORT - 2)))
    return host, port


def _send_to_controller(payload: dict[str, Any]) -> None:
    host, port = _get_controller_target()
    ReqManager().send(host, port, payload)


def start_workload_clock() -> None:
    global _TEST_START_MONOTONIC
    _TEST_START_MONOTONIC = time.monotonic()


def current_stage(stages: Optional[list[StageSpec]] = None) -> StageSpec:
    loaded_stages = stages or _load_stage_specs()
    return _current_stage(loaded_stages)


def build_payload(base_config: dict[str, Any], stage: StageSpec, request_index: int) -> dict[str, Any]:
    return _build_request_payload(base_config, stage, request_index)


def send_workload_end_signal() -> None:
    _send_to_controller(
        {
            "workload_end": True,
            "request_metrics": {
                "load_stage": "end",
                "client_send_ts": time.time(),
            },
        }
    )


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):  # type: ignore[override]
    start_workload_clock()


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs):  # type: ignore[override]
    send_workload_end_signal()


class DisaggLoadShape(LoadTestShape):
    """Time-based load shape for disaggregated LightX2V scenarios.

    Configure stages with DISAGG_WORKLOAD_STAGES_JSON as a JSON file path. Each stage supports:
    - duration_s
    - user_count
    - spawn_rate
    - wait_time_s
    - config_variants
    """

    stages = _load_stage_specs()

    def tick(self):
        elapsed_s = _elapsed_since_start()
        total_duration_s = sum(stage.duration_s for stage in self.stages)
        if total_duration_s > 0 and elapsed_s >= total_duration_s:
            return None

        stage = _current_stage(self.stages)
        return stage.user_count, stage.spawn_rate


class DisaggUser(User):
    base_config = _load_base_config()
    stages = _load_stage_specs()
    req_mgr = ReqManager()

    def wait_time(self):  # type: ignore[override]
        stage = _current_stage(self.stages)
        return stage.wait_time_s

    @task
    def submit_request(self):
        stage = _current_stage(self.stages)
        request_index = int(time.time() * 1000) % 1_000_000
        payload = _build_request_payload(self.base_config, stage, request_index)
        send_started = time.perf_counter()
        try:
            host, port = _get_controller_target()
            self.req_mgr.send(host, port, payload)
            events.request.fire(
                request_type="zmq",
                name=f"{stage.name}:config_push",
                response_time=(time.perf_counter() - send_started) * 1000.0,
                response_length=len(str(payload)),
                exception=None,
            )
        except Exception as exc:
            events.request.fire(
                request_type="zmq",
                name=f"{stage.name}:config_push",
                response_time=(time.perf_counter() - send_started) * 1000.0,
                response_length=0,
                exception=exc,
            )


__all__ = [
    "DisaggLoadShape",
    "DisaggUser",
    "StageSpec",
    "start_workload_clock",
    "current_stage",
    "build_payload",
    "send_workload_end_signal",
    "load_base_config",
    "load_stage_specs",
]