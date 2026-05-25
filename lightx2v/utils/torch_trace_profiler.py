"""PyTorch profiler trace export (TensorBoard / Chrome Trace) controlled by env vars.

See docs/ZH_CN/source/method_tutorials/torch_profiling.md for usage.
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from loguru import logger

ProfileFormat = Literal["tensorboard", "chrome", "both"]

_FALSE_VALUES = frozenset({"0", "false", "no", "off", ""})


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name)
    if not value:
        return default
    return value.lower() not in _FALSE_VALUES


def _env_enabled() -> bool:
    return _env_bool("LIGHTX2V_TORCH_PROFILE")


def _env_format() -> ProfileFormat:
    value = _env("LIGHTX2V_TORCH_PROFILE_FORMAT", "tensorboard").lower()
    if value not in ("tensorboard", "chrome", "both"):
        logger.warning(f"[Profile] Unknown format '{value}', fallback to tensorboard")
        return "tensorboard"
    return value  # type: ignore[return-value]


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(path)


@dataclass
class TorchTraceProfileConfig:
    enabled: bool
    profile_format: ProfileFormat
    tb_dir: str
    chrome_path: str
    steps: int
    wait: int
    warmup: int
    active: int
    once: bool = True
    with_stack: bool = False
    exported_chrome_path: str = field(default="", init=False)

    @classmethod
    def from_env(cls, default_tb_dir: Optional[str] = None) -> "TorchTraceProfileConfig":
        wait = int(_env("LIGHTX2V_TORCH_PROFILE_WAIT", "1"))
        warmup = int(_env("LIGHTX2V_TORCH_PROFILE_WARMUP", "3"))
        active = int(_env("LIGHTX2V_TORCH_PROFILE_ACTIVE", "1"))
        min_steps = wait + warmup + active

        steps_raw = _env("LIGHTX2V_TORCH_PROFILE_STEPS")
        steps = int(steps_raw) if steps_raw else min_steps
        if steps < min_steps:
            logger.warning(
                f"[Profile] steps={steps} < wait+warmup+active={min_steps}, "
                f"raising steps to {min_steps}"
            )
            steps = min_steps

        tb_dir = _env("LIGHTX2V_TORCH_PROFILE_TB_DIR")
        if not tb_dir:
            tb_dir = default_tb_dir or os.path.join(
                os.getcwd(), "save_results", "torch_profile"
            )
        tb_dir = _resolve_path(tb_dir)

        chrome_path = _env("LIGHTX2V_TORCH_PROFILE_CHROME")
        if not chrome_path:
            chrome_path = os.path.join(os.getcwd(), "save_results", "trace.json")
        chrome_path = _resolve_path(chrome_path)

        return cls(
            enabled=_env_enabled(),
            profile_format=_env_format(),
            tb_dir=tb_dir,
            chrome_path=chrome_path,
            steps=steps,
            wait=wait,
            warmup=warmup,
            active=active,
            once=_env_bool("LIGHTX2V_TORCH_PROFILE_ONCE", default=True),
            with_stack=_env_bool("LIGHTX2V_TORCH_PROFILE_STACK"),
        )

    def tensorboard_path_for_step(self, step: int) -> str:
        host = socket.gethostname().replace(".", "_")
        pid = os.getpid()
        timestamp = time.time_ns()
        return os.path.join(
            self.tb_dir,
            f"{host}_{pid}.{pid}.{timestamp}.pt.trace.json",
        )


def _copy_chrome_trace_for_tensorboard(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def make_on_trace_ready(cfg: TorchTraceProfileConfig) -> Optional[Callable]:
    if cfg.profile_format == "tensorboard":
        from torch.profiler import tensorboard_trace_handler

        os.makedirs(cfg.tb_dir, exist_ok=True)
        return tensorboard_trace_handler(cfg.tb_dir)

    def handler(prof) -> None:
        step = prof.step_num
        chrome_path = cfg.chrome_path
        os.makedirs(os.path.dirname(chrome_path) or ".", exist_ok=True)
        prof.export_chrome_trace(chrome_path)
        cfg.exported_chrome_path = chrome_path

        if cfg.profile_format == "both":
            tb_path = cfg.tensorboard_path_for_step(step)
            _copy_chrome_trace_for_tensorboard(chrome_path, tb_path)
            logger.info(
                f"[Profile] step={step} chrome={chrome_path} tensorboard={tb_path}"
            )
        else:
            logger.info(f"[Profile] step={step} chrome={chrome_path}")

    return handler


def log_profile_start(cfg: TorchTraceProfileConfig) -> None:
    logger.info(
        f"[Profile] torch trace enabled: format={cfg.profile_format}, "
        f"tb_dir={cfg.tb_dir}, chrome={cfg.chrome_path}, "
        f"steps={cfg.steps}, "
        f"schedule=wait{cfg.wait}_warmup{cfg.warmup}_active{cfg.active}, "
        f"once={cfg.once}, with_stack={cfg.with_stack}"
    )


def log_profile_done(cfg: TorchTraceProfileConfig) -> None:
    tb_port = os.environ.get("TENSORBOARD_PORT", "16006")
    lines = ["[Profile] torch trace export finished."]

    if cfg.profile_format in ("tensorboard", "both"):
        lines.extend(
            [
                f"  TensorBoard logdir: {cfg.tb_dir}",
                f"  tensorboard --logdir {cfg.tb_dir} --port {tb_port} --bind_all",
                f"  Open PYTORCH PROFILER: http://127.0.0.1:{tb_port}/#pytorch_profiler",
                "  (Docker + Remote SSH: bash scripts/run_tensorboard_docker_bridge.sh)",
            ]
        )

    if cfg.profile_format in ("chrome", "both"):
        chrome_path = cfg.exported_chrome_path or cfg.chrome_path
        lines.extend(
            [
                f"  Chrome trace: {chrome_path}",
                "  Open in Perfetto: https://ui.perfetto.dev/",
            ]
        )

    logger.info("\n".join(lines))


class TorchTraceProfiler:
    _session_done = False

    def __init__(self, cfg: TorchTraceProfileConfig):
        self.cfg = cfg
        self._ran = False

    @classmethod
    def from_env(cls, default_tb_dir: Optional[str] = None) -> "TorchTraceProfiler":
        return cls(TorchTraceProfileConfig.from_env(default_tb_dir=default_tb_dir))

    @classmethod
    def reset_session(cls) -> None:
        cls._session_done = False

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    @property
    def ran(self) -> bool:
        return self._ran

    def should_run(self) -> bool:
        if not self.enabled:
            return False
        if self.cfg.once and TorchTraceProfiler._session_done:
            return False
        return True

    def run(self, step_fn: Callable[[], None]) -> None:
        import torch.profiler as torch_profiler
        from torch.profiler import ProfilerActivity, schedule

        cfg = self.cfg
        if cfg.profile_format in ("tensorboard", "both"):
            os.makedirs(cfg.tb_dir, exist_ok=True)

        on_trace_ready = make_on_trace_ready(cfg)
        my_schedule = schedule(
            wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=1
        )

        with torch_profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=my_schedule,
            record_shapes=False,
            profile_memory=False,
            with_stack=cfg.with_stack,
            on_trace_ready=on_trace_ready,
        ) as prof:
            for _ in range(cfg.steps):
                step_fn()
                prof.step()

        self._ran = True
        if cfg.once:
            TorchTraceProfiler._session_done = True
        log_profile_done(cfg)
