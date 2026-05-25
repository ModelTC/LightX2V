"""PyTorch profiler trace export (TensorBoard / Chrome Trace).

Wrap the target call site with TorchTraceProfileContext to collect traces.
See docs/ZH_CN/source/method_tutorials/torch_profiling.md for usage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, TypeVar

from loguru import logger

ProfileFormat = Literal["tensorboard", "chrome"]

T = TypeVar("T")


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(path)


def _default_tb_dir() -> str:
    return os.path.join(os.getcwd(), "save_results", "torch_profile")


def _default_chrome_path() -> str:
    return os.path.join(os.getcwd(), "save_results", "trace.json")


def _func_label(func) -> str:
    qualname = getattr(func, "__qualname__", func.__name__)
    module = getattr(func, "__module__", "")
    return f"{module}.{qualname}" if module else qualname


@dataclass
class TorchTraceProfileConfig:
    profile_format: ProfileFormat = "tensorboard"
    tb_dir: str = field(default_factory=_default_tb_dir)
    chrome_path: str = field(default_factory=_default_chrome_path)
    wait: int = 1
    warmup: int = 3
    active: int = 1
    with_stack: bool = False
    tensorboard_port: int = 16006
    exported_chrome_path: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.tb_dir = _resolve_path(self.tb_dir)
        self.chrome_path = _resolve_path(self.chrome_path)

    @property
    def steps(self) -> int:
        return self.wait + self.warmup + self.active


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
        logger.info(f"[Profile] step={step} chrome={chrome_path}")

    return handler


def log_profile_start(cfg: TorchTraceProfileConfig, name: Optional[str] = None) -> None:
    label = f"{name}: " if name else ""
    logger.info(
        f"[Profile] {label}torch trace start: format={cfg.profile_format}, "
        f"tb_dir={cfg.tb_dir}, chrome={cfg.chrome_path}, "
        f"steps={cfg.steps}, "
        f"schedule=wait{cfg.wait}_warmup{cfg.warmup}_active{cfg.active}, "
        f"with_stack={cfg.with_stack}"
    )


def log_profile_done(cfg: TorchTraceProfileConfig) -> None:
    lines = ["[Profile] torch trace export finished."]

    if cfg.profile_format == "tensorboard":
        lines.extend(
            [
                f"  TensorBoard logdir: {cfg.tb_dir}",
                f"  tensorboard --logdir {cfg.tb_dir} --port {cfg.tensorboard_port} --bind_all",
                f"  Open PYTORCH PROFILER: http://127.0.0.1:{cfg.tensorboard_port}/#pytorch_profiler",
                "  (Remote SSH: forward this port in your IDE and open the URL locally)",
            ]
        )
    else:
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
    _profile_owner: Optional[str] = None
    _warned_skipped: set[str] = set()

    def __init__(self, cfg: TorchTraceProfileConfig):
        self.cfg = cfg
        self._ran = False

    @classmethod
    def reset_session(cls) -> None:
        cls._session_done = False
        cls._profile_owner = None
        cls._warned_skipped = set()
        TorchTraceProfileContext._registered_labels.clear()

    @property
    def ran(self) -> bool:
        return self._ran

    def try_claim(self, label: str) -> bool:
        if self._session_done:
            return False
        if self._profile_owner is None:
            self._profile_owner = label
            return True
        if self._profile_owner == label:
            return True
        if label not in self._warned_skipped:
            logger.warning(
                f"[Profile] Skip torch trace for '{label}': "
                f"already profiling '{self._profile_owner}' (one call site per process)"
            )
            self._warned_skipped.add(label)
        return False

    def run(self, step_fn: Callable[[], T]) -> T:
        import torch.profiler as torch_profiler
        from torch.profiler import ProfilerActivity, schedule

        cfg = self.cfg
        if cfg.profile_format == "tensorboard":
            os.makedirs(cfg.tb_dir, exist_ok=True)

        on_trace_ready = make_on_trace_ready(cfg)
        my_schedule = schedule(wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=1)
        result: T

        with torch_profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=my_schedule,
            record_shapes=False,
            profile_memory=False,
            with_stack=cfg.with_stack,
            on_trace_ready=on_trace_ready,
        ) as prof:
            for _ in range(cfg.steps):
                result = step_fn()
                prof.step()

        self._ran = True
        self._session_done = True
        log_profile_done(cfg)
        return result


class TorchTraceProfileContext:
    """Call-site context for torch trace profiling."""

    _registered_labels: set[str] = set()

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        profile_format: ProfileFormat = "tensorboard",
        tb_dir: Optional[str] = None,
        chrome_path: Optional[str] = None,
        wait: int = 1,
        warmup: int = 3,
        active: int = 1,
        with_stack: bool = False,
        tensorboard_port: int = 16006,
    ):
        """
        Args:
            name: Optional label for logs and call-site identification. If omitted,
                the qualified name of the profiled function is used at runtime.
            profile_format: Export format, ``tensorboard`` or ``chrome``.
            tb_dir: TensorBoard logdir when ``profile_format=tensorboard``.
            chrome_path: Chrome trace output path when ``profile_format=chrome``.
            wait: Schedule wait steps (no collection).
            warmup: Schedule warmup steps (collect but do not export).
            active: Schedule active steps (export trace).
            with_stack: Whether to record Python call stacks.
            tensorboard_port: Port shown in post-profile log hints.
        """
        self.label = name
        cfg = TorchTraceProfileConfig(
            profile_format=profile_format,
            tb_dir=tb_dir or _default_tb_dir(),
            chrome_path=chrome_path or _default_chrome_path(),
            wait=wait,
            warmup=warmup,
            active=active,
            with_stack=with_stack,
            tensorboard_port=tensorboard_port,
        )
        self._profiler = TorchTraceProfiler(cfg)
        if not TorchTraceProfiler._session_done:
            self._register_label()

    def _register_label(self) -> None:
        label = self.label or "torch_trace"
        if label in self._registered_labels:
            return
        self._registered_labels.add(label)
        if len(self._registered_labels) > 1:
            logger.warning(
                "[Profile] Multiple TorchTraceProfileContext call sites detected: "
                f"{sorted(self._registered_labels)}. Only the first invoked site will be profiled once."
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def run(self, func: Callable[..., T], /, *args, **kwargs) -> T:
        label = self.label or _func_label(func)
        if not self._profiler.try_claim(label):
            return func(*args, **kwargs)

        log_profile_start(self._profiler.cfg, name=label)
        return self._profiler.run(lambda: func(*args, **kwargs))
