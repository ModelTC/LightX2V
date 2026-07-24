"""Environment-controlled one-step and one-block transformer profiling."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v.utils import op_shape_trace as ost
from lightx2v.utils.region_profile import active_profile, transformer_profile_regions
from lightx2v.utils.torch_trace_profiler import (
    TorchTraceProfileConfig,
    log_profile_done,
    log_profile_start,
    make_on_trace_ready,
)
from lightx2v_platform.base.global_var import AI_DEVICE

PROFILE_MODE_ENV = "LIGHTX2V_PROFILE_MODE"
PROFILE_STEP_ENV = "LIGHTX2V_PROFILE_STEP"
PROFILE_LAYER_ENV = "LIGHTX2V_PROFILE_LAYER"
PROFILE_OUTPUT_DIR_ENV = "LIGHTX2V_PROFILE_OUTPUT_DIR"
_PROFILE_OFF_VALUES = {"", "0", "false", "off", "none"}
_profile_suspended: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "lightx2v_transformer_profile_suspended",
    default=False,
)
torch_device_module = getattr(torch, AI_DEVICE)


@dataclass(frozen=True)
class ProfileRunMeta:
    wait: int
    warmup: int
    active: int
    with_stack: bool
    record_shapes: bool = False
    pre_steps: int = 0

    def format_header(self) -> str:
        parts = [
            f"pre_steps={self.pre_steps}",
            f"schedule=wait{self.wait}_warmup{self.warmup}_active{self.active}",
            f"with_stack={self.with_stack}",
            f"record_shapes={self.record_shapes}",
        ]
        return "Profile  " + "  ".join(parts)


def _profile_mode() -> str | None:
    mode = os.environ.get(PROFILE_MODE_ENV, "").strip().lower()
    if mode in _PROFILE_OFF_VALUES:
        return None
    if mode not in {"full", "block"}:
        raise ValueError(f"{PROFILE_MODE_ENV}={mode!r} is invalid; expected 'full' or 'block'.")
    return mode


def _nonnegative_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name}={raw_value!r} is invalid; expected a non-negative integer.") from exc
    if value < 0:
        raise ValueError(f"{name}={value} is invalid; expected a non-negative integer.")
    return value


def _profile_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    configured = os.environ.get(PROFILE_OUTPUT_DIR_ENV)
    if configured is None:
        return repo_root / "prof_results"
    root = Path(configured).expanduser()
    return root if root.is_absolute() else repo_root / root


def _trace_paths(out_dir: Path) -> set[Path]:
    return set(out_dir.glob("*.pt.trace.json"))


def _latest_trace(out_dir: Path, previous_traces: set[Path]) -> Path:
    traces = [path for path in out_dir.glob(f"*_{os.getpid()}.*.pt.trace.json") if path not in previous_traces]
    if not traces:
        traces = [path for path in out_dir.glob("*.pt.trace.json") if path not in previous_traces]
    if not traces:
        raise RuntimeError(f"No new torch profiler trace was generated under {out_dir}.")
    return max(traces, key=lambda path: path.stat().st_mtime)


def _rank_suffix() -> str:
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return f"_rank_{torch.distributed.get_rank()}"
    return ""


@contextmanager
def _one_call_torch_profile(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TorchTraceProfileConfig(
        tb_dir=str(out_dir),
        wait=0,
        warmup=0,
        active=1,
        with_stack=False,
    )
    log_profile_start(cfg, name)

    import torch.profiler as torch_profiler
    from torch.profiler import ProfilerActivity, schedule

    with torch_profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=make_on_trace_ready(cfg),
    ) as profiler:
        try:
            with torch_profiler.record_function(name):
                yield
        finally:
            if hasattr(torch_device_module, "synchronize"):
                torch_device_module.synchronize()
            profiler.step()

    log_profile_done(cfg)


@contextmanager
def suspend_transformer_profile():
    """Temporarily prevent a configured profile from being consumed."""

    token = _profile_suspended.set(True)
    try:
        yield
    finally:
        _profile_suspended.reset(token)


class TransformerProfile:
    """Capture one configured diffusion step or one block within that step."""

    default_step = 1
    default_layer = 2

    def __init__(
        self,
        model_name: str,
        block_profile: Any | None = None,
        *,
        infer_steps: int | None = None,
        num_layers: int | None = None,
    ):
        self.model_name = model_name
        self.block_profile = block_profile
        self.mode = _profile_mode()
        self.step = self.default_step
        self.layer = self.default_layer
        self._profiled = False
        self._block_mode_active = False
        self._block_recorded = False

        if self.mode is not None:
            self.step = _nonnegative_env_int(PROFILE_STEP_ENV, self.default_step)
            self.layer = _nonnegative_env_int(PROFILE_LAYER_ENV, self.default_layer)
            if infer_steps is not None and self.step >= infer_steps:
                raise ValueError(f"{PROFILE_STEP_ENV}={self.step} is out of range for infer_steps={infer_steps}.")
            if self.mode == "block" and num_layers is not None and self.layer >= num_layers:
                raise ValueError(f"{PROFILE_LAYER_ENV}={self.layer} is out of range for num_layers={num_layers}.")
            logger.info(f"[Profile] {self.model_name} target: mode={self.mode}, step={self.step}, layer={self.layer if self.mode == 'block' else 'all'}")

    def mode_for_step(self, step_index: int) -> str | None:
        if self.mode is None or self._profiled or _profile_suspended.get():
            return None
        return self.mode if int(step_index) == self.step else None

    def _out_dir(self, mode: str) -> Path:
        target = f"full_step_{self.step}" if mode == "full" else f"block_step_{self.step}_layer_{self.layer}"
        return _profile_root() / f"{self.model_name}_transformer_profile" / target

    @contextmanager
    def record_transformer(self, mode: str | None):
        if mode is None:
            yield
            return

        if mode == "full":
            out_dir = self._out_dir(mode)
            previous_traces = _trace_paths(out_dir)
            with _one_call_torch_profile(out_dir, f"{self.model_name}_transformer_step_{self.step}"):
                yield
            self._profiled = True
            logger.info(f"[Profile] {self.model_name} full-step trace: {_latest_trace(out_dir, previous_traces)}")
            return

        self._block_mode_active = True
        self._block_recorded = False
        try:
            yield
            if not self._block_recorded:
                raise RuntimeError(f"Transformer block {self.layer} was not executed during profile step {self.step}.")
            self._profiled = True
        finally:
            self._block_mode_active = False

    def should_record_block(self, block_idx: int) -> bool:
        return self._block_mode_active and not self._block_recorded and block_idx == self.layer

    @contextmanager
    def record_block(self, block_idx: int):
        if not self.should_record_block(block_idx):
            raise RuntimeError(f"Block {block_idx} is not the configured profile target.")

        out_dir = self._out_dir("block")
        previous_traces = _trace_paths(out_dir)
        if self.block_profile is None:
            with transformer_profile_regions():
                with _one_call_torch_profile(out_dir, f"block_{block_idx}"):
                    yield
            self._block_recorded = True
            logger.info(f"[Profile] {self.model_name} block trace: {_latest_trace(out_dir, previous_traces)}")
            return

        suffix = _rank_suffix()
        op_trace_path = out_dir / f"block_{block_idx}{suffix}_op_trace.jsonl"
        try:
            ost.begin_recording(op_trace_path)
            with active_profile(self.block_profile):
                with transformer_profile_regions():
                    with _one_call_torch_profile(out_dir, f"block_{block_idx}"):
                        yield
        finally:
            try:
                ost.flush(op_trace_path)
            finally:
                ost.end_recording()

        analyze = import_module(self.block_profile.block_profile_report_module).analyze
        trace_path = _latest_trace(out_dir, previous_traces)
        report_path = out_dir / f"block_{block_idx}{suffix}_layer_trace.txt"
        profile_meta = ProfileRunMeta(pre_steps=0, wait=0, warmup=0, active=1, with_stack=False)
        report, _, _ = analyze(
            trace_path,
            op_trace_path,
            block_idx=block_idx,
            step_id=None,
            profile_meta=profile_meta,
        )
        report_path.write_text(report, encoding="utf-8")
        self._block_recorded = True
        logger.info(f"[Profile] {self.model_name} block trace: {trace_path}")
        logger.info(f"[Profile] {self.model_name} op trace: {op_trace_path}")
        logger.info(f"[Profile] {self.model_name} layer report: {report_path}")


__all__ = [
    "PROFILE_LAYER_ENV",
    "PROFILE_MODE_ENV",
    "PROFILE_OUTPUT_DIR_ENV",
    "PROFILE_STEP_ENV",
    "ProfileRunMeta",
    "TransformerProfile",
    "suspend_transformer_profile",
]
