"""Small env-driven transformer profile helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from lightx2v.utils import op_shape_trace as ost
from lightx2v.utils.region_profile import active_profile
from lightx2v.utils.torch_trace_profiler import (
    TorchTraceProfileConfig,
    TorchTraceProfiler,
    log_profile_done,
    log_profile_start,
    make_on_trace_ready,
)
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)

PROFILE_MODE_ENV = "LIGHTX2V_PROFILE_MODE"
LEGACY_TRACE_MODE_ENV = "LIGHTX2V_TRACE_MODE"
_PROFILE_OFF_VALUES = {"", "0", "false", "off", "none"}
_MODEL_DISPLAY_NAMES = {
    "hunyuan3d": "Hunyuan3D",
    "seko_talk": "SekoTalk",
}


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


def get_profile_mode() -> str | None:
    mode = os.environ.get(PROFILE_MODE_ENV, os.environ.get(LEGACY_TRACE_MODE_ENV, "")).strip().lower()
    if mode in _PROFILE_OFF_VALUES:
        return None
    if mode not in {"full", "block"}:
        raise ValueError(f"{PROFILE_MODE_ENV}/{LEGACY_TRACE_MODE_ENV}={mode!r} is invalid. Use 'full' or 'block'.")
    return mode


def make_profile_out_dir(out_dir_name: str, mode: str, infer_step: int, layer: int | None = None) -> Path:
    repo_dir = Path(os.environ.get("lightx2v_path", os.getcwd()))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if mode == "full":
        return repo_dir / "prof_results" / out_dir_name / f"full_step_{infer_step}_{stamp}"
    return repo_dir / "prof_results" / out_dir_name / f"block_step_{infer_step}_layer_{layer}_{stamp}"


def log_profile_target(display_name: str, mode: str, infer_step: int, out_dir: Path, layer: int | None = None) -> None:
    if mode == "full":
        logger.info(f"[Profile] {display_name} full profile target: infer_step={infer_step}, out_dir={out_dir}")
        return
    logger.info(f"[Profile] {display_name} block profile target: infer_step={infer_step}, layer={layer}, out_dir={out_dir}")


@contextmanager
def _one_step_torch_profile(out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = TorchTraceProfileConfig(
        tb_dir=str(out_dir),
        wait=0,
        warmup=0,
        active=1,
        with_stack=False,
    )
    TorchTraceProfiler.reset_session()
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
    ) as prof:
        yield
        if hasattr(torch_device_module, "synchronize"):
            torch_device_module.synchronize()
        prof.step()

    log_profile_done(cfg)


class TransformerProfile:
    # Developer-side profile target. Change these concrete indices directly
    # when inspecting a different infer step or a different transformer block.
    profile_infer_step = 1
    profile_block_idx = 2

    def __init__(self, model_type: str, block_profile: Any | None = None):
        self.display_name = _MODEL_DISPLAY_NAMES.get(model_type, model_type)
        self.out_dir_name = f"{model_type}_transformer_profile"
        self.profile_name = f"{model_type}_transformer"
        self.block_profile = block_profile
        self.infer_step = self.profile_infer_step
        self.block_idx = self.profile_block_idx
        self.out_dir = None
        self.op_trace_path = None

    def mode_for_step(self, step_index: int) -> str | None:
        if int(step_index) != self.infer_step:
            return None
        return get_profile_mode()

    @contextmanager
    def record_full(self):
        infer_step = self.infer_step
        self.infer_step = None
        self.out_dir = make_profile_out_dir(self.out_dir_name, "full", infer_step)
        log_profile_target(self.display_name, "full", infer_step, self.out_dir)
        with _one_step_torch_profile(self.out_dir, f"{self.profile_name}_full_step_{infer_step}"):
            yield
        logger.info(f"[Profile] {self.profile_name} full trace: {latest_trace(self.out_dir)}")

    def start_block(self):
        if self.block_profile is None:
            raise RuntimeError("Block profile requires block_profile.")
        infer_step = self.infer_step
        self.infer_step = None
        self.out_dir = make_profile_out_dir(self.out_dir_name, "block", infer_step, self.block_idx)
        self.op_trace_path = self.out_dir / f"block_{self.block_idx}_op_trace.jsonl"
        log_profile_target(self.display_name, "block", infer_step, self.out_dir, self.block_idx)
        return self

    @contextmanager
    def record_block(self, block_idx: int):
        try:
            with _enabled_env(self.block_profile.profile_env):
                ost.begin_recording(self.op_trace_path)
                with active_profile(self.block_profile):
                    with _one_step_torch_profile(self.out_dir, f"block_{block_idx}"):
                        yield
        finally:
            ost.flush(self.op_trace_path)
            ost.end_recording()

    def write_block_report(self) -> None:
        analyze = import_module(self.block_profile.block_profile_report_module).analyze
        trace_path = latest_trace(self.out_dir)
        if not self.op_trace_path.is_file():
            raise RuntimeError(f"No op trace found for block profile under {self.out_dir}")

        report_path = self.out_dir / f"block_{self.block_idx}_layer_trace.txt"
        profile_meta = ProfileRunMeta(pre_steps=0, wait=0, warmup=0, active=1, with_stack=False, record_shapes=False)
        report, _, _ = analyze(trace_path, self.op_trace_path, block_idx=self.block_idx, step_id=None, profile_meta=profile_meta)
        report_path.write_text(report)
        logger.info(f"[Profile] {self.display_name} block trace: {trace_path}")
        logger.info(f"[Profile] {self.display_name} op trace: {self.op_trace_path}")
        logger.info(f"[Profile] {self.display_name} layer report: {report_path}")
        self.out_dir = None
        self.op_trace_path = None


def latest_trace(out_dir: Path) -> Path:
    traces = list(out_dir.glob("*.pt.trace.json"))
    if not traces:
        raise RuntimeError(f"No torch profiler trace found under {out_dir}")
    return max(traces, key=lambda path: path.stat().st_mtime)


@contextmanager
def _enabled_env(env_name: str):
    old_value = os.environ.get(env_name)
    os.environ[env_name] = "1"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = old_value
