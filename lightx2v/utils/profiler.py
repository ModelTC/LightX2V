import asyncio
import threading
import time
from contextlib import contextmanager
from functools import wraps

import torch
import torch.distributed as dist
from loguru import logger

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)
_excluded_time_local = threading.local()
_no_sync_local = threading.local()
_memory_profile_local = threading.local()


def _get_excluded_time_stack():
    if not hasattr(_excluded_time_local, "stack"):
        _excluded_time_local.stack = []
    return _excluded_time_local.stack


def _is_profiler_no_sync_active() -> bool:
    return getattr(_no_sync_local, "active", False)


def _get_memory_profile_stack():
    if not hasattr(_memory_profile_local, "stack"):
        _memory_profile_local.stack = []
    return _memory_profile_local.stack


@contextmanager
def no_sync_profiling(enabled: bool = True):
    """Scope where ProfilingContext* skip cuda.synchronize on enter/exit.

    Use around AR segment loops with async VAE so nested profilers (step_pre,
    infer_main, segment end2end, AR segments total) do not insert GPU barriers
    that flatten DiT/VAE overlap. The region must end with explicit sync
    (e.g. async_vae_decoder.finish()).
    """
    if not enabled:
        yield
        return
    prev = getattr(_no_sync_local, "active", False)
    _no_sync_local.active = True
    try:
        yield
    finally:
        _no_sync_local.active = prev


class _ProfilingContext:
    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None, profile_memory=False):
        """
        recorder_mode = 0: disable recorder
        recorder_mode = 1: enable recorder
        recorder_mode = 2: enable recorder and force disable logger
        """
        self.name = name
        self.rank_info = "Single GPU"
        self.enable_recorder = recorder_mode > 0
        self.enable_logger = recorder_mode <= 1
        self.metrics_func = metrics_func
        self.metrics_labels = metrics_labels
        self.profile_memory = profile_memory

    def _record_memory_peak(self, peak_allocated, peak_reserved):
        self.peak_memory_allocated = max(self.peak_memory_allocated, peak_allocated)
        if peak_reserved is not None:
            self.peak_memory_reserved = max(self.peak_memory_reserved or 0, peak_reserved)

    def _read_memory_peak(self):
        peak_allocated = torch_device_module.max_memory_allocated()
        max_memory_reserved = getattr(torch_device_module, "max_memory_reserved", None)
        peak_reserved = max_memory_reserved() if max_memory_reserved is not None else None
        return peak_allocated, peak_reserved

    def _begin_memory_profile(self):
        stack = _get_memory_profile_stack()
        self._memory_profile_active = False
        if not self.profile_memory:
            return

        reset_peak_memory_stats = getattr(torch_device_module, "reset_peak_memory_stats", None)
        max_memory_allocated = getattr(torch_device_module, "max_memory_allocated", None)
        if reset_peak_memory_stats is None or max_memory_allocated is None:
            return

        try:
            if stack:
                peak_allocated, peak_reserved = self._read_memory_peak()
                for context in stack:
                    context._record_memory_peak(peak_allocated, peak_reserved)
            reset_peak_memory_stats()
        except (RuntimeError, NotImplementedError):
            return

        memory_allocated = getattr(torch_device_module, "memory_allocated", None)
        try:
            self.start_memory_allocated = memory_allocated() if memory_allocated is not None else 0
        except (RuntimeError, NotImplementedError):
            self.start_memory_allocated = 0
        self.peak_memory_allocated = 0
        self.peak_memory_reserved = None
        self._memory_profile_active = True
        stack.append(self)

    def _end_memory_profile(self):
        if not self._memory_profile_active:
            return ""

        stack = _get_memory_profile_stack()
        try:
            peak_allocated, peak_reserved = self._read_memory_peak()
            for context in stack:
                context._record_memory_peak(peak_allocated, peak_reserved)
            torch_device_module.reset_peak_memory_stats()
        except (RuntimeError, NotImplementedError):
            pass
        finally:
            if stack and stack[-1] is self:
                stack.pop()
            elif self in stack:
                stack.remove(self)
            self._memory_profile_active = False

        peak_allocated = self.peak_memory_allocated
        peak_reserved = self.peak_memory_reserved
        start_allocated = self.start_memory_allocated
        peak_delta = max(peak_allocated - start_allocated, 0)
        gib = 1024**3
        memory_suffix = f", start_allocated {start_allocated / gib:.3f} GiB, peak_allocated {peak_allocated / gib:.3f} GiB, peak_delta {peak_delta / gib:.3f} GiB"
        if peak_reserved is not None:
            memory_suffix += f", peak_reserved {peak_reserved / gib:.3f} GiB"
        return memory_suffix

    def __enter__(self):
        self.rank_info = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Single GPU"
        self._skip_sync = _is_profiler_no_sync_active()
        if not self._skip_sync:
            torch_device_module.synchronize()
        self._begin_memory_profile()
        self.start_time = time.perf_counter()
        _get_excluded_time_stack().append(0.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._skip_sync:
            torch_device_module.synchronize()
        total_elapsed = time.perf_counter() - self.start_time
        memory_suffix = self._end_memory_profile()
        excluded = _get_excluded_time_stack().pop()
        elapsed = total_elapsed - excluded
        if self.enable_recorder and self.metrics_func:
            if self.metrics_labels:
                self.metrics_func.labels(*self.metrics_labels).observe(elapsed)
            else:
                self.metrics_func.observe(elapsed)
        if self.enable_logger:
            suffix = " (non-sync)" if self._skip_sync else ""
            logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds{memory_suffix}{suffix}")
        return False

    async def __aenter__(self):
        self.rank_info = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Single GPU"
        self._skip_sync = _is_profiler_no_sync_active()
        if not self._skip_sync:
            torch_device_module.synchronize()
        self._begin_memory_profile()
        self.start_time = time.perf_counter()
        _get_excluded_time_stack().append(0.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._skip_sync:
            torch_device_module.synchronize()
        total_elapsed = time.perf_counter() - self.start_time
        memory_suffix = self._end_memory_profile()
        excluded = _get_excluded_time_stack().pop()
        elapsed = total_elapsed - excluded
        if self.enable_recorder and self.metrics_func:
            if self.metrics_labels:
                self.metrics_func.labels(*self.metrics_labels).observe(elapsed)
            else:
                self.metrics_func.observe(elapsed)
        if self.enable_logger:
            suffix = " (non-sync)" if self._skip_sync else ""
            logger.info(f"[Profile] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds{memory_suffix}{suffix}")
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _NullContext:
    # Context manager without decision branch logic overhead
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __call__(self, func):
        return func


class _ExcludedProfilingContext:
    """用于标记应该从外层 profiling 中排除的时间段"""

    def __init__(self, name=None):
        self.name = name
        self.rank_info = "Single GPU"

    def __enter__(self):
        self.rank_info = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Single GPU"
        self._skip_sync = _is_profiler_no_sync_active()
        if not self._skip_sync:
            torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._skip_sync:
            torch_device_module.synchronize()
        elapsed = time.perf_counter() - self.start_time
        stack = _get_excluded_time_stack()
        for i in range(len(stack)):
            stack[i] += elapsed
        if self.name and CHECK_PROFILING_DEBUG_LEVEL(1):
            suffix = " (non-sync)" if self._skip_sync else ""
            logger.info(f"[Profile-Excluded] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds (excluded from outer profiling){suffix}")
        return False

    async def __aenter__(self):
        self.rank_info = f"Rank {dist.get_rank()}" if dist.is_initialized() else "Single GPU"
        self._skip_sync = _is_profiler_no_sync_active()
        if not self._skip_sync:
            torch_device_module.synchronize()
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._skip_sync:
            torch_device_module.synchronize()
        elapsed = time.perf_counter() - self.start_time
        stack = _get_excluded_time_stack()
        for i in range(len(stack)):
            stack[i] += elapsed
        if self.name and CHECK_PROFILING_DEBUG_LEVEL(1):
            suffix = " (non-sync)" if self._skip_sync else ""
            logger.info(f"[Profile-Excluded] {self.rank_info} - {self.name} cost {elapsed:.6f} seconds (excluded from outer profiling){suffix}")
        return False

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper


class _ProfilingContextL1(_ProfilingContext):
    """Level 1 profiling context with Level1_Log prefix."""

    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None, profile_memory=False):
        super().__init__(f"Level1_Log {name}", recorder_mode, metrics_func, metrics_labels, profile_memory)


class _ProfilingContextL2(_ProfilingContext):
    """Level 2 profiling context with Level2_Log prefix."""

    def __init__(self, name, recorder_mode=0, metrics_func=None, metrics_labels=None, profile_memory=False):
        super().__init__(f"Level2_Log {name}", recorder_mode, metrics_func, metrics_labels, profile_memory)


"""
PROFILING_DEBUG_LEVEL=0: [Default] disable all profiling
PROFILING_DEBUG_LEVEL=1: enable ProfilingContext4DebugL1
PROFILING_DEBUG_LEVEL=2: enable ProfilingContext4DebugL1 and ProfilingContext4DebugL2
"""
ProfilingContext4DebugL1 = _ProfilingContextL1 if CHECK_PROFILING_DEBUG_LEVEL(1) else _NullContext  # if user >= 1, enable profiling
ProfilingContext4DebugL2 = _ProfilingContextL2 if CHECK_PROFILING_DEBUG_LEVEL(2) else _NullContext  # if user >= 2, enable profiling
ExcludedProfilingContext = _ExcludedProfilingContext if CHECK_PROFILING_DEBUG_LEVEL(1) else _NullContext
