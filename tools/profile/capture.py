"""在真实同步调用上采集：同进程scheduler预热，不隐藏重放目标。"""

from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import importlib
import inspect
import json
import os
import runpy
import socket
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path


def loaded_sources():
    """只记录已加载的仓库Python源；不冒充native库内部身份。"""
    root = Path(__file__).resolve().parents[2]
    result = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py" and Path(filename).is_file():
            path = Path(filename).resolve()
            if path.is_relative_to(root):
                result[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


class Capture:
    """call执行一次目标；wait/warmup/active均由入口自己的真实调用推进。

    默认关闭。trace与统计都是诊断，不能作为无开销正式性能计时。
    synchronize只在完整目标边界使用，不在内部kernel之间插屏障。
    """

    def __init__(
        self,
        output,
        *,
        enabled=False,
        skip=0,
        warmup=1,
        count=1,
        trace=True,
        activities=("cpu", "cuda"),
        record_shapes=False,
        synchronize=None,
        workloads=False,
        stats_max_calls=4096,
        moe_distribution=False,
        metadata=None,
    ):
        if any(type(n) is not int for n in (skip, warmup, count, stats_max_calls)) or min(skip, warmup) < 0 or min(count, stats_max_calls) < 1:
            raise ValueError("skip/warmup must be nonnegative; count/stats_max_calls must be positive")
        if not activities or set(activities) - {"cpu", "cuda"}:
            raise ValueError("activities must contain cpu and/or cuda")
        self.output, self.enabled = Path(output), enabled
        self.skip, self.warmup, self.count = skip, warmup if trace else 0, count
        self.trace, self.activities, self.record_shapes = trace, tuple(activities), record_shapes
        self.synchronize, self.workloads = synchronize, workloads
        self.stats_max_calls, self.moe_distribution = stats_max_calls, moe_distribution
        self.metadata = metadata or {}
        self.calls = self.captured = self.failed_calls = 0
        self._profiler = None
        self._entered = self._busy = False
        self._trace_files = []
        self._warmup_completion = None

    def __enter__(self):
        if not self.enabled:
            return self
        if self._entered or self.calls:
            raise RuntimeError("use a new Capture for each session")
        json.dumps(self.metadata)
        if self.workloads:
            from lightx2v.utils.workload_stats import WorkloadStats

            self._stats_class = WorkloadStats
        self.output.mkdir(parents=True, exist_ok=False)
        self._owner = threading.get_ident()
        self._stream = (self.output / "calls.jsonl").open("x", encoding="utf-8")
        self._entered = True
        self._session("running")
        return self

    def _session(self, status):
        value = {
            "kind": "scope_capture",
            "schema_version": 1,
            "status": status,
            "pid": os.getpid(),
            "calls": self.calls,
            "captured": self.captured,
            "failed_calls": self.failed_calls,
            "schedule": {"wait": self.skip, "warmup": self.warmup, "active": self.count},
            "trace": self.trace,
            "activities": list(self.activities),
            "trace_files": self._trace_files,
            "record_shapes": self.record_shapes,
            "workloads": self.workloads,
            "stats_max_calls": self.stats_max_calls,
            "moe_distribution": self.moe_distribution,
            "warmup_completion": self._warmup_completion,
            "boundary_sync": self.synchronize is not None,
            "metadata": self.metadata,
            "loaded_python_sources": loaded_sources(),
            "timing_boundary": "Host target call plus optional trailing sync; profiler/stats are diagnostic. Async GPU completion needs trace correlations or explicit boundary sync.",
        }
        temporary = self.output / "session.json.tmp"
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "session.json")

    def _finish_warmup(self):
        import torch

        if "cuda" in self.activities and torch.cuda.is_available():
            start = time.perf_counter_ns()
            device = torch.cuda.current_device()
            (self.synchronize or torch.cuda.synchronize)()
            self._warmup_completion = {"device": device, "host_ms": (time.perf_counter_ns() - start) / 1e6}

    def _start_profiler(self):
        import torch

        def export(profiler):
            if self.calls <= self.skip + self.warmup:
                return
            name = f"{socket.gethostname()}_{os.getpid()}.{time.time_ns()}.pt.trace.json"
            path = self.output / name
            profiler.export_chrome_trace(str(path))
            value = json.loads(path.read_bytes())
            # TensorBoard旧插件识别runtime分类；保留原分类供离线工具还原。
            for event in value.get("traceEvents", []):
                if event.get("cat") in {"cuda_runtime", "cuda_driver"}:
                    event.setdefault("args", {})["original_category"] = event["cat"]
                    event["cat"] = "runtime"
            value["lightx2v_capture"] = {
                "workloads": self.workloads,
                "record_shapes": self.record_shapes,
                "metadata": self.metadata,
            }
            destination = path.with_suffix(path.suffix + ".gz")
            with gzip.open(destination, "wt", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
            path.unlink()
            self._trace_files.append(destination.name)

        profiler = torch.profiler.profile(
            activities=[getattr(torch.profiler.ProfilerActivity, item.upper()) for item in self.activities],
            record_shapes=self.record_shapes,
            schedule=torch.profiler.schedule(wait=self.skip, warmup=self.warmup, active=self.count, repeat=1),
            on_trace_ready=export,
        )
        profiler.__enter__()
        self._profiler = profiler

    def call(self, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        if not self._entered or self._owner != threading.get_ident() or self._busy:
            raise RuntimeError("Capture requires a live session and nonrecursive calls on its owning thread")
        if not (inspect.isfunction(function) or inspect.ismethod(function)) or any(check(function) for check in (inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction)):
            raise TypeError("target must be an ordinary synchronous Python function or instance method")
        if self.failed_calls and self.trace:
            raise RuntimeError("a failed trace session cannot resume")
        self._busy = True
        self.calls += 1
        offset = self.skip + self.warmup
        phase = "wait" if self.calls <= self.skip else "warmup" if self.calls <= offset else "active" if self.calls <= offset + self.count else "done"
        scope = f"profile::{function.__module__}.{function.__qualname__}"
        row = {"call": self.calls, "scope": scope, "phase": phase, "status": "failed"}
        start = end = None
        try:
            if self.synchronize:
                self.synchronize()
            if self.trace and self._profiler is None:
                if offset == 0:
                    self._finish_warmup()
                self._start_profiler()
            stats = nullcontext()
            if phase == "active" and self.workloads:
                stats = self._stats_class(
                    self.output / f"call_{self.calls:04d}.stats",
                    max_calls=self.stats_max_calls,
                    moe_distribution=self.moe_distribution,
                    trace_ranges=self.trace,
                    provenance={**self.metadata, "scope": scope, "scope_call": self.calls},
                    execution_context=self._execution_context,
                )
            if self.trace and phase in {"warmup", "active"}:
                import torch

                annotation = torch.profiler.record_function(scope)
            else:
                annotation = nullcontext()
            with stats, annotation:
                start = time.perf_counter_ns()
                try:
                    result = function(*args, **kwargs)
                    if self.synchronize:
                        self.synchronize()
                finally:
                    end = time.perf_counter_ns()
            if phase == "active":
                self.captured += 1
            if self._profiler is not None:
                if self.calls == offset:
                    self._finish_warmup()
                self._profiler.step()
            row["status"] = "complete"
            return result
        except BaseException as error:
            self.failed_calls += 1
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["host_ms"] = (end - start) / 1e6 if end is not None else None
            self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._stream.flush()
            self._busy = False

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        cleanup_failed = False
        try:
            if self._profiler is not None:
                self._profiler.__exit__(exc_type, exc, tb)
        except BaseException:
            cleanup_failed = True
            raise
        finally:
            self._stream.close()
            status = "failed" if exc_type or self.failed_calls or cleanup_failed else "complete"
            if status == "complete" and (not self.calls or ((self.trace or self.workloads) and self.captured < self.count)):
                status = "capture_incomplete"
            self._session(status)
            self._entered = False
        return False

    def _execution_context(self):
        import torch

        from tools.profile.library_precision import runtime_settings

        return runtime_settings(torch, "cuda" if "cuda" in self.activities and torch.cuda.is_available() else "cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="module:Class.method 或 module:function")
    parser.add_argument("--output", type=Path, required=True, help="新的TensorBoard logdir")
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="同进程真实调用的scheduler预热次数")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--activities", choices=("cpu", "cpu,cuda"), default="cpu,cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--observe-only", action="store_true", help="只记录host边界；不启动profiler或统计")
    mode.add_argument("--stats-only", action="store_true", help="只采核心计算量；不启动profiler")
    parser.add_argument("--workloads", action="store_true", help="联合统计，诊断开销会影响时间线")
    parser.add_argument("--stats-max-calls", type=int, default=4096, help="每种core backend的样本上限")
    parser.add_argument("--moe-distribution", action="store_true", help="额外采集生产专家分布；稀疏计算量无需此开关")
    parser.add_argument("--record-shapes", action="store_true", help="仅ATen形状调试；默认关闭，不用于计算量")
    parser.add_argument("--synchronize-cuda", action="store_true", help="在整个目标调用前后同步当前设备")
    parser.add_argument("--library-log-dir", type=Path, help="新进程启用库日志；诊断用途")
    parser.add_argument("--metadata", type=Path, help="模型/配置/请求/源码身份JSON对象")
    parser.add_argument("--module", action="store_true", help="入口是Python模块")
    parser.add_argument("entry")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.observe_only and (args.workloads or args.moe_distribution):
        parser.error("observe-only does not collect statistics")
    from tools.profile.library_precision import library_logs

    with library_logs(args.library_log_dir):
        _run_entry(args, parser)
    return 0


def _run_entry(args, parser):
    module, separator, attribute = args.target.partition(":")
    if not separator:
        parser.error("target must be module:attribute")
    previous_path, previous_argv = sys.path, sys.argv
    sys.path = sys.path.copy()
    if not args.module:
        sys.path.insert(0, str(Path(args.entry).resolve().parent))
    owner = original = None
    changed = False
    try:
        owner = importlib.import_module(module)
        parts = attribute.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        name = parts[-1]
        original = inspect.getattr_static(owner, name)
        if not inspect.isfunction(original) or any(check(original) for check in (inspect.iscoroutinefunction, inspect.isasyncgenfunction, inspect.isgeneratorfunction)):
            parser.error("target must be an ordinary synchronous function or instance method")
        entry = args.entry
        if args.module:
            spec = importlib.util.find_spec(entry)
            if spec and spec.submodule_search_locations is not None:
                spec = importlib.util.find_spec(entry + ".__main__")
            entry = spec.origin if spec else None
        owner_module = inspect.getmodule(owner) if inspect.isclass(owner) else owner
        source = getattr(owner_module, "__file__", None)
        if source and entry and Path(source).resolve() == Path(entry).resolve():
            parser.error("target is redefined by the __main__ entry; use a separate driver importing the target, or the Capture API")
        local = name in vars(owner)
        metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
        if not isinstance(metadata, dict):
            parser.error("metadata must be an object")
        if args.library_log_dir:
            metadata["library_log_session"] = str(args.library_log_dir.resolve() / "session.json")
        metadata["target_source"] = {"path": original.__code__.co_filename, "qualname": original.__qualname__}
        synchronize = None
        if args.synchronize_cuda:
            import torch

            synchronize = torch.cuda.synchronize
        with Capture(
            args.output,
            enabled=True,
            skip=args.skip,
            warmup=args.warmup,
            count=args.count,
            trace=not (args.observe_only or args.stats_only),
            activities=args.activities.split(","),
            record_shapes=args.record_shapes,
            synchronize=synchronize,
            workloads=args.workloads or args.stats_only,
            stats_max_calls=args.stats_max_calls,
            moe_distribution=args.moe_distribution,
            metadata=metadata,
        ) as capture:

            @functools.wraps(original)
            def wrapped(*call_args, **call_kwargs):
                return capture.call(original, *call_args, **call_kwargs)

            setattr(owner, name, wrapped)
            changed = True
            sys.argv = [args.entry, *args.args]
            try:
                if args.module:
                    runpy.run_module(args.entry, run_name="__main__", alter_sys=True)
                else:
                    runpy.run_path(args.entry, run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
        if capture.failed_calls or not capture.calls or ((capture.trace or capture.workloads) and capture.captured < capture.count):
            parser.exit(2, f"采集失败或调用不足，查看 {args.output / 'session.json'}\n")
    finally:
        if changed:
            if local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        sys.path, sys.argv = previous_path, previous_argv


if __name__ == "__main__":
    raise SystemExit(main())
