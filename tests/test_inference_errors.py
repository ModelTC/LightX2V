"""CPU-only regression tests; subprocesses exercise the real fail-fast exit."""

import ast
import asyncio
import importlib.util
import os
import subprocess
import sys
import threading
import time
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "lightx2v/utils/inference_errors.py"
spec = importlib.util.spec_from_file_location("inference_errors", MODULE)
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)


class FatalDeviceErrorTests(unittest.TestCase):
    def test_guard_preserves_ordinary_identity_suppression_and_base_exceptions(self):
        for error in (ValueError("invalid input"), RuntimeError("ordinary failure"), KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as caught:
                    with errors.fatal_device_error_guard():
                        raise error
                self.assertIs(caught.exception, error)
        observed = []

        class Suppressor:
            def __enter__(self):
                return self

            def __exit__(self, kind, error, traceback):
                observed.append(error)
                return True

        error = ValueError("suppressed by original context")
        with Suppressor(), errors.fatal_device_error_guard():
            raise error
        self.assertEqual(observed, [error])

    def test_exit_survives_closed_stderr(self):
        code = f"""
import importlib.util, os
spec = importlib.util.spec_from_file_location('inference_errors', {str(MODULE)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
os.close(2)
m.terminate_inference_process('fatal device error')
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 1)

    def test_fatal_errors_and_recoverable_errors(self):
        for message in (
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: device-side assert triggered",
            "CUDA error: the launch timed out and was terminated",
            "CUDA error: unspecified launch failure",
        ):
            self.assertTrue(errors.is_fatal_device_error(RuntimeError(message)), message)
        for message in ("MUSA out of memory", "invalid image dimensions", "Task cancelled", "file not found"):
            self.assertFalse(errors.is_fatal_device_error(RuntimeError(message)), message)


# Expected Python error forms for the documented CUDA/Moore Threads faults.
DOCUMENTED_FATAL_MESSAGES = (
    "CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error",
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: the launch timed out and was terminated",
    "CUDA error: device-side assert triggered",
    "CUDA error: hardware stack error",
    "CUDA error: an illegal instruction was encountered",
    "CUDA error: misaligned address",
    "CUDA error: operation not supported on global/shared address space",
    "CUDA error: invalid program counter",
    "CUDA error: unspecified launch failure",
    "CUDA error: tensor memory not completely freed",
    "CUDA error: the MPS client has been terminated by the server",
    "CUDA error: an async error has occured in external entity outside of CUDA",
    "MUSA error: GPU Initialization Failure",
    "MUSA error: GPU hardware failure",
    "MUSA error: Uncorrectable uncontained ECC error",
    "MUSA error: ECC error counter overflow",
    "MUSA error: Page retire failed",
    "MUSA error: AXI Error",
    "MUSA error: AXI bus error",
    "MUSA error: GPU bios trap",
    "MUSA error: GPU shutdown",
    "MUSA error: Mtlink unrecoverable error",
)


class FatalWhitelistTests(unittest.TestCase):
    def setUp(self):
        unexpected_exit = patch.object(errors, "terminate_inference_process", side_effect=AssertionError("unexpected process exit"))
        unexpected_exit.start()
        self.addCleanup(unexpected_exit.stop)

    def test_whitelisted_messages_really_exit_in_subprocesses(self):
        for message in DOCUMENTED_FATAL_MESSAGES:
            with self.subTest(message=message):
                code = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('errors', {str(MODULE)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.terminate_on_fatal_device_error(RuntimeError({message!r}))
print('UNEXPECTED_RETURN')
"""
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                self.assertIn("LightX2V fatal inference failure", result.stderr)
                self.assertNotIn("UNEXPECTED_RETURN", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_confirmed_device_errors_and_multiline_diagnostics(self):
        for message in DOCUMENTED_FATAL_MESSAGES:
            for suffix in ("", "\nCUDA kernel errors might be asynchronously reported."):
                with self.subTest(message=message, suffix=suffix):
                    self.assertTrue(errors.is_fatal_device_error(RuntimeError(message + suffix)))

    def test_ambiguous_unconfirmed_and_ordinary_messages_do_not_exit(self):
        for message in (
            "CUDA error: context is destroyed",
            "CUDA error: context has been destroyed",
            "CUDA error: GPU page fault",
            "GPU Page fault or AXI bus error",
            "MUSA error: GPU Page fault or AXI bus error",
            "MUSA error: an illegal memory access was encountered",
            "MUSA error: device-side assert triggered",
            "MUSA error: unspecified launch failure",
            "MUSA error: the launch timed out and was terminated",
            "MUSA error: Uncorrectable contained ECC error",
            "MUSA error: Single Bit ECC Error",
            "MUSA error: GPU memory page fault",
            "MUSA error: Mtlink down",
            "MUSA error: GPU run timeout",
            "MUSA error: FW communication failure",
            "MUSA error: Musa processor exception",
            "CUDA error: external device error",
            "CUDA error: an error has occurred in a device outside of GPU",
            "CUDA error: operation not supported",
            "CUDA error: the resource is already mapped",
            "CUDA error: out of memory",
            "CUDA error: too many resources requested for launch",
            "Request launch timed out after 600s",
            "illegal memory access",
            "AXI bus error",
            "file not found: CUDA error: an illegal memory access was encountered",
            "Input contains 'CUDA error: device-side assert triggered'",
            "warning\nCUDA error: unspecified launch failure",
            "CUDA error: unspecified launch failure or invalid input",
        ):
            with self.subTest(message=message):
                # Exercise the production handler, not only the classifier.
                errors.terminate_on_fatal_device_error(RuntimeError(message))
                self.assertFalse(errors.is_fatal_device_error(RuntimeError(message)))

    def test_validation_and_io_errors_with_device_text_do_not_exit(self):
        for cls in (ValueError, OSError):
            for message in ("CUDA error: device-side assert triggered", "MUSA error: AXI bus error"):
                with self.subTest(cls=cls, message=message):
                    error = cls(message)
                    errors.terminate_on_fatal_device_error(error)
                    self.assertFalse(errors.is_fatal_device_error(error))

    def test_wrapped_ambiguous_error_remains_nonfatal(self):
        wrapper = RuntimeError("request failed")
        wrapper.__cause__ = RuntimeError("GPU Page fault or AXI bus error")
        self.assertFalse(errors.is_fatal_device_error(wrapper))
        errors.terminate_on_fatal_device_error(wrapper)


class WorkerIntegrationTests(unittest.TestCase):
    def worker_class(self, terminate):
        # Load the real worker class with CPU stubs for model/GPU imports.
        path = MODULE.parents[1] / "server/services/inference/worker.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TorchrunInferenceWorker")
        namespace = dict(
            asyncio=asyncio,
            os=os,
            time=time,
            Path=Path,
            Any=object,
            Dict=dict,
            logger=SimpleNamespace(exception=lambda *a: None),
            is_fatal_device_error=errors.is_fatal_device_error,
            terminate_inference_process=terminate,
            terminate_on_fatal_device_error=lambda error: terminate(str(error)) if errors.is_fatal_device_error(error) else None,
        )
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
        return namespace["TorchrunInferenceWorker"]

    def test_initialization_does_not_swallow_fatal_runner_errors(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal):

                def terminate(reason):
                    raise SystemExit(1)

                cls = self.worker_class(terminate)
                worker = cls.__new__(cls)
                worker.world_size, worker.rank = 1, 0
                worker.dist_manager = SimpleNamespace()
                ns = cls.init.__globals__
                ns["torch"] = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
                ns["logger"] = SimpleNamespace(info=lambda *a: None, exception=lambda *a: None)
                ns["build_startup_config"] = lambda *a: {"parallel": False}

                def build_runner(_):
                    raise RuntimeError("CUDA error: an illegal memory access was encountered" if fatal else "invalid model")

                ns["build_runner"] = build_runner
                args = SimpleNamespace(config_json=None, model_cls="test", model_variant=None, model_path="unused", task="t2i")
                if fatal:
                    with self.assertRaises(SystemExit):
                        worker.init(args)
                else:
                    self.assertFalse(worker.init(args))

    def test_fatal_worker_error_exits_before_distributed_barrier(self):
        calls = []

        def terminate(reason):
            calls.append(reason)
            raise SystemExit(1)

        cls = self.worker_class(terminate)
        worker = cls.__new__(cls)
        worker.rank, worker.world_size, worker.lora_dir = 1, 2, None

        def fail(_):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        worker.runner = SimpleNamespace(prepare_request=lambda data: data, set_reuse=lambda *a: None, run_request=fail)
        worker.dist_manager = SimpleNamespace(barrier=lambda: self.fail("fatal context must not enter barrier"))
        with self.assertRaises(SystemExit):
            asyncio.run(worker.process_request({"task_id": "fatal"}))
        self.assertIn("illegal memory access", calls[0])

    def test_ordinary_worker_error_remains_a_failed_task(self):
        cls = self.worker_class(lambda reason: self.fail(reason))
        worker = cls.__new__(cls)
        worker.rank, worker.world_size, worker.lora_dir = 0, 1, None

        def fail(_):
            raise RuntimeError("MUSA out of memory")

        worker.runner = SimpleNamespace(prepare_request=lambda data: data, set_reuse=lambda *a: None, run_request=fail)
        result = asyncio.run(worker.process_request({"task_id": "oom"}))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "RuntimeError")


class ExceptionChainTests(unittest.TestCase):
    def test_fatal_cause_and_context_are_recognized(self):
        for attribute in ("__cause__", "__context__"):
            with self.subTest(attribute=attribute):
                wrapper = RuntimeError("secondary synchronization error")
                setattr(wrapper, attribute, RuntimeError("CUDA error: an illegal memory access was encountered"))
                self.assertTrue(errors.is_fatal_device_error(wrapper))

    def test_cycles_and_ordinary_chains_are_not_fatal(self):
        first, second = RuntimeError("invalid input"), RuntimeError("out of memory")
        first.__cause__, second.__context__ = second, first
        self.assertFalse(errors.is_fatal_device_error(first))
        self.assertFalse(errors.is_fatal_device_error(None))


class ProfilingRegressionTests(unittest.TestCase):
    def profiler_classes(self, events):
        # Execute the actual context classes and stack helpers with CPU device stubs.
        path = MODULE.parent / "profiler.py"
        tree = ast.parse(path.read_text())
        names = {
            "_ProfilingContext",
            "_ExcludedProfilingContext",
            "_get_excluded_time_stack",
            "_get_memory_profile_stack",
            "_is_profiler_no_sync_active",
        }
        body = [n for n in tree.body if getattr(n, "name", None) in names]
        ns = dict(
            asyncio=asyncio,
            time=time,
            wraps=wraps,
            is_fatal_device_error=errors.is_fatal_device_error,
            _excluded_time_local=threading.local(),
            _memory_profile_local=threading.local(),
            _no_sync_local=threading.local(),
            torch_device_module=SimpleNamespace(synchronize=lambda: events.append("sync")),
            dist=SimpleNamespace(is_initialized=lambda: False),
            logger=SimpleNamespace(info=lambda *a: None),
            CHECK_PROFILING_DEBUG_LEVEL=lambda *a: False,
        )
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
        return ns

    def test_fatal_sync_and_async_exit_skip_device_and_memory_access(self):
        for name in ("_ProfilingContext", "_ExcludedProfilingContext"):
            for asynchronous in (False, True):
                with self.subTest(name=name, asynchronous=asynchronous):
                    events = []
                    ns = self.profiler_classes(events)
                    context = ns[name]("test")

                    def raise_fatal():
                        if name == "_ProfilingContext":
                            ns["_get_memory_profile_stack"]().append(context)
                            context._memory_profile_active = True
                            context._end_memory_profile = lambda: self.fail("device memory access")
                        raise fatal

                    fatal = RuntimeError("CUDA error: an illegal memory access was encountered")
                    if asynchronous:

                        async def run():
                            async with context:
                                raise_fatal()

                        def invoke():
                            asyncio.run(run())
                    else:

                        def invoke():
                            with context:
                                raise_fatal()

                    events.clear()
                    with self.assertRaises(RuntimeError) as caught:
                        invoke()
                    self.assertIs(caught.exception, fatal)
                    self.assertEqual(events, ["sync"])
                    if name == "_ProfilingContext":
                        self.assertEqual(ns["_get_memory_profile_stack"](), [])
                        self.assertEqual(ns["_get_excluded_time_stack"](), [])

    def test_profiled_worker_exits_before_sync_or_barrier(self):
        events = []
        ns = self.profiler_classes(events)

        def terminate(reason):
            events.append("exit")
            raise SystemExit(1)

        cls = WorkerIntegrationTests().worker_class(terminate)
        worker = cls.__new__(cls)
        worker.rank, worker.world_size, worker.lora_dir = 1, 2, None

        @ns["_ProfilingContext"]("pipeline")
        def run_request(_):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        worker.runner = SimpleNamespace(prepare_request=lambda data: data, set_reuse=lambda *a: None, run_request=run_request)
        worker.dist_manager = SimpleNamespace(barrier=lambda: self.fail("fatal error entered barrier"))
        with self.assertRaises(SystemExit):
            asyncio.run(worker.process_request({"task_id": "profiled-fatal"}))
        self.assertEqual(events, ["sync", "exit"])
        self.assertEqual(ns["_get_excluded_time_stack"](), [])

    def test_normal_and_ordinary_exception_still_synchronize(self):
        for name in ("_ProfilingContext", "_ExcludedProfilingContext"):
            for asynchronous in (False, True):
                for fail in (False, True):
                    with self.subTest(name=name, asynchronous=asynchronous, fail=fail):
                        events = []
                        ns = self.profiler_classes(events)
                        context = ns[name]("test")
                        ordinary = ValueError("invalid dimensions")

                        def run():
                            with context:
                                if fail:
                                    raise ordinary

                        async def arun():
                            async with context:
                                if fail:
                                    raise ordinary

                        if fail:
                            with self.assertRaises(ValueError) as caught:
                                asyncio.run(arun()) if asynchronous else run()
                            self.assertIs(caught.exception, ordinary)
                        else:
                            asyncio.run(arun()) if asynchronous else run()
                        self.assertEqual(events, ["sync", "sync"])


class ImageEncodingRegressionTests(unittest.TestCase):
    def encoder(self, convert, terminate):
        path = MODULE.parents[1] / "server/services/inference/pipeline_image_encode.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_pipeline_return_to_png_bytes")
        ns = dict(
            Any=Any,
            Optional=Optional,
            _first_image=lambda x: x,
            _image_to_png_bytes=convert,
            is_fatal_device_error=errors.is_fatal_device_error,
            terminate_inference_process=terminate,
            terminate_on_fatal_device_error=lambda error: terminate(str(error)) if errors.is_fatal_device_error(error) else None,
            logger=SimpleNamespace(exception=lambda *a: None),
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        return ns[fn.name]

    def test_fatal_encoding_error_terminates_instead_of_returning_none(self):
        for chained in (False, True):
            with self.subTest(chained=chained):
                fatal = RuntimeError("CUDA error: an illegal memory access was encountered")

                def convert(_):
                    if chained:
                        raise RuntimeError("copy failed") from fatal
                    raise fatal

                reasons = []

                def terminate(reason):
                    reasons.append(reason)
                    raise SystemExit(1)

                with self.assertRaises(SystemExit):
                    self.encoder(convert, terminate)(object())
                self.assertEqual(len(reasons), 1)

    def test_ordinary_encoding_error_and_success_keep_existing_behavior(self):
        def ordinary(_):
            raise ValueError("invalid image")

        def terminate(reason):
            self.fail(reason)

        self.assertIsNone(self.encoder(ordinary, terminate)(object()))
        self.assertEqual(self.encoder(lambda _: b"png", terminate)(object()), b"png")


class CleanupAndCommunicationTests(unittest.TestCase):
    def test_fatal_barrier_exits_on_both_ranks(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                calls = []

                def terminate(reason):
                    calls.append(reason)
                    raise SystemExit(1)

                cls = WorkerIntegrationTests().worker_class(terminate)
                worker = cls.__new__(cls)
                worker.rank, worker.world_size, worker.lora_dir = rank, 2, None
                worker.runner = SimpleNamespace(prepare_request=lambda x: x, set_reuse=lambda *a: None, run_request=lambda x: None)

                def barrier():
                    raise RuntimeError("CUDA error: an illegal memory access was encountered")

                worker.dist_manager = SimpleNamespace(broadcast_task_data=lambda x: x, barrier=barrier)
                with self.assertRaises(SystemExit):
                    asyncio.run(worker.process_request({"task_id": "barrier-fatal"}))
                self.assertEqual(len(calls), 1)

    def test_worker_loop_fatal_error_does_not_retry_barrier(self):
        for source in ("broadcast", "request", "recovery_barrier"):
            with self.subTest(source=source):
                calls = []

                def terminate(reason):
                    calls.append("exit")
                    raise SystemExit(1)

                cls = WorkerIntegrationTests().worker_class(terminate)
                worker = cls.__new__(cls)
                worker.rank, worker.world_size = 1, 2
                fatal = RuntimeError("CUDA error: an illegal memory access was encountered")

                def broadcast():
                    if source == "broadcast":
                        raise fatal
                    return {"task_id": "loop-fatal"}

                async def request(_):
                    if source == "recovery_barrier":
                        raise ValueError("ordinary error")
                    raise fatal

                def barrier():
                    calls.append("barrier")
                    if source == "recovery_barrier":
                        raise fatal
                    self.fail("fatal exception must not retry barrier")

                worker.dist_manager = SimpleNamespace(broadcast_task_data=broadcast, barrier=barrier)
                worker.process_request = request
                cls.worker_loop.__globals__["logger"] = SimpleNamespace(error=lambda *a: None)
                with self.assertRaises(SystemExit):
                    asyncio.run(worker.worker_loop())
                self.assertEqual(calls, ["barrier", "exit"] if source == "recovery_barrier" else ["exit"])

    def test_real_qwen_cleanup_is_bypassed_only_for_fatal_errors(self):
        # Execute the actual local pipeline and inherited cleanup in subprocesses.
        for message in ("CUDA error: an illegal memory access was encountered", "invalid dimensions"):
            with self.subTest(message=message):
                code = f"""
import ast, importlib.util, sys
from pathlib import Path
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('errors', {str(MODULE)!r})
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
root = Path({str(MODULE.parents[1])!r})
ns = {{'terminate_on_fatal_device_error': errors.terminate_on_fatal_device_error,
      'logger': SimpleNamespace(info=lambda *a: None)}}
def method(path, clsname, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == clsname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[name]
end = method(root / 'models/runners/default_runner.py', 'DefaultRunner', 'end_run')
cache = method(root / 'models/runners/default_runner.py', 'DefaultRunner', 'maybe_empty_cache')
run = method(root / 'models/runners/qwen_image/qwen_image_runner.py', 'QwenImageRunner', '_run_pipeline_local')
def memory():
    print('DEVICE_CLEANUP', file=sys.stderr)
    return 10**12, 10**12
ns['AI_DEVICE'] = 'cuda'
ns['torch_device_module'] = SimpleNamespace(mem_get_info=memory)
r = SimpleNamespace(config={{'task': 't2i'}}, model=None, reuse=False, input_info=object(),
    prepare_reuse_output=lambda: None, run_input_encoder=lambda: {{}}, stage_reuse_cache=lambda: None,
    set_latent_shape=lambda: None, discard_reuse_result=lambda: None)
r.end_run = lambda: end(r)
r.maybe_empty_cache = lambda **kw: cache(r, **kw)
def fail():
    raise RuntimeError({message!r})
r.run_dit = fail
run(r, object())
"""
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 1)
                if "illegal memory access" in message:
                    self.assertIn("LightX2V fatal inference failure", result.stderr)
                    self.assertNotIn("DEVICE_CLEANUP", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                else:
                    self.assertIn("DEVICE_CLEANUP", result.stderr)
                    self.assertIn("invalid dimensions", result.stderr)


if __name__ == "__main__":
    unittest.main()
