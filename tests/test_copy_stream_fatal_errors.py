"""Keep real copy methods and observable stream exits in subprocess regressions."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CopyStreamFatalTests(unittest.TestCase):
    def run_copy(self, runner, failure=None, fatal=False, reverse=False):
        code = f"""
import ast, importlib.util
from collections import deque
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
injected = RuntimeError('CUDA error: an illegal memory access was encountered' if {fatal!r} else 'ordinary failure')
def step(name):
    print(name, flush=True)
    if name == {failure!r}: raise injected
class StreamContext:
    def __enter__(self): step('ENTER')
    def __exit__(self, kind, error, traceback): step('EXIT')
host = SimpleNamespace(copy_=lambda *a, **k: step('COPY'))
event = SimpleNamespace(record=lambda *a: step('EVENT_RECORD'))
def make_event():
    step('EVENT_CREATE')
    return event
stream = SimpleNamespace(wait_stream=lambda _: step('WAIT'))
frames = SimpleNamespace(shape=(1, 2, 2, 3), dtype=None, device='cuda',
    record_stream=lambda _: step('RECORD'))
ns = {{'fatal_device_error_guard': errors.fatal_device_error_guard,
      'torch': SimpleNamespace(empty=lambda *a, **k: host, uint8=None,
          cuda=SimpleNamespace(current_stream=lambda _: None, stream=lambda _: StreamContext(), Event=make_event))}}
if {runner!r} == 'relay':
    path, classname, name = 'relayvsr/video_io.py', 'AsyncVideoWriter', 'submit'
else:
    path, classname, name = 'swiftvr/swiftvr_runner.py', 'SwiftVRRunner', 'copy_frames_to_cpu'
tree = ast.parse((root / 'lightx2v/models/runners' / path).read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
fn.decorator_list = []
if {reverse!r}:
    context = next(n for n in ast.walk(fn) if isinstance(n, ast.With))
    assert len(context.items) == 2
    context.items.reverse()
exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)
pending = deque()
sentinel = object()
def submit(*a):
    step('SUBMIT')
    return sentinel
writer = SimpleNamespace(pending=pending, queue_size=1, copy_stream=stream,
    executor=SimpleNamespace(submit=submit), _write=lambda *a: None)
try:
    if {runner!r} == 'relay':
        result = ns[name](writer, frames)
        assert result is None and list(pending) == [sentinel]
    else:
        result = ns[name](frames, stream)
        assert result == (host, event)
    print('RETURN', flush=True)
except Exception as error:
    assert error is injected
    print('RAISE ordinary failure', flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def assert_valid_fatal_exit(self, result):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("RETURN", result.stdout)
        self.assertNotIn("RAISE", result.stdout)

    def assert_before_stream_exit(self, result):
        self.assert_valid_fatal_exit(result)
        self.assertNotIn("EXIT", result.stdout.splitlines(), "stream exited before fatal termination")
        self.assertNotIn("SUBMIT", result.stdout)

    def test_fatal_copy_operations_skip_stream_exit_and_submission(self):
        for runner in ("relay", "swift"):
            for stage in ("COPY", "RECORD", "EVENT_CREATE", "EVENT_RECORD"):
                with self.subTest(runner=runner, stage=stage):
                    result = self.run_copy(runner, stage, fatal=True)
                    self.assert_before_stream_exit(result)
                    self.assertEqual(result.stdout.splitlines()[-1], stage)

    def test_ordinary_errors_keep_identity_and_stream_exit(self):
        prefix = ["WAIT", "ENTER", "COPY", "RECORD", "EVENT_CREATE", "EVENT_RECORD"]
        for runner in ("relay", "swift"):
            for stage in prefix[2:]:
                with self.subTest(runner=runner, stage=stage):
                    result = self.run_copy(runner, stage)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), prefix[: prefix.index(stage) + 1] + ["EXIT", "RAISE ordinary failure"])
                    self.assertEqual(result.stderr, "")

    def test_success_preserves_stream_exit_and_return(self):
        for runner in ("relay", "swift"):
            with self.subTest(runner=runner):
                result = self.run_copy(runner)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["WAIT", "ENTER", "COPY", "RECORD", "EVENT_CREATE", "EVENT_RECORD", "EXIT"] + (["SUBMIT"] if runner == "relay" else []) + ["RETURN"])
                self.assertEqual(result.stderr, "")

    def test_reversed_context_order_is_rejected_by_exit_assertion(self):
        for runner in ("relay", "swift"):
            with self.subTest(runner=runner):
                result = self.run_copy(runner, "COPY", fatal=True, reverse=True)
                self.assert_valid_fatal_exit(result)
                self.assertEqual(result.stdout.splitlines(), ["WAIT", "ENTER", "COPY", "EXIT"])
                with self.assertRaisesRegex(AssertionError, "stream exited before fatal termination"):
                    self.assert_before_stream_exit(result)
