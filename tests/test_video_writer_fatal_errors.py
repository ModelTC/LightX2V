"""Exercise real writer callbacks in a thread without relying on Future polling."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VideoWriterFatalTests(unittest.TestCase):
    def run_writer(self, runner, failure=None, no_event=False):
        code = f"""
import ast, importlib.util, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
kind = {failure!r}
injected = ValueError('invalid frames') if kind == 'value' else RuntimeError(
    'CUDA error: an illegal memory access was encountered' if kind == 'fatal' else 'ordinary failure')
gate = threading.Event()
def sync():
    gate.wait()
    print('SYNC', flush=True)
    if kind:
        raise injected
writes = []
def write(value):
    writes.append(value)
    print('WRITE', flush=True)
def second():
    print('SECOND_CALLBACK', flush=True)
    return 'second'
frames = SimpleNamespace(numpy=lambda: ['frame'])
ns = {{'fatal_device_error_guard': errors.fatal_device_error_guard,
      'av': SimpleNamespace(VideoFrame=SimpleNamespace(from_numpy_buffer=lambda *a, **k: SimpleNamespace()))}}
if {runner!r} == 'relay':
    path, classname, name = 'relayvsr/video_io.py', 'AsyncVideoWriter', '_write'
    writer = SimpleNamespace(container=SimpleNamespace(mux=write),
        stream=SimpleNamespace(encode=lambda _: ['packet']), frames=0, fps=24)
    args = (writer, frames, SimpleNamespace(synchronize=sync))
else:
    path, classname, name = 'swiftvr/swiftvr_runner.py', 'SwiftVRRunner', 'write_video_frames'
    writer = SimpleNamespace(send=write)
    args = (writer, frames, None if {no_event!r} else SimpleNamespace(synchronize=sync))
tree = ast.parse((root / 'lightx2v/models/runners' / path).read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
fn.decorator_list = []
exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)
with ThreadPoolExecutor(max_workers=1) as pool:
    first = pool.submit(ns[name], *args)
    following = pool.submit(second)
    gate.set()
    if kind != 'fatal':
        if kind:
            assert first.exception(timeout=2) is injected
            try:
                first.result(timeout=2)
            except Exception as error:
                assert error is injected
            else:
                raise AssertionError('ordinary exception lost')
            assert writes == []
        else:
            assert first.result(timeout=2) is None
            assert writes == (['packet'] if {runner!r} == 'relay' else ['frame'])
            if {runner!r} == 'relay':
                assert writer.frames == 1
        assert following.result(timeout=2) == 'second'
# Fatal cases deliberately never inspect either Future or call writer.close().
print('DONE', flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def test_fatal_sync_exits_without_future_polling_or_next_callback(self):
        for runner in ("relay", "swift"):
            with self.subTest(runner=runner):
                result = self.run_writer(runner, "fatal")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["SYNC"])
                self.assertIn("LightX2V fatal inference failure", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_ordinary_errors_remain_in_future_and_queue_continues(self):
        for runner in ("relay", "swift"):
            for failure in ("runtime", "value"):
                with self.subTest(runner=runner, failure=failure):
                    result = self.run_writer(runner, failure)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines(), ["SYNC", "SECOND_CALLBACK", "DONE"])
                    self.assertEqual(result.stderr, "")

    def test_success_preserves_write_result_and_queue(self):
        for runner, no_event in (("relay", False), ("swift", False), ("swift", True)):
            with self.subTest(runner=runner, no_event=no_event):
                result = self.run_writer(runner, no_event=no_event)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = ([] if no_event else ["SYNC"]) + ["WRITE", "SECOND_CALLBACK", "DONE"]
                self.assertEqual(result.stdout.splitlines(), expected)
                self.assertEqual(result.stderr, "")
