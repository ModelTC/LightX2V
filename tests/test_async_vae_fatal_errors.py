"""Upstream AR worker failures must terminate before context exit or Future polling."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class AsyncVAEFatalTests(unittest.TestCase):
    def run_callback(self, method, fatal):
        code = f"""
import ast, __future__, importlib.util, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
injected = RuntimeError('CUDA error: an illegal memory access was encountered' if {fatal!r} else 'ordinary failure')
gate = threading.Event()
def fail(*args):
    gate.wait()
    print('FAIL', flush=True)
    raise injected
class Context:
    def __enter__(self): pass
    def __exit__(self, *args): print('CONTEXT_EXIT', flush=True)
event = SimpleNamespace(record=lambda *a: None)
ns = {{'fatal_device_error_guard': errors.fatal_device_error_guard,
      'torch': SimpleNamespace(cuda=SimpleNamespace(device=lambda *a: Context(),
          stream=lambda *a: Context(), Event=lambda **kw: event), no_grad=lambda: Context())}}
tree = ast.parse((root / 'lightx2v/utils/async_vae.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncVAEChunkDecoder')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == {method!r})
# Keep the real fatal guard decorator and inner context order.
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<vae>', 'exec', flags=__future__.annotations.compiler_flag), ns)
runner = SimpleNamespace(_stream=SimpleNamespace(device='cuda', wait_event=lambda *a: None), _on_chunk=lambda *a: None)
if {method!r} == '_deliver_chunk':
    args = (runner, SimpleNamespace(device='cuda'), 0, SimpleNamespace(synchronize=fail))
else:
    args = (runner, fail, (), {{}}, object(), 0)
def second():
    print('SECOND', flush=True)
with ThreadPoolExecutor(max_workers=1) as pool:
    first = pool.submit(ns[fn.name], *args)
    following = pool.submit(second)
    gate.set()
    if not {fatal!r}:
        assert first.exception(timeout=2) is injected
        following.result(timeout=2)
print('DONE', flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def test_fatal_workers_exit_before_cleanup_and_next_callback(self):
        for method in ("_deliver_chunk", "_decode_chunk"):
            with self.subTest(method=method):
                result = self.run_callback(method, True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["FAIL"])
                self.assertIn("LightX2V fatal inference failure", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_ordinary_workers_preserve_exception_and_queue(self):
        for method in ("_deliver_chunk", "_decode_chunk"):
            with self.subTest(method=method):
                result = self.run_callback(method, False)
                self.assertEqual(result.returncode, 0, result.stderr)
                cleanup = ["CONTEXT_EXIT"] * 3 if method == "_decode_chunk" else []
                self.assertEqual(result.stdout.splitlines(), ["FAIL"] + cleanup + ["SECOND", "DONE"])
                self.assertEqual(result.stderr, "")

    def test_prepare_failure_checks_fatal_before_restoring_weights(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal):
                code = f"""
import ast, __future__, importlib.util
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
injected = RuntimeError('CUDA error: an illegal memory access was encountered' if {fatal!r} else 'ordinary failure')
def fail():
    print('TRANSFER', flush=True)
    raise injected
ns = {{'terminate_on_fatal_device_error': errors.terminate_on_fatal_device_error,
      'logger': SimpleNamespace(info=lambda *a: None),
      'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))}}
tree = ast.parse((root / 'lightx2v/utils/async_vae.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AsyncVAEChunkDecoder')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_vae')
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<prepare>', 'exec', flags=__future__.annotations.compiler_flag), ns)
runner = SimpleNamespace(enabled=False, _device=None, _restore_vae=lambda: print('RESTORE', flush=True))
try:
    ns[fn.name](runner, SimpleNamespace(cpu_offload=True, to_cuda=fail))
except Exception as error:
    assert error is injected
    print('RAISE', flush=True)
"""
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 1 if fatal else 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), ["TRANSFER"] if fatal else ["TRANSFER", "RESTORE", "RAISE"])
                self.assertNotIn("Traceback", result.stderr)
                if fatal:
                    self.assertIn("LightX2V fatal inference failure", result.stderr)
                else:
                    self.assertEqual(result.stderr, "")
