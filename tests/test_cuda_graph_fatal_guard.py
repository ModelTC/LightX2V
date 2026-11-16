"""Run the real capture method with CPU contexts and real fatal process exits."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CudaGraphGuardTests(unittest.TestCase):
    def run_capture(self, failure=None, fatal=False):
        code = f"""
import ast, importlib.util
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
injected = RuntimeError('CUDA error: an illegal memory access was encountered' if {fatal!r} else 'ordinary failure')
def step(name):
    print(name, flush=True)
    if name == {failure!r}:
        raise injected
class Context:
    def __init__(self, name): self.name = name
    def __enter__(self): step(self.name + '_enter')
    def __exit__(self, *args): step(self.name + '_exit')
class Logits:
    def __getitem__(self, key): return 'logits'
calls = 0
def infer(_):
    global calls
    calls += 1
    step('warmup' if calls == 1 else 'infer')
    return {{'logits': Logits()}}
entry = SimpleNamespace(pre_infer_out=object(), graph=object(), logits=None)
ns = {{'fatal_device_error_guard': errors.fatal_device_error_guard,
      'torch': SimpleNamespace(cuda=SimpleNamespace(
          current_stream=lambda _: SimpleNamespace(wait_stream=lambda _: step('wait')),
          stream=lambda _: Context('stream'), graph=lambda *a, **k: Context('graph'))),
      'dist': SimpleNamespace(get_rank=lambda: 0), 'logger': SimpleNamespace(info=lambda *a: None)}}
tree = ast.parse((root / 'lightx2v/models/runners/hunyuan_image3/cuda_graph.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HunyuanImage3ARCudaGraphController')
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_capture')
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<capture>', 'exec'), ns)
runner = SimpleNamespace(_build_entry=lambda *a: entry, _copy_runtime_inputs=lambda *a: None,
    _pool=object(), _capture_stream=SimpleNamespace(wait_stream=lambda _: None,
        synchronize=lambda: step('sync')), device=None, capture_warmups=1,
    model=SimpleNamespace(infer_ar_prepared=infer), _tp_barrier=lambda: None,
    config={{'parallel_context': SimpleNamespace(custom_all_reduce_capture=lambda: Context('collective'))}},
    _entries={{}})
try:
    returned = ns['_capture'](runner, type('Key', (), dict(cache_capacity=1))(), object(), 1)
    assert returned is entry and entry.logits == 'logits' and runner._entries
    print('RETURN', flush=True)
except Exception as error:
    assert error is injected
    print('RAISE ' + type(error).__name__ + ': ' + str(error), flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def test_fatal_body_context_exit_and_wait_stop_before_outer_cleanup(self):
        for stage, forbidden in (
            ("warmup", "stream_exit"),
            ("infer", "graph_exit"),
            ("graph_enter", "collective_exit"),
            ("graph_exit", "collective_exit"),
            ("wait", "collective_exit"),
        ):
            with self.subTest(stage=stage):
                result = self.run_capture(stage, fatal=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-1], stage)
                self.assertNotIn(forbidden, result.stdout)
                self.assertIn("LightX2V fatal inference failure", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_ordinary_errors_preserve_identity_and_context_cleanup(self):
        full = ["stream_enter", "warmup", "stream_exit", "sync", "collective_enter", "graph_enter", "infer", "graph_exit", "wait", "collective_exit"]
        for stage, cleanup in (
            ("warmup", ["stream_exit"]),
            ("infer", ["graph_exit", "collective_exit"]),
            ("graph_enter", ["collective_exit"]),
            ("graph_exit", ["collective_exit"]),
            ("wait", ["collective_exit"]),
        ):
            with self.subTest(stage=stage):
                result = self.run_capture(stage)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), full[: full.index(stage) + 1] + cleanup + ["RAISE RuntimeError: ordinary failure"])
                self.assertNotIn("LightX2V fatal inference failure", result.stderr)

    def test_success_keeps_return_and_context_order(self):
        result = self.run_capture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["stream_enter", "warmup", "stream_exit", "sync", "collective_enter", "graph_enter", "infer", "graph_exit", "wait", "collective_exit", "RETURN"])
