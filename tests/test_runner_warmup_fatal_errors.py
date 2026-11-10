"""Exercise the complete AR warmup method, including its RNG context."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WarmupFatalTests(unittest.TestCase):
    def run_warmup(self, stage, fatal, mutation=False):
        code = r"""
import ast
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path
root, stage, fatal, mutation = sys.argv[1:]
fatal, mutation = fatal == 'True', mutation == 'True'
spec = importlib.util.spec_from_file_location('errors', Path(root) / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
error = RuntimeError('CUDA error: an illegal memory access was encountered' if fatal else 'ordinary failure')
def step(name):
    print(name, flush=True)
    if name == stage:
        raise error
class RNG:
    def __enter__(self): step('enter')
    def __exit__(self, *args): step('exit')
def device():
    step('device')
    return 0
def rng(**kwargs):
    step('construct')
    return RNG()
tree = ast.parse((Path(root) / 'lightx2v/models/runners/autoregressive_runner.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AutoregressiveRunnerMixin')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run_warmup')
method.decorator_list = []
if mutation:
    outer = next(n for n in method.body if isinstance(n, ast.Try))
    outer.handlers = []
ns = dict(torch=SimpleNamespace(random=SimpleNamespace(fork_rng=rng)),
          torch_device_module=SimpleNamespace(current_device=device), AI_DEVICE='cuda',
          time=SimpleNamespace(perf_counter=lambda: 1), logger=SimpleNamespace(info=lambda *a: None),
          fatal_device_error_guard=errors.fatal_device_error_guard,
          terminate_on_fatal_device_error=errors.terminate_on_fatal_device_error)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<warmup>', 'exec'), ns)
runner = SimpleNamespace(validate_warmup=lambda: None, _run_warmup=lambda: step('body'),
                         clear_warmup_state=lambda: step('cleanup'), _maybe_freeze_gc=lambda: step('freeze'))
try:
    result = ns['run_warmup'](runner)
except RuntimeError as caught:
    assert caught is error
    print('SAME_ERROR', flush=True)
else:
    assert result is None
    assert runner.warmup_time_seconds == 0
    print('RETURN_NONE', flush=True)
"""
        return subprocess.run([sys.executable, "-c", code, str(ROOT), stage, str(fatal), str(mutation)], capture_output=True, text=True, timeout=5)

    def assert_fatal(self, result, expected):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout.splitlines(), expected)

    def test_fatal_context_and_body_failures_exit_before_cleanup(self):
        stages = ["device", "construct", "enter", "body", "exit"]
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                self.assert_fatal(self.run_warmup(stage, True), stages[: index + 1])

    def test_ordinary_failures_preserve_identity_and_cleanup(self):
        stages = ["device", "construct", "enter", "body", "exit"]
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                result = self.run_warmup(stage, False)
                expected = stages[: index + 1]
                if stage == "body":
                    expected += ["exit"]
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), expected + ["cleanup", "SAME_ERROR"])
                self.assertNotIn("fatal inference failure", result.stderr)

    def test_success_retains_cleanup_return_and_timing(self):
        result = self.run_warmup("none", False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["device", "construct", "enter", "body", "exit", "cleanup", "freeze", "RETURN_NONE"])

    def test_removing_outer_check_is_detected(self):
        for stage in ("device", "construct", "enter", "exit"):
            with self.subTest(stage=stage):
                result = self.run_warmup(stage, True, mutation=True)
                # The mutant must run normally and expose delayed handling, not crash.
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("cleanup", result.stdout.splitlines())
                self.assertIn("SAME_ERROR", result.stdout.splitlines())
                self.assertNotIn("Traceback", result.stderr)
                with self.assertRaises(AssertionError):
                    self.assert_fatal(result, [])
