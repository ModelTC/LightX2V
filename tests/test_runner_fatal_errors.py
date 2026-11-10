"""CPU subprocess coverage for fatal errors before runner failure cleanup."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunnerFatalErrorTests(unittest.TestCase):
    def run_runner(self, runner, failure=None, fatal=True, offload=True, cleanup_fatal=False):
        code = f"""
import ast, importlib.util
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
failure = {failure!r}
fatal = {fatal!r}
def step(name, result=None):
    print(name, flush=True)
    if name == "cleanup" and {cleanup_fatal!r}:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    if name == failure:
        raise RuntimeError('CUDA error: an illegal memory access was encountered' if fatal else 'ordinary failure')
    return result
ns = {{'terminate_on_fatal_device_error': errors.terminate_on_fatal_device_error,
       'fatal_device_error_guard': errors.fatal_device_error_guard, 'suppress': suppress}}
if {runner!r} == 'ltx25':
    class Parent:
        def run_pipeline(self, info):
            return step('infer', 'ok')
    ns['Parent'] = Parent
    tree = ast.parse((root / 'lightx2v/models/runners/ltx2/ltx25_runner.py').read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LTX25Runner')
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'run_pipeline')
    cls = ast.ClassDef(name='Runner', bases=[ast.Name(id='Parent', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<runner>', 'exec'), ns)
    runner = ns['Runner']()
    runner.end_run = lambda: step('cleanup')
    runner.inputs = object()
    runner.input_info = object()
    result = runner.run_pipeline(object())
else:
    tree = ast.parse((root / 'lightx2v/models/runners/minimax_h3/minimax_h3_runner.py').read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and any(isinstance(m, ast.FunctionDef) and m.name == 'run_main' for m in n.body))
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'run_main')
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<runner>', 'exec'), ns)
    runner = SimpleNamespace(
        config={{'cpu_offload': {offload!r}}},
        _prepare_resampling_source=lambda: None, _release_low_memory_vae=lambda: None,
        init_run=lambda: step('init'),
        run_segment=lambda _: step('infer', (None, None)),
        _offload_transformer=lambda: step('offload'),
        run_vae_decoder=lambda *a: step('decode', (None, None)),
        process_images_after_vae_decoder=lambda: step('postprocess', 'ok'),
        end_run=lambda: step('cleanup'),
    )
    result = ns['run_main'](runner)
print('RETURN', result, flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def test_ltx25_fatal_inference_skips_cleanup(self):
        result = self.run_runner("ltx25", "infer")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["infer"])
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_ltx25_ordinary_failure_still_cleans_up(self):
        result = self.run_runner("ltx25", "infer", fatal=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["infer", "cleanup"])
        self.assertIn("ordinary failure", result.stderr)
        self.assertNotIn("LightX2V fatal inference failure", result.stderr)

    def test_ltx25_fatal_cleanup_error_exits_immediately(self):
        result = self.run_runner("ltx25", "infer", fatal=False, cleanup_fatal=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.splitlines(), ["infer", "cleanup"])
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_minimax_fatal_errors_skip_subsequent_device_work(self):
        for offload in (False, True):
            stages = ["init", "infer", "decode", "postprocess", "cleanup"]
            if offload:
                stages.append("offload")
            for stage in stages:
                with self.subTest(offload=offload, stage=stage):
                    result = self.run_runner("minimax", stage, offload=offload)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(result.stdout.splitlines()[-1], stage)
                    self.assertIn("LightX2V fatal inference failure", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_minimax_ordinary_errors_preserve_cleanup(self):
        for stage in ("init", "infer", "decode", "postprocess", "offload", "cleanup"):
            with self.subTest(stage=stage):
                result = self.run_runner("minimax", stage, fatal=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn("cleanup", result.stdout.splitlines())
                self.assertNotIn("LightX2V fatal inference failure", result.stderr)
                self.assertIn("ordinary failure", result.stderr)

    def test_success_paths_are_unchanged(self):
        for runner in ("ltx25", "minimax"):
            with self.subTest(runner=runner):
                result = self.run_runner(runner)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("RETURN ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
