"""Exercise real fallback methods with CPU stand-ins for device operations."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RunnerFallbackTests(unittest.TestCase):
    def run_fallback(self, case, fatal):
        code = f"""
import ast, importlib.util
from pathlib import Path
from types import SimpleNamespace
root = Path({str(ROOT)!r})
spec = importlib.util.spec_from_file_location('errors', root / 'lightx2v/utils/inference_errors.py')
errors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(errors)
ns = {{'terminate_on_fatal_device_error': errors.terminate_on_fatal_device_error,
      'logger': SimpleNamespace(info=lambda *a: None, warning=lambda *a: None, error=lambda *a: None)}}
def load(path, classname, method):
    tree = ast.parse((root / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
    return ns[method]
def fail(*a, **kw):
    print('DEVICE_OPERATION', flush=True)
    raise RuntimeError('CUDA error: an illegal memory access was encountered' if {fatal!r} else 'ordinary failure')
case = {case!r}
model = SimpleNamespace(_update_lora=fail, _remove_lora=fail, _load_lora_file=fail)
runner = SimpleNamespace(model=model)
if case == 'lora_multi':
    ns['os'] = SimpleNamespace(path=SimpleNamespace(isfile=lambda *a: True))
    fn = load('lightx2v/models/runners/default_runner.py', 'DefaultRunner', 'switch_lora_multi')
    result = fn(runner, [('a', 1.0), ('b', 1.0)])
else:
    fn = load('lightx2v/models/runners/default_runner.py', 'DefaultRunner', 'switch_lora')
    result = fn(runner, '' if case == 'lora_remove' else 'a')
assert result is False
print('ORDINARY_FALLBACK', flush=True)
"""
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)

    def test_fatal_device_error_never_becomes_a_fallback(self):
        for case in ("lora_update", "lora_remove", "lora_multi"):
            with self.subTest(case=case):
                result = self.run_fallback(case, True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("DEVICE_OPERATION", result.stdout)
                self.assertIn("LightX2V fatal inference failure", result.stderr)
                self.assertNotIn("ORDINARY_FALLBACK", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_ordinary_device_operation_errors_keep_the_existing_fallback(self):
        for case in ("lora_update", "lora_remove", "lora_multi"):
            with self.subTest(case=case):
                result = self.run_fallback(case, False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("DEVICE_OPERATION", result.stdout)
                self.assertIn("ORDINARY_FALLBACK", result.stdout)
                self.assertNotIn("LightX2V fatal inference failure", result.stderr)


if __name__ == "__main__":
    unittest.main()
