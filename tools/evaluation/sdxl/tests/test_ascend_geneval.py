"""Only device wiring may differ from the pinned GenEval scoring program."""
import importlib.util
import types
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / 'scripts/platform/ascend_npu/ascend_geneval_score.py'
SPEC = importlib.util.spec_from_file_location('ascend_geneval_test', FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class DeviceProgramTests(unittest.TestCase):
    SOURCE = '''DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"
values = [True, False, True]
score = sum(values) / len(values)
assert len(values) == 3
'''

    def test_device_assertion_is_preserved_and_math_unchanged(self):
        for available in (True, False):
            namespace = {'torch': types.SimpleNamespace(npu=types.SimpleNamespace(is_available=lambda: available))}
            code = M.device_program(self.SOURCE, 'pinned.py')
            if available:
                exec(code, namespace)
                self.assertEqual(namespace['DEVICE'], 'npu')
                self.assertEqual(namespace['score'], 2 / 3)
            else:
                with self.assertRaises(AssertionError):
                    exec(code, namespace)

    def test_changed_upstream_device_program_is_rejected(self):
        for source in (self.SOURCE.replace('assert DEVICE == "cuda"', ''),
                       self.SOURCE + '\nDEVICE = "cuda"\n'):
            with self.assertRaises(ValueError):
                M.device_program(source, 'unexpected.py')


if __name__ == '__main__':
    unittest.main()
