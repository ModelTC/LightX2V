import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hymotion_contract", ROOT / "evaluate.py")
EVALUATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATION)


class ArchiveContractTests(unittest.TestCase):
    def test_identical_worker_is_accepted_but_changed_timer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "evaluate.py"
            source = (ROOT / "evaluate.py").read_text()
            worker.write_text(source)
            shutil.copytree(ROOT / "configs", root / "configs")
            self.assertEqual(EVALUATION.checked_worker_script({"worker_script": str(worker)}), worker)
            changed = source.replace("started = time.perf_counter()", "started = time.perf_counter() + 1", 1)
            self.assertNotEqual(changed, source)
            worker.write_text(changed)
            with self.assertRaises(ValueError):
                EVALUATION.checked_worker_script({"worker_script": str(worker)})


if __name__ == "__main__":
    unittest.main()
