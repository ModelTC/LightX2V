"""CPU checks for exact launcher forwarding and same-PID evaluator lifetime."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "scripts/platform/nvidia/geneval_inline.py"
SPEC = importlib.util.spec_from_file_location("inline_geneval_test", FILE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
LAUNCHER = "import runpy, sys\ncache, script, *arguments = sys.argv[1:]\nsys.argv = [script, *arguments]\nrunpy.run_path(script, run_name='__main__')\n"


class InlineGenEvalTests(unittest.TestCase):
    def fixture(self, directory, mismatch=False, exit_code=None):
        root = Path(directory)
        script = root / "evaluation/evaluate_images.py"
        script.parent.mkdir()
        output = root / "result.json"
        if exit_code is None:
            script.write_text("import json,os,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'arguments':sys.argv[2:]}))\n")
        else:
            script.write_text(f"raise SystemExit({exit_code})\n")
        paths = root / "paths.json"
        paths.write_text(json.dumps({"geneval_python": sys.executable, "geneval": str(root), "geneval_clip": str(root / "cache")}))
        worker = root / "frozen.py"
        code = repr(LAUNCHER + ("# changed\n" if mismatch else "")) if mismatch else "launcher"
        worker.write_text(
            "import json,subprocess,sys\nfrom pathlib import Path\n"
            + "def score():\n launcher = "
            + repr(LAUNCHER)
            + "\n subprocess.run([sys.executable,'-c',"
            + code
            + ",paths['geneval_clip'],str(Path(paths['geneval'])/'evaluation/evaluate_images.py'),"
            + repr(str(output))
            + ",'preserved-argument'],check=True,timeout=3300)\n"
            + "if __name__=='__main__':\n paths=json.loads(Path(sys.argv[sys.argv.index('--paths')+1]).read_text())\n score()\n"
        )
        return [str(worker), "score", "--paths", str(paths), "--dataset", "geneval"], output, paths

    def test_exact_launcher_and_arguments_execute_in_held_pid(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            command, output, _ = self.fixture(directory)
            original_run, argv = subprocess.run, sys.argv
            M.run_score_in_process(command)
            record = json.loads(output.read_text())
            self.assertEqual(record, {"pid": os.getpid(), "arguments": ["preserved-argument"]})
            self.assertIs(subprocess.run, original_run)
            self.assertIs(sys.argv, argv)

    def test_changed_launcher_is_rejected_and_globals_restored(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            command, output, _ = self.fixture(directory, mismatch=True)
            original_run, argv = subprocess.run, sys.argv
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                M.run_score_in_process(command)
            self.assertFalse(output.exists())
            self.assertIs(subprocess.run, original_run)
            self.assertIs(sys.argv, argv)

    def test_evaluator_failure_propagates_and_globals_restore(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            command, _, _ = self.fixture(directory, exit_code=7)
            original_run, argv = subprocess.run, sys.argv
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                M.run_score_in_process(command)
            self.assertEqual(caught.exception.returncode, 7)
            self.assertIs(subprocess.run, original_run)
            self.assertIs(sys.argv, argv)

    def test_other_python_environment_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            command, _, paths = self.fixture(directory)
            values = json.loads(paths.read_text())
            values["geneval_python"] += "-other"
            paths.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "legacy interpreter"):
                M.run_score_in_process(command)


if __name__ == "__main__":
    unittest.main()
