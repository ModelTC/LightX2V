import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1] / "scripts/platform/nvidia/worker.py"


class WorkerLifecycleTests(unittest.TestCase):
    def test_completed_work_keeps_pid_alive_until_post_sample_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "task.py"
            script.write_text("value = sum(range(100))\nassert value == 4950\n")
            complete, release = root / "complete.json", root / "release.json"
            process = subprocess.Popen([sys.executable, str(WORKER), "--completion", str(complete), "--release", str(release), "--", str(script)])
            try:
                deadline = time.monotonic() + 5
                while not complete.exists() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll())
                    time.sleep(0.02)
                marker = json.loads(complete.read_text())
                self.assertEqual(marker["pid"], process.pid)
                self.assertIsNone(process.poll())
                self.assertLess(marker["end"], time.monotonic())
                release.write_text("{}")
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)

    def test_failed_work_cannot_write_a_success_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "task.py"
            script.write_text("raise RuntimeError('intentional lifecycle test failure')\n")
            complete = root / "complete.json"
            result = subprocess.run(
                [sys.executable, str(WORKER), "--completion", str(complete), "--release", str(root / "release"), "--", str(script)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(complete.exists())


if __name__ == "__main__":
    unittest.main()
