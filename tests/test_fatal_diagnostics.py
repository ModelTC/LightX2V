"""Fatal diagnostics must preserve caller frames without delaying process exit."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "lightx2v/utils/inference_errors.py"


class FatalDiagnosticTests(unittest.TestCase):
    def code(self, failure=""):
        return f"""
import importlib.util
spec = importlib.util.spec_from_file_location('errors', {str(MODULE)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
{failure}
def offending_operator():
    raise RuntimeError('MUSA error: device-side assert triggered')
try:
    offending_operator()
except RuntimeError as error:
    m.terminate_on_fatal_device_error(error)
print('UNEXPECTED_RETURN', flush=True)
"""

    def test_reports_original_and_chained_frame_locations(self):
        code = self.code().replace(
            "    m.terminate_on_fatal_device_error(error)",
            "    try:\n        raise RuntimeError('wrapper failure') from error\n    except RuntimeError as wrapper:\n        m.terminate_on_fatal_device_error(wrapper)",
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("LightX2V fatal inference failure", result.stderr)
        self.assertIn("in offending_operator", result.stderr)
        self.assertIn('File "<string>", line', result.stderr)
        self.assertGreaterEqual(result.stderr.count("RuntimeError stack"), 2)
        self.assertNotIn("UNEXPECTED_RETURN", result.stdout)

    def test_preserves_shared_stderr_flags(self):
        read_fd, write_fd = os.pipe()
        try:
            before = os.get_blocking(write_fd)
            result = subprocess.run([sys.executable, "-c", self.code()], stderr=write_fd, stdout=subprocess.PIPE, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(os.get_blocking(write_fd), before)
            os.close(write_fd)
            write_fd = None
            self.assertIn(b"in offending_operator", os.read(read_fd, 8192))
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def test_full_stderr_does_not_block_exit_or_change_flags(self):
        read_fd, write_fd = os.pipe()
        try:
            os.set_blocking(write_fd, False)
            while True:
                try:
                    os.write(write_fd, b"x" * 4096)
                except BlockingIOError:
                    break
            os.set_blocking(write_fd, True)
            result = subprocess.run([sys.executable, "-c", self.code()], stderr=write_fd, stdout=subprocess.PIPE, timeout=5)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(os.get_blocking(write_fd))
            self.assertNotIn(b"UNEXPECTED_RETURN", result.stdout)
        finally:
            os.close(read_fd)
            os.close(write_fd)

    def test_regular_file_diagnostics_append(self):
        with tempfile.TemporaryFile() as sink:
            sink.write(b"EXISTING LOG\n")
            sink.flush()
            sink.seek(0)
            result = subprocess.run([sys.executable, "-c", self.code()], stderr=sink, stdout=subprocess.PIPE, timeout=5)
            self.assertEqual(result.returncode, 1)
            sink.seek(0)
            output = sink.read()
            self.assertTrue(output.startswith(b"EXISTING LOG\nLightX2V fatal inference failure"))
            self.assertIn(b"in offending_operator", output)
            self.assertLessEqual(len(output), 4096 + len(b"EXISTING LOG\n"))

    def test_diagnostic_failures_cannot_prevent_exit(self):
        for failure in (
            "m.os.close(2)",
            "m.os.open = lambda *a: (_ for _ in ()).throw(OSError('unsupported stderr'))",
            "m.traceback.walk_tb = lambda *a: (_ for _ in ()).throw(RuntimeError('format failure'))",
            "m.os.close = lambda *a: (_ for _ in ()).throw(OSError('close failure'))",
        ):
            with self.subTest(failure=failure):
                result = subprocess.run([sys.executable, "-c", self.code(failure)], capture_output=True, timeout=5)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn(b"Traceback", result.stderr)
                self.assertNotIn(b"UNEXPECTED_RETURN", result.stdout)
