import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sdxl_environment", ROOT / "evaluate.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class GenerationEnvironmentTests(unittest.TestCase):
    def test_child_uses_upstream_workspace_default_despite_parent_override(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "stale-device"}):
            env = M.generation_environment("GPU-explicit")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            actual = json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-c",
                        "import json,os; print(json.dumps({k:os.environ.get(k) for k in "
                        "['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}))",
                    ],
                    env=env,
                    text=True,
                )
            )
        self.assertEqual(
            actual,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-explicit",
                "CUBLAS_WORKSPACE_CONFIG": None,
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )

    def test_missing_profile_cannot_be_accepted_as_default(self):
        with self.assertRaises(ValueError):
            M.verify_generation_environment(None)


if __name__ == "__main__":
    unittest.main()
