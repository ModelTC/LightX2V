import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = TRAIN_ROOT / "scripts/run_wan21_dmd_pdmd_head_10k_fsdp2.sh"
CONFIG = TRAIN_ROOT / "configs/train/dmd/wan2_1_t2v_1_3b_head_comparison_10k_fsdp2.yaml"
OLD_CONFIG = CONFIG.with_name("wan2_1_t2v_1_3b_head_comparison_fsdp2.yaml")


class Wan10kComparisonConfigTest(unittest.TestCase):
    def test_preserves_baseline_except_requested_experiment_changes(self):
        old = yaml.safe_load(OLD_CONFIG.read_text())
        new = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(old["training"]["max_train_iters"], 1000)
        self.assertTrue(old["resume"]["auto_resume"])
        expected = copy.deepcopy(old)
        expected["training"]["max_train_iters"] = 10000
        expected["training"]["dmd"]["residual_head"].update(
            {
                "fit_grad_accum_steps": 4,
                "gate_mode": "calibrated",
            }
        )
        expected["resume"]["auto_resume"] = False
        self.assertEqual(new, expected)

    def test_head_microbatch_budget_and_common_cadence_unchanged(self):
        config = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(config["training"]["gradient_accumulation_iters"], 1)
        self.assertEqual(config["training"]["dmd"]["residual_head"]["fit_steps"], 5)
        self.assertEqual(config["training"]["save_every_iters"], 100)
        self.assertEqual(config["training"]["save_total_limit"], 1)
        self.assertEqual(config["inference"]["infer_every_iters"], 100)
        self.assertEqual(config["data"]["val"]["max_samples"], 8)


class Wan10kLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "new_comparison"
        self.env = {**os.environ, "WAN_DMD_RUN_ROOT": str(self.output), "WAN_DMD_PYTHON": sys.executable, "DMD_GPUS": "0,1", "PDMD_GPUS": "2,3", "HEAD_GPUS": "5,6"}

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, *args):
        return subprocess.run(["bash", str(SCRIPT), *args], env=self.env, text=True, capture_output=True, timeout=15)

    def fake_nvidia_smi(self, body):
        binary_dir = self.root / "bin"
        binary_dir.mkdir()
        binary = binary_dir / "nvidia-smi"
        binary.write_text("#!/usr/bin/env bash\n" + body)
        binary.chmod(0o755)
        self.env["PATH"] = str(binary_dir) + os.pathsep + self.env["PATH"]

    def test_syntax(self):
        result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_needs_no_gpu_or_python(self):
        self.env["WAN_DMD_PYTHON"] = "/missing/python"
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--preflight", result.stdout)
        self.assertFalse(self.output.exists())

    def test_dry_run_does_not_check_gpu_or_write(self):
        self.env["WAN_DMD_PYTHON"] = "/missing/python"
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10000-iteration", result.stdout)
        self.assertIn("head: GPUs=5,6", result.stdout)
        self.assertIn("NOT verified", result.stdout)
        self.assertFalse(self.output.exists())

    def test_disjoint_gpu_groups_required(self):
        self.env["HEAD_GPUS"] = "3,6"
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU 3 appears more than once", result.stderr)
        self.assertFalse(self.output.exists())

    def test_exactly_two_gpu_indices_required(self):
        for value in ("0", "0,1,2", "GPU-one,GPU-two", "00,1"):
            with self.subTest(value=value):
                self.env["DMD_GPUS"] = value
                result = self.run_script("--dry-run")
                self.assertEqual(result.returncode, 2)
                self.assertIn("exactly two", result.stderr)

    def test_existing_root_is_never_reused(self):
        self.output.mkdir()
        marker = self.output / "previous_run.txt"
        marker.write_text("keep me")
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing an existing run root", result.stderr)
        self.assertEqual(marker.read_text(), "keep me")

    def test_driver_failure_is_not_treated_as_idle(self):
        self.fake_nvidia_smi("exit 9\n")
        result = self.run_script("--preflight")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CalledProcessError", result.stderr)
        self.assertFalse(self.output.exists())

    def test_busy_selected_gpu_blocks_without_training(self):
        self.fake_nvidia_smi(
            'if [[ "$1" == --query-gpu=* ]]; then\n  for gpu in 0 1 2 3 5 6; do printf "%s, GPU-%s, 0, 0\\n" "$gpu" "$gpu"; done\nelse\n  printf "GPU-2, 999999, another_training_process\\n"\nfi\n'
        )
        result = self.run_script("--preflight")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Selected GPUs are busy", result.stderr)
        self.assertIn("no processes killed", result.stderr)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
