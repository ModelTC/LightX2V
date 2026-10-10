"""CPU-only checks for iter1/save/iter2 on two ACP workers; no model execution."""

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from lightx2v_train.trainers.dmd.tests import test_omni_pdmd_official_launch_config as fixtures

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_lorafp32_save_smoke_fsdp16.yaml"
BASE = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_dmd8_lorafp32_save_smoke_16gpu_acp.sh"


class SaveSmoke16ConfigTests(unittest.TestCase):
    def test_only_topology_duration_checkpoint_policy_and_auto_resume_change(self):
        base = yaml.safe_load(BASE.read_text())
        smoke = yaml.safe_load(CONFIG.read_text())
        self.assertEqual(smoke["distributed"]["fsdp2"]["size"], 16)
        self.assertEqual(smoke["training"]["max_train_iters"], 2)
        self.assertEqual(smoke["training"]["save_every_iters"], 1)
        self.assertEqual(smoke["training"]["save_total_limit"], 2)
        self.assertIs(smoke["resume"]["auto_resume"], False)
        smoke["distributed"]["fsdp2"]["size"] = base["distributed"]["fsdp2"]["size"]
        for key in ("max_train_iters", "save_every_iters", "save_total_limit"):
            smoke["training"][key] = base["training"][key]
        smoke["resume"] = base["resume"]
        self.assertEqual(smoke, base)

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class SaveSmoke16LauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.manifest, _, self.env = fixtures.OmniPdmdOfficialLaunchConfigTests.fixture(self.root)
        self.env.update({"NNODES": "2", "NPROC_PER_NODE": "8", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"})
        self.env.pop("H3_DMD_SAVE_SMOKE_CONFIG", None)

    def run_launcher(self, **overrides):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env={**self.env, **overrides}, capture_output=True, text=True, timeout=30)

    def test_read_only_dry_run_selects_dmd8_fp32_lora_and_two_nodes(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in (str(CONFIG), "--nnodes=2 --nproc_per_node=8", "plain DMD8", "iter1 -> save1 -> iter2 -> save2 -> exit", "student LoRA param_dtype=fp32", "fake_update_ratio=5", "save_total_limit=2", "Verified cache: completed=64"):
            self.assertIn(text, result.stdout)
        self.assertNotIn("/stale/", result.stdout)
        self.assertFalse((self.root / "outputs").exists())
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_old_output_is_never_resumed_or_overwritten(self):
        output = self.root / "outputs"
        output.mkdir()
        marker = output / "existing-checkpoint.txt"
        marker.write_text("preserve")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NEW empty output directory", result.stderr)
        self.assertEqual(marker.read_text(), "preserve")

    def test_rejects_mismatched_topology_and_non_smoke_recipe(self):
        for overrides, message in (
            ({"NNODES": "4"}, "NNODES=2"),
            ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, "eight distinct GPUs"),
            ({"H3_DMD_SAVE_SMOKE_CONFIG": str(BASE)}, "requires FSDP16"),
        ):
            with self.subTest(overrides=overrides):
                result = self.run_launcher(**overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_changed_cache_is_rejected(self):
        self.manifest.write_text("{}\n")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer matches", result.stderr)


if __name__ == "__main__":
    unittest.main()
