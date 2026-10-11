"""CPU-only checks for the independent 8-step official-flow Ref2AV recipe."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from lightx2v_train.trainers.dmd.tests import test_omni_pdmd_official_launch_config as four_step

CONFIG = four_step.TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd8_official_fsdp32.yaml"
LAUNCHER = four_step.TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd8_official_fsdp32_32gpu_acp.sh"


class OmniPdmd8OfficialLaunchConfigTests(unittest.TestCase):
    def test_only_rollout_steps_differ_from_four_step_config(self):
        helper = four_step.OmniPdmdOfficialLaunchConfigTests()
        old, new = helper.config(), helper.config(CONFIG)
        old.pop("config_path")
        new.pop("config_path")
        self.assertEqual(new["training"]["dmd"]["num_inference_steps"], 8)
        new["training"]["dmd"]["num_inference_steps"] = 4
        self.assertEqual(new, old)

    @staticmethod
    def fixture(root):
        manifest, receipt, environment = four_step.OmniPdmdOfficialLaunchConfigTests.fixture(root)
        environment.pop("H3_PDMD8_OFFICIAL_CONFIG", None)
        # A stale four-step override must not select that recipe here.
        environment["H3_PDMD_OFFICIAL_CONFIG"] = str(four_step.CONFIG)
        return manifest, receipt, environment

    @staticmethod
    def dry_run(environment):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=environment, capture_output=True, text=True)

    def test_bash_syntax(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dry_run_selects_eight_steps_without_launching_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            for expected in (
                CONFIG.name,
                "--nnodes=4 --nproc_per_node=8",
                "steps=8, optimizer_updates=250000, student_updates=41666, critic_updates=208334",
                "Checkpoint: save_every_iters=50, save_total_limit=5",
                "global_microbatch=32, batch_mode=count_random",
                "lora_param_dtype=fp32",
                "fake: train_type=full, lr=1e-05",
                "precision teacher: transformer_param_dtype=bf16, running_dtype=bf16",
            ):
                self.assertIn(expected, result.stdout)
            self.assertNotIn(four_step.CONFIG.name, result.stdout)
            self.assertNotIn("/stale/", result.stdout)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertFalse((root / "outputs").exists())

    def test_default_output_and_rendezvous_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            environment.pop("H3_REF2AV_DMD_OUTPUT")
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("outputs/minimax_h3_ref2av_omni_imageonly_pdmd8_official_lorafp32_250000updates", result.stdout)
            self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_pdmd8_official_lorafp32_250000updates", result.stdout)
            environment["H3_REF2AV_DMD_OUTPUT"] = str(root / "custom-output")
            environment["H3_RDZV_ID"] = "custom-eight-step"
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(root / "custom-output"), result.stdout)
            self.assertIn("--rdzv_id=custom-eight-step", result.stdout)

    def test_rejects_four_step_and_legacy_eight_step_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, environment = self.fixture(Path(directory))
            for config, message in (
                (four_step.CONFIG, "requires 8-step fake-first"),
                (four_step.OLD_CONFIG, "requires the official PDMD training flow"),
            ):
                with self.subTest(config=config):
                    result = self.dry_run({**environment, "H3_PDMD8_OFFICIAL_CONFIG": str(config)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("--nnodes=4 --nproc_per_node=8", result.stdout)


if __name__ == "__main__":
    unittest.main()
