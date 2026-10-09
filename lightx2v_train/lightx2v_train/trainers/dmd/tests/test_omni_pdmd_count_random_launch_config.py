"""CPU checks for count-balanced/random-orientation PDMD with BF16 teacher storage."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from lightx2v_train.runtime.config import load_config

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_fsdp32_32gpu_acp.sh"
OLD_CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_match124_bf16_fsdp32.yaml"
OLD_LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_fsdp32_32gpu_acp.sh"


class OmniPdmdCountRandomLaunchConfigTests(unittest.TestCase):
    def config(self, path=CONFIG, environment=None):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/model",
                "H3_REF2AV_CACHE": "/cache/metadata.jsonl",
                "H3_REF2AV_DMD_OUTPUT": "/outputs",
                "H3_PDMD": "false",
                **(environment or {}),
            },
        ):
            return load_config(str(path))

    def test_only_sampler_and_teacher_storage_change_from_existing_omni32_pdmd(self):
        baseline, current = self.config(OLD_CONFIG), self.config()
        self.assertEqual(baseline["model"]["teacher"]["transformer_param_dtype"], "fp32")
        self.assertEqual(current["model"]["teacher"]["transformer_param_dtype"], "bf16")
        self.assertEqual(current["model"]["transformer_param_dtype"], "bf16")
        self.assertEqual(current["model"]["fake"]["transformer_param_dtype"], "fp32")
        baseline["model"]["teacher"]["transformer_param_dtype"] = "bf16"
        for config in (baseline, current):
            config.pop("config_path")
            config["data"]["train"].pop("reference_cost_sampler")
        self.assertEqual(current, baseline)
        self.assertTrue(current["model"]["capabilities"]["distribution_matching"]["projected_dmd"])
        self.assertEqual(current["training"]["dmd"]["num_inference_steps"], 8)
        self.assertEqual(current["training"]["dmd"]["update_order"], "student_first")
        self.assertEqual(current["training"]["dmd"]["fake_update_ratio"], 5)
        self.assertEqual(current["training"]["max_train_iters"], 10000)
        self.assertEqual(current["training"]["gradient_accumulation_iters"], 1)
        self.assertEqual(current["data"]["train"]["batch_size"], 1)
        self.assertEqual(current["distributed"]["fsdp2"]["size"], 32)

    def test_new_sampler_balances_counts_but_never_fixes_orientation(self):
        sampler = self.config()["data"]["train"]["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "count_random")
        self.assertEqual(sampler["image_counts"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(sampler["require_image_only"])
        self.assertFalse(sampler["require_all_image_counts"])
        self.assertTrue(sampler["balance_image_counts"])
        self.assertFalse(sampler["balance_orientation"])
        self.assertFalse(sampler["strict_full_epoch"])
        self.assertEqual(sampler["remainder_policy"], "rotating_drop")
        baseline = self.config(OLD_CONFIG)["data"]["train"]["reference_cost_sampler"]
        for field in ("seed", "cost_key", "require_compute_cost"):
            self.assertEqual(sampler[field], baseline[field])

    def test_existing_config_still_defaults_to_count_coverage(self):
        sampler = self.config(OLD_CONFIG)["data"]["train"]["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "count_coverage")
        self.assertEqual(sampler["image_counts"], list(range(1, 10)))
        self.assertFalse(sampler["balance_image_counts"])
        self.assertFalse(sampler["balance_orientation"])

    def fixture(self, root):
        transformer = root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}")
        # Two complete count groups, deliberately only landscape: random
        # orientation must not require sixteen samples in both orientations.
        rows = [
            {
                "source_id": str(index),
                "reference_image_count": 1 + index // 32,
                "reference_video_count": 0,
                "reference_audio_count": 0,
                "target_num_frames": 124,
                "target_height": 768,
                "target_width": 1344,
                "target_orientation": "landscape",
                "packed_sequence_tokens_124": 18000 + index,
            }
            for index in range(64)
        ]
        manifest = root / "metadata.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = {
            "completed_count": 64,
            "failed_count": 0,
            "input_total_rows": 64,
            "preprocess_fingerprint": "count-random-fixture",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "reference_image_counts": {"1": 32, "2": 32},
        }
        manifest.with_suffix(".complete.json").write_text(json.dumps(receipt))
        environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29596",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(root / "model"),
            "H3_REF2AV_CACHE": str(manifest),
            "H3_REF2AV_DMD_OUTPUT": str(root / "outputs"),
            "H3_CONFIG_PATH": "/stale/legacy-shift12.yaml",
            "H3_PDMD": "false",
            "H3_REF2AV_EXPECTED_ROWS": "13553",
        }
        for name in ("H3_PDMD_COUNT_RANDOM_CONFIG", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE", "H3_RDZV_ID"):
            environment.pop(name, None)
        return manifest, receipt, environment

    @staticmethod
    def dry_run(environment, launcher=LAUNCHER):
        return subprocess.run(["bash", str(launcher), "--dry-run"], env=environment, capture_output=True, text=True)

    def test_wrapper_ignores_stale_config_projection_and_old_expected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _, environment = self.fixture(root)
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            self.assertIn("PDMD, steps=8, iters=10000", result.stdout)
            self.assertIn("global_microbatch=32, batch_mode=count_random", result.stdout)
            self.assertIn(CONFIG.name, result.stdout)
            self.assertNotIn("/stale/legacy-shift12.yaml", result.stdout)
            self.assertNotIn("16+16", result.stdout)
            self.assertNotIn("16 landscape + 16 portrait", result.stdout)
            self.assertIn("Verified cache: completed=64, failed=0, input_rows=64", result.stdout)
            for role, dtype in (("student", "bf16"), ("fake", "fp32"), ("teacher", "bf16")):
                self.assertIn(f"precision {role}: transformer_param_dtype={dtype},", result.stdout)
            self.assertFalse((root / "outputs").exists())
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after, "Dry-run must not rewrite the model/cache/receipt or create training files.")
            self.assertTrue(manifest.is_file())

    def test_default_output_and_rendezvous_are_distinct_and_overrideable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            environment.pop("H3_REF2AV_DMD_OUTPUT")
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("pdmd8_count_random_10k", result.stdout)
            self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_pdmd8_count_random_10k", result.stdout)
            self.assertNotIn("pdmd8_paper_10k", result.stdout)
            environment["H3_REF2AV_DMD_OUTPUT"] = str(root / "custom-output")
            environment["H3_RDZV_ID"] = "custom-count-random-run"
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(root / "custom-output"), result.stdout)
            self.assertIn("--rdzv_id=custom-count-random-run", result.stdout)
            self.assertFalse((root / "custom-output").exists())

    def test_dedicated_config_override_is_used_instead_of_generic_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            custom = self.config(environment=environment)
            custom["data"]["train"]["reference_cost_sampler"]["seed"] = 123
            custom_path = root / "custom-count-random.yaml"
            custom_path.write_text(yaml.safe_dump(custom))
            result = self.dry_run({**environment, "H3_PDMD_COUNT_RANDOM_CONFIG": str(custom_path)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(custom_path), result.stdout)
            self.assertNotIn("/stale/legacy-shift12.yaml", result.stdout)

    def test_existing_launcher_keeps_its_default_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            for name in ("H3_CONFIG_PATH", "H3_REF2AV_EXPECTED_ROWS", "H3_REF2AV_DMD_OUTPUT"):
                environment.pop(name, None)
            environment["H3_PDMD"] = "true"
            result = self.dry_run(environment, OLD_LAUNCHER)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("global_microbatch=32, batch_mode=count_coverage", result.stdout)
            self.assertIn(OLD_CONFIG.name, result.stdout)
            self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_pdmd8_paper_10k", result.stdout)
            self.assertNotIn("batch_mode=count_random", result.stdout)

    def test_invalid_count_random_sampler_overrides_fail_preflight(self):
        changes = (
            {"balance_orientation": True},
            {"balance_image_counts": False},
            {"image_counts": list(range(1, 10))},
            {"require_image_only": False},
            {"strict_full_epoch": True},
            {"remainder_policy": "strict"},
            {"batch_mode": "unknown"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            custom_path = root / "invalid-count-random.yaml"
            for change in changes:
                with self.subTest(change=change):
                    custom = self.config(environment=environment)
                    custom["data"]["train"]["reference_cost_sampler"].update(change)
                    custom_path.write_text(yaml.safe_dump(custom))
                    result = self.dry_run({**environment, "H3_PDMD_COUNT_RANDOM_CONFIG": str(custom_path)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("requires", result.stderr)
                    self.assertNotIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            self.assertFalse((root / "outputs").exists())

    def test_missing_or_invalid_receipts_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, receipt, environment = self.fixture(root)
            receipt_path = manifest.with_suffix(".complete.json")
            for changed, message in (
                ({**receipt, "manifest_sha256": "0" * 64}, "no longer matches its complete receipt"),
                ({**receipt, "input_total_rows": 65}, "Invalid completed/failed/source accounting"),
                ({**receipt, "preprocess_fingerprint": ""}, "Missing preprocessing fingerprint"),
            ):
                with self.subTest(message=message):
                    receipt_path.write_text(json.dumps(changed))
                    result = self.dry_run(environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            receipt_path.unlink()
            result = self.dry_run(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing completed", result.stderr)
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
