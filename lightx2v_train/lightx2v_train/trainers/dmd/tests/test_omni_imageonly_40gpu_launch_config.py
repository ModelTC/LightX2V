"""CPU-only regression checks for the independent 5 x 8 GPU PDMD entry point."""

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
CONFIG32 = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_match124_bf16_fsdp32.yaml"
CONFIG40 = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd8_match124_bf16_fsdp40.yaml"
LAUNCHER32 = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_fsdp32_32gpu_acp.sh"
LAUNCHER40 = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd8_fsdp40_40gpu_acp.sh"


class OmniImageOnly40GpuLaunchConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        transformer = self.root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        self.manifest = self.root / "metadata.jsonl"
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.receipt_path = self.manifest.with_suffix(".complete.json")
        self.receipt = {
            "completed_count": 1,
            "failed_count": 0,
            "input_total_rows": 1,
            "preprocess_fingerprint": "fixture-fingerprint",
            "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "reference_image_counts": {"1": 1},
        }
        self.write_receipt()
        self.environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29599",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_REF2AV_CACHE": str(self.manifest),
            "H3_REF2AV_DMD_OUTPUT": str(self.root / "outputs"),
            "H3_REF2AV_EXPECTED_ROWS": "1",
            "H3_PDMD": "true",
        }
        for key in ("H3_CONFIG_PATH", "H3_RDZV_ID", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE"):
            self.environment.pop(key, None)

    def write_receipt(self, **overrides):
        self.receipt_path.write_text(json.dumps({**self.receipt, **overrides}), encoding="utf-8")

    def config(self, path=CONFIG40):
        with patch.dict(os.environ, {**self.environment, "H3_PDMD": "false"}, clear=True):
            return load_config(str(path))

    def run_launcher(self, overrides=None, launcher=LAUNCHER40):
        return subprocess.run(
            ["bash", str(launcher), "--dry-run"],
            env={**self.environment, **(overrides or {})},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def run_custom_config(self, config):
        custom_path = self.root / "custom.yaml"
        custom_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return self.run_launcher({"H3_CONFIG_PATH": str(custom_path)})

    def test_resolved_config_only_changes_fsdp_size(self):
        original = self.config(CONFIG32)
        expanded = self.config()
        self.assertEqual(original["distributed"]["fsdp2"]["size"], 32)
        self.assertEqual(expanded["distributed"]["fsdp2"]["size"], 40)
        for config in (original, expanded):
            config.pop("config_path")
        expanded["distributed"]["fsdp2"]["size"] = 32
        self.assertEqual(expanded, original)

    def test_pdmd_and_mixed_precision_defaults_are_preserved(self):
        config = self.config()
        model = config["model"]
        training = config["training"]
        self.assertEqual(model["running_dtype"], "bf16")
        self.assertEqual(model["transformer_param_dtype"], "bf16")
        for role in ("fake", "teacher"):
            self.assertEqual(model[role]["transformer_param_dtype"], "fp32")
        self.assertIs(model["use_autocast"], False)
        self.assertEqual(config["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(training["fake"]["train_type"], "full")
        self.assertIs(model["capabilities"]["distribution_matching"]["projected_dmd"], True)
        self.assertEqual(training["dmd"]["num_inference_steps"], 8)
        self.assertEqual(training["dmd"]["update_order"], "student_first")
        self.assertEqual(training["dmd"]["fake_update_ratio"], 5)
        self.assertEqual(training["max_train_iters"], 10000)

    def test_dry_run_uses_40gpu_config_and_separate_output_and_rendezvous(self):
        result = self.run_launcher({"H3_REF2AV_DMD_OUTPUT": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "--nnodes=5 --nproc_per_node=8",
            f"--config {CONFIG40}",
            "5 x 8 GPUs, PDMD, steps=8, iters=10000",
            "global_microbatch=40, batch_mode=count_coverage",
            "outputs/minimax_h3_ref2av_omni_imageonly_pdmd8_fsdp40_paper_10k",
            "--rdzv_id=h3_ref2av_omni_imageonly_pdmd8_fsdp40_paper_10k",
            "Verified cache: completed=1, failed=0, input_rows=1",
            "60000 optimizer updates",
        ):
            self.assertIn(expected, result.stdout)
        for role, dtype in (("student", "bf16"), ("fake", "fp32"), ("teacher", "fp32")):
            self.assertIn(f"precision {role}: transformer_param_dtype={dtype},", result.stdout)

        original = self.run_launcher({"H3_REF2AV_DMD_OUTPUT": ""}, launcher=LAUNCHER32)
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertIn("--nnodes=4 --nproc_per_node=8", original.stdout)
        self.assertIn(f"--config {CONFIG32}", original.stdout)
        self.assertIn("global_microbatch=32, batch_mode=count_coverage", original.stdout)
        self.assertIn("outputs/minimax_h3_ref2av_omni_imageonly_pdmd8_paper_10k", original.stdout)
        self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_pdmd8_paper_10k", original.stdout)

    def test_output_override_is_honored_without_starting_training(self):
        result = self.run_launcher({"H3_RDZV_ID": "custom-40gpu-job"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"output={self.root / 'outputs'}", result.stdout)
        self.assertIn("--rdzv_id=custom-40gpu-job", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_40gpu_launcher_rejects_32gpu_config_override(self):
        result = self.run_launcher({"H3_CONFIG_PATH": str(CONFIG32)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FSDP40, no sequence parallel", result.stderr)

    def test_existing_cache_receipt_and_hash_checks_remain_required(self):
        for overrides, message in (
            ({"input_total_rows": 2}, "Invalid completed/failed/source accounting"),
            ({"preprocess_fingerprint": ""}, "Missing preprocessing fingerprint"),
            ({"manifest_sha256": "bad-digest"}, "no longer matches its complete receipt"),
        ):
            with self.subTest(receipt_override=overrides):
                self.write_receipt(**overrides)
                result = self.run_launcher()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.write_receipt()
        self.manifest.write_text('{"changed": true}\n', encoding="utf-8")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer matches its complete receipt", result.stderr)
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.receipt_path.unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing completed 32-shard merge receipt", result.stderr)

    def test_student_precision_and_lora_settings_are_not_pinned(self):
        custom = self.config()
        custom["model"]["transformer_param_dtype"] = "fp32"
        custom["training"]["student"]["lora"].update(rank=64, alpha=8)
        result = self.run_custom_config(custom)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("student: train_type=lora, rank=64, alpha=8\n", result.stdout)
        self.assertIn("critic: train_type=full\n", result.stdout)
        self.assertIn("precision student: transformer_param_dtype=fp32,", result.stdout)

    def test_score_role_precision_overrides_are_accepted_and_reported(self):
        custom = self.config()
        mixed = {**custom["distributed"]["fsdp2"]["mixed_precision"], "param_dtype": "fp32"}
        for role in ("fake", "teacher"):
            custom["model"][role].update(
                transformer_param_dtype="bf16",
                running_dtype="fp32",
                use_autocast=True,
                distributed={"fsdp2": {"mixed_precision": {"param_dtype": "fp32"}}},
            )
        result = self.run_custom_config(custom)
        self.assertEqual(result.returncode, 0, result.stderr)
        for role in ("fake", "teacher"):
            self.assertIn(
                f"precision {role}: transformer_param_dtype=bf16, running_dtype=fp32, "
                f"use_autocast=True, fsdp={json.dumps(mixed, sort_keys=True)}\n",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
