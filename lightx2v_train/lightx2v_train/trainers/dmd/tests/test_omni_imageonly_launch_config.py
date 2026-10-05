"""CPU-only schema / ACP dry-run checks for Omni image-only Ref2AV DMD."""

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
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_match124_bf16_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_fsdp32_32gpu_acp.sh"


class OmniImageOnlyLaunchConfigTests(unittest.TestCase):
    def config(self):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/model",
                "H3_REF2AV_CACHE": "/cache/metadata.jsonl",
                "H3_REF2AV_DMD_OUTPUT": "/outputs",
                # Direct config loading cannot silently switch to PDMD.
                "H3_PDMD": "true",
            },
        ):
            return load_config(str(CONFIG))

    def test_training_is_standard_dmd8_10k(self):
        config = self.config()
        training = config["training"]
        self.assertEqual(training["method"], "dmd")
        self.assertEqual(training["max_train_iters"], 10000)
        self.assertEqual(training["gradient_accumulation_iters"], 1)
        self.assertEqual(training["dmd"]["num_inference_steps"], 8)
        self.assertEqual(training["dmd"]["update_order"], "student_first")
        self.assertEqual(training["dmd"]["fake_update_ratio"], 5)
        self.assertEqual(training["dmd"]["latent_dtype"], "fp32")
        self.assertEqual(training["student"]["train_type"], "lora")
        self.assertEqual(training["student"]["lora"]["rank"], 128)
        self.assertEqual(training["fake"]["train_type"], "full")
        self.assertEqual(training["teacher"]["guidance_scale"], 1.0)
        matching = config["model"]["capabilities"]["distribution_matching"]
        self.assertIs(matching["projected_dmd"], False)
        self.assertIs(matching["geometry_from_metadata"], True)
        self.assertEqual(matching["allowed_resolutions"], [[768, 1344], [1344, 768]])
        self.assertEqual(training["dmd"]["generation_shapes"], [{"value": [124, 768, 1344]}])

    def test_global_count_diversity_without_category_downsampling(self):
        config = self.config()
        data = config["data"]["train"]
        sampler = data["reference_cost_sampler"]
        self.assertEqual(data["batch_size"], 1)
        self.assertEqual(sampler["batch_mode"], "count_coverage")
        self.assertIs(sampler["require_image_only"], True)
        self.assertEqual(sampler["image_counts"], list(range(1, 10)))
        self.assertIs(sampler["require_all_image_counts"], False)
        self.assertIs(sampler["balance_image_counts"], False)
        self.assertIs(sampler["balance_orientation"], False)
        self.assertEqual(sampler["remainder_policy"], "rotating_drop")
        self.assertEqual(config["distributed"]["fsdp2"]["size"], 32)
        self.assertIs(config["distributed"]["sequence_parallel"]["enabled"], False)

    def test_mixed_precision_recipe_preserves_full_fake_master_weights(self):
        config = self.config()
        self.assertEqual(config["model"]["running_dtype"], "bf16")
        self.assertEqual(config["model"]["transformer_param_dtype"], "bf16")
        self.assertEqual(config["model"]["fake"]["transformer_param_dtype"], "fp32")
        mixed = config["distributed"]["fsdp2"]["mixed_precision"]
        self.assertEqual(mixed["param_dtype"], "bf16")
        self.assertEqual(mixed["reduce_dtype"], "fp32")

    def test_launcher_dry_run_and_rejects_incompatible_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transformer = root / "model/transformer_ref"
            transformer.mkdir(parents=True)
            (transformer / "config.json").write_text("{}")
            manifest = root / "metadata.jsonl"
            manifest.write_text("{}\n")
            receipt_path = manifest.with_suffix(".complete.json")
            receipt_path.write_text(
                json.dumps(
                    {
                        "completed_count": 1,
                        "failed_count": 0,
                        "input_total_rows": 1,
                        "preprocess_fingerprint": "fixture-fingerprint",
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "reference_image_counts": {"1": 1},
                    }
                )
            )
            environment = {
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29599",
                "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
                "H3_PYTHON": sys.executable,
                "H3_MODEL_PATH": str(root / "model"),
                "H3_REF2AV_CACHE": str(manifest),
                "H3_REF2AV_DMD_OUTPUT": str(root / "outputs"),
                "H3_REF2AV_EXPECTED_ROWS": "1",
                "H3_PDMD": "false",
            }
            for key in ("H3_CONFIG_PATH", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE"):
                environment.pop(key, None)

            def run(overrides=None):
                return subprocess.run(
                    ["bash", str(LAUNCHER), "--dry-run"],
                    env={**environment, **(overrides or {})},
                    capture_output=True,
                    text=True,
                )

            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            self.assertIn("standard DMD, steps=8, iters=10000", result.stdout)
            self.assertIn("global_microbatch=32, batch_mode=count_coverage", result.stdout)
            self.assertIn("Verified cache: completed=1, failed=0, input_rows=1", result.stdout)
            self.assertFalse((root / "outputs").exists())

            result = run({"H3_PDMD": "true"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("standard DMD recipe", result.stderr)
            result = run({"H3_REF2AV_EXPECTED_ROWS": "100000"})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected 100000, found 1", result.stderr)

            manifest.write_text('{"changed": true}\n')
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no longer matches its complete receipt", result.stderr)
            manifest.write_text("{}\n")
            receipt_path.rename(root / "receipt.backup.json")
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing completed 32-shard merge receipt", result.stderr)
            (root / "receipt.backup.json").rename(receipt_path)

            custom = self.config()
            custom["training"]["dmd"]["fake_update_ratio"] = 1
            custom_path = root / "custom.yaml"
            custom_path.write_text(yaml.safe_dump(custom))
            result = run({"H3_CONFIG_PATH": str(custom_path)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("student-first, 5 critic updates", result.stderr)


if __name__ == "__main__":
    unittest.main()
