"""CPU-only recipe/preflight tests for source-aligned 4-NFE H3 Ref PDMD."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd4_official_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_pdmd4_official_fsdp32_32gpu_acp.sh"
OLD_CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_pdmd8_count_random_fsdp32.yaml"
_spec = importlib.util.spec_from_file_location("_official_pdmd_config_test", TRAIN_ROOT / "lightx2v_train/runtime/config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


class OmniPdmdOfficialLaunchConfigTests(unittest.TestCase):
    def config(self, path=CONFIG, environment=None):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/model",
                "H3_REF2AV_CACHE": "/cache/metadata.jsonl",
                "H3_REF2AV_DMD_OUTPUT": "/outputs",
                **(environment or {}),
            },
        ):
            return _config.load_config(str(path))

    def test_preserves_dataset_base_compute_precision_and_uses_fp32_lora(self):
        old, new = self.config(OLD_CONFIG), self.config()
        self.assertEqual(new["data"], old["data"])
        self.assertEqual(new["distributed"], old["distributed"])
        for key in ("name", "pretrained_model_name_or_path", "running_dtype", "transformer_param_dtype", "use_autocast", "attention_backend", "fake", "teacher"):
            self.assertEqual(new["model"][key], old["model"][key], key)
        self.assertEqual(new["model"]["transformer_param_dtype"], "bf16")
        self.assertEqual(new["model"]["fake"]["transformer_param_dtype"], "fp32")
        self.assertEqual(new["model"]["teacher"]["transformer_param_dtype"], "bf16")
        student, fake = new["training"]["student"], new["training"]["fake"]
        self.assertEqual(student["train_type"], "lora")
        self.assertEqual(student["lora"]["rank"], 128)
        self.assertEqual(student["lora"]["alpha"], 8)
        self.assertEqual(student["lora"]["param_dtype"], "fp32")
        self.assertEqual(new["distributed"]["fsdp2"]["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(student["lora"]["target_modules"], old["training"]["student"]["lora"]["target_modules"])
        self.assertEqual(fake["train_type"], "full")

    def test_source_optimizer_step_accounting_and_training_schedule(self):
        config = self.config()
        training, matching = config["training"], config["model"]["capabilities"]["distribution_matching"]
        dmd = training["dmd"]
        self.assertTrue(dmd["official_pdmd"])
        self.assertTrue(matching["official_pdmd"])
        self.assertTrue(matching["projected_dmd"])
        self.assertEqual(training["method"], "dmd")
        self.assertEqual(training["max_train_iters"], 250000)
        self.assertEqual(dmd["num_inference_steps"], 4)
        self.assertEqual(dmd["update_order"], "fake_first")
        self.assertEqual(dmd["fake_update_ratio"], 5)
        self.assertEqual(dmd["model_mode"], "eval")
        self.assertEqual(dmd["latent_dtype"], "fp32")
        self.assertEqual(dmd["official_rollout_video_shift"], 1.0)
        self.assertEqual(dmd["official_rollout_audio_shift"], 1.0)
        self.assertEqual(matching["video_flow_shift"], 12.0)
        self.assertEqual(matching["audio_flow_shift"], 3.0)
        self.assertEqual(dmd["critic_video_threshold"], 0.95)
        self.assertEqual(dmd["critic_audio_floor"], 0.85)
        self.assertEqual(config["scheduler"]["min_sigma"], 0.0)
        self.assertEqual(config["scheduler"]["max_sigma"], 1.0)
        self.assertEqual(dmd["score_sampling"]["type"], "continuous_uniform")
        self.assertEqual(config["seed"], 42)

    def test_source_loss_optimizer_and_user_checkpoint_settings(self):
        config = self.config()
        matching = config["model"]["capabilities"]["distribution_matching"]
        for weight in ("video_loss_weight", "audio_loss_weight", "audio_dmd_loss_weight"):
            self.assertEqual(matching[weight], 0.8)
        self.assertTrue(matching["dmd_normalization"])
        self.assertEqual(matching["dmd_normalization_epsilon"], 0.0)
        self.assertEqual(matching["dmd_reduction"], "mean")
        self.assertEqual(matching["official_pdmd_normalizer_floor"], 1e-5)
        self.assertEqual(matching["official_pdmd_projection_epsilon"], 1e-8)
        self.assertEqual(matching["official_pdmd_loss_clamp"], 5.0)
        training = config["training"]
        for role, lr in (("student", 5e-5), ("fake", 1e-5)):
            self.assertEqual(
                training[role]["optimizer"],
                {
                    "learning_rate": lr,
                    "adam_beta1": 0.0,
                    "adam_beta2": 0.9,
                    "weight_decay": 0.0,
                    "adam_epsilon": 1e-8,
                },
            )
        self.assertEqual(training["teacher"]["guidance_scale"], 1.0)
        self.assertEqual(training["max_grad_norm"], 1.0)
        self.assertEqual(training["lr_scheduler"], "constant")
        self.assertEqual(training["lr_warmup_iters"], 0)
        self.assertEqual(training["save_every_iters"], 50)
        self.assertEqual(training["save_total_limit"], 5)

    def test_existing_eight_step_recipe_does_not_enable_official_flow(self):
        old = self.config(OLD_CONFIG)
        self.assertFalse(old["training"]["dmd"].get("official_pdmd", False))
        self.assertFalse(old["model"]["capabilities"]["distribution_matching"].get("official_pdmd", False))
        self.assertEqual(old["training"]["max_train_iters"], 10000)
        self.assertEqual(old["training"]["dmd"]["num_inference_steps"], 8)
        self.assertEqual(old["training"]["dmd"]["update_order"], "student_first")

    @staticmethod
    def fixture(root):
        transformer = root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}")
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
            "preprocess_fingerprint": "official-pdmd-fixture",
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
            "H3_CONFIG_PATH": "/stale/legacy.yaml",
            "H3_PDMD": "false",
            "H3_PDMD_COUNT_RANDOM_CONFIG": "/stale/count-random.yaml",
            "H3_REF2AV_EXPECTED_ROWS": "13553",
        }
        for name in ("H3_PDMD_OFFICIAL_CONFIG", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE", "H3_RDZV_ID"):
            environment.pop(name, None)
        return manifest, receipt, environment

    @staticmethod
    def dry_run(environment):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=environment, capture_output=True, text=True)

    def test_dry_run_is_read_only_ignores_stale_recipe_and_reports_precisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            for expected in (
                "--nnodes=4 --nproc_per_node=8",
                CONFIG.name,
                "optimizer_updates=250000, student_updates=41666, critic_updates=208334",
                "Checkpoint: save_every_iters=50, save_total_limit=5",
                "global_microbatch=32, batch_mode=count_random",
                "rank=128, alpha=8",
                "lora_param_dtype=fp32",
                "fake: train_type=full",
                "Verified cache: completed=64, failed=0, input_rows=64",
            ):
                self.assertIn(expected, result.stdout)
            self.assertNotIn("/stale/", result.stdout)
            for role, dtype in (("student", "bf16"), ("fake", "fp32"), ("teacher", "bf16")):
                self.assertIn(f"precision {role}: transformer_param_dtype={dtype}, running_dtype=bf16, use_autocast=False", result.stdout)
            self.assertFalse((root / "outputs").exists())
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_default_output_rendezvous_and_overrides_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            environment.pop("H3_REF2AV_DMD_OUTPUT")
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("pdmd4_official_lorafp32_250000updates", result.stdout)
            self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_pdmd4_official_lorafp32_250000updates", result.stdout)
            environment["H3_REF2AV_DMD_OUTPUT"] = str(root / "custom-output")
            environment["H3_RDZV_ID"] = "custom-official-run"
            result = self.dry_run(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(root / "custom-output"), result.stdout)
            self.assertIn("--rdzv_id=custom-official-run", result.stdout)
            self.assertFalse((root / "custom-output").exists())

    def test_dedicated_override_allows_length_save_interval_and_lora_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            config = self.config(environment=environment)
            config["training"]["max_train_iters"] = 6000
            config["training"]["save_every_iters"] = 75
            config["training"]["student"]["lora"]["param_dtype"] = "bf16"
            custom = root / "custom-official.yaml"
            custom.write_text(yaml.safe_dump(config))
            result = self.dry_run({**environment, "H3_PDMD_OFFICIAL_CONFIG": str(custom)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("optimizer_updates=6000, student_updates=1000, critic_updates=5000", result.stdout)
            self.assertIn("Checkpoint: save_every_iters=75, save_total_limit=5", result.stdout)
            self.assertIn("lora_param_dtype=bf16", result.stdout)
            self.assertIn(str(custom), result.stdout)

    def test_old_recipe_override_cannot_launch_as_official(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, environment = self.fixture(Path(directory))
            result = self.dry_run({**environment, "H3_PDMD_OFFICIAL_CONFIG": str(OLD_CONFIG)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires the official PDMD training flow and losses", result.stderr)
            self.assertNotIn("--nnodes=4 --nproc_per_node=8", result.stdout)

    def test_invalid_receipts_fail_before_torchrun(self):
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
            self.assertIn("Missing completed cache merge receipt", result.stderr)
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
