"""CPU-only config and ACP dry-run checks for Ref DMD optimization plus HEAD."""

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
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_head8_match124_mixed_fsdp32.yaml"
REFERENCE_CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_dmd_lora_match124_image_audio_1to5_uniform_fsdp32_32gpu_full_fake.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_head8_fsdp32_32gpu_acp.sh"


def load_recipe(path=CONFIG, environment=None):
    with patch.dict(
        os.environ,
        {
            "H3_MODEL_PATH": "/model",
            "H3_REF2AV_CACHE": "/cache/metadata.jsonl",
            "H3_REF2AV_DMD_OUTPUT": "/outputs",
            # A stale PDMD environment must not turn this recipe into PDMD.
            "H3_PDMD": "true",
            **(environment or {}),
        },
    ):
        return load_config(str(path))


class OmniImageOnlyHeadConfigTests(unittest.TestCase):
    def test_uses_original_ref_optimizer_and_lora_recipe(self):
        training = load_recipe()["training"]
        reference = load_recipe(REFERENCE_CONFIG, {"H3_PDMD": "false"})["training"]
        for name, learning_rate in (("student", 5e-5), ("fake", 4e-7)):
            with self.subTest(role=name):
                self.assertEqual(training[name], reference[name])
                optimizer = training[name]["optimizer"]
                self.assertEqual(optimizer["learning_rate"], learning_rate)
                self.assertEqual(optimizer["adam_beta1"], 0.0)
                self.assertEqual(optimizer["adam_beta2"], 0.999)
                self.assertEqual(optimizer["weight_decay"], 0.01)
        self.assertEqual(training["student"]["train_type"], "lora")
        self.assertEqual(training["student"]["lora"]["rank"], 128)
        self.assertEqual(training["student"]["lora"]["alpha"], 8)
        self.assertEqual(training["fake"]["train_type"], "full")

    def test_head_is_enabled_calibrated_fake_first_without_projection(self):
        config = load_recipe()
        training = config["training"]
        self.assertEqual(training["method"], "dmd")
        self.assertIs(config["model"]["capabilities"]["distribution_matching"]["projected_dmd"], False)
        self.assertEqual(training["dmd"]["update_order"], "fake_first")
        self.assertEqual(training["dmd"]["fake_update_ratio"], 5)
        head = training["dmd"]["residual_head"]
        self.assertIs(head["enabled"], True)
        self.assertEqual(head["gate_mode"], "calibrated")
        self.assertEqual(head["fit_steps"], 1)
        self.assertEqual(head["fit_grad_accum_steps"], 1)
        self.assertEqual(ResidualHeadConfig.from_mapping(head).checkpoint_metadata(), head)
        for key, value in {
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "noise_bins": 5,
            "ema_decay": 0.9,
            "min_checks": 3,
            "gate_ramp": 0.25,
            "max_grad_norm": 1.0,
            "weight_decay": 0.0,
            "min_relative_improvement": 0.0,
        }.items():
            self.assertEqual(head[key], value, key)

    def test_nfe_iteration_and_accumulation_defaults(self):
        training = load_recipe()["training"]
        self.assertEqual(training["max_train_iters"], 10000)
        self.assertEqual(training["gradient_accumulation_iters"], 1)
        self.assertEqual(training["dmd"]["num_inference_steps"], 8)
        self.assertEqual(training["dmd"]["latent_dtype"], "fp32")

    def test_trainable_fp32_masters_bf16_compute_and_bf16_teacher(self):
        config = load_recipe()
        model = config["model"]
        self.assertEqual(model["transformer_param_dtype"], "fp32")
        self.assertEqual(model["fake"]["transformer_param_dtype"], "fp32")
        self.assertEqual(model["teacher"]["transformer_param_dtype"], "bf16")
        mixed = config["distributed"]["fsdp2"]["mixed_precision"]
        for name in ("student", "fake", "teacher"):
            override = {} if name == "student" else model.get(name, {})
            effective_model = {**model, **override}
            effective_mixed = {**mixed, **override.get("distributed", {}).get("fsdp2", {}).get("mixed_precision", {})}
            with self.subTest(role=name):
                self.assertEqual(effective_model["running_dtype"], "bf16")
                self.assertEqual(effective_mixed["param_dtype"], "bf16")
                self.assertEqual(effective_mixed["reduce_dtype"], "fp32")
        self.assertIs(config["distributed"]["fsdp2"]["stream_load_pretrained"], True)

    def test_32_gpu_count_coverage_and_metadata_geometry_are_preserved(self):
        config = load_recipe()
        distributed = config["distributed"]
        self.assertIs(distributed["fsdp2"]["enabled"], True)
        self.assertEqual(distributed["fsdp2"]["size"], 32)
        self.assertIs(distributed["sequence_parallel"]["enabled"], False)
        data = config["data"]["train"]
        self.assertEqual(data["batch_size"], 1)
        self.assertEqual(data["num_workers"], 8)
        self.assertIs(data["pin_memory"], True)
        sampler = data["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "count_coverage")
        self.assertEqual(sampler["image_counts"], list(range(1, 10)))
        self.assertIs(sampler["require_image_only"], True)
        self.assertIs(sampler["balance_image_counts"], False)
        self.assertIs(sampler["balance_orientation"], False)
        self.assertIs(sampler["require_all_image_counts"], False)
        self.assertEqual(sampler["remainder_policy"], "rotating_drop")
        matching = config["model"]["capabilities"]["distribution_matching"]
        self.assertIs(matching["geometry_from_metadata"], True)
        self.assertEqual(matching["allowed_resolutions"], [[768, 1344], [1344, 768]])
        self.assertEqual(config["training"]["dmd"]["generation_shapes"], [{"value": [124, 768, 1344]}])


class OmniImageOnlyHeadLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        transformer = self.root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        self.manifest = self.root / "metadata.jsonl"
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.receipt = self.manifest.with_suffix(".complete.json")
        self.receipt.write_text(
            json.dumps(
                {
                    "completed_count": 1,
                    "failed_count": 0,
                    "input_total_rows": 1,
                    "preprocess_fingerprint": "fixture-fingerprint",
                    "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                    "reference_image_counts": {"1": 1},
                }
            ),
            encoding="utf-8",
        )
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
        for key in ("H3_CONFIG_PATH", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE", "H3_RDZV_ID"):
            self.environment.pop(key, None)

    def run_launcher(self, config=None, overrides=None):
        environment = {**self.environment, **(overrides or {})}
        if config is not None:
            path = self.root / "custom.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            environment["H3_CONFIG_PATH"] = str(path)
        return subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def custom_config(self):
        return load_recipe(environment=self.environment)

    def test_shell_syntax(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_dry_run_reports_actual_recipe_without_launching(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "--nnodes=4 --nproc_per_node=8",
            "DMD + calibrated HEAD, steps=8, iters=10000, grad_accum=1",
            "update_order=fake_first, fake_update_ratio=5",
            "global_microbatch=32, batch_mode=count_coverage",
            "Verified cache: completed=1, failed=0, input_rows=1",
            "student: train_type=lora, rank=128, alpha=8",
            "fake: train_type=full",
            '"learning_rate": 5e-05',
            '"learning_rate": 4e-07',
            '"gate_mode": "calibrated"',
            '"fit_steps": 1',
            '"fit_grad_accum_steps": 1',
        ):
            self.assertIn(expected, result.stdout)
        for role, dtype in (("student", "fp32"), ("fake", "fp32"), ("teacher", "bf16")):
            self.assertIn(f"precision {role}: transformer_param_dtype={dtype}, running_dtype=bf16", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_default_output_and_rendezvous_are_head_specific(self):
        result = self.run_launcher(overrides={"H3_REF2AV_DMD_OUTPUT": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("outputs/minimax_h3_ref2av_omni_imageonly_head8_legacy_10k", result.stdout)
        self.assertIn("--rdzv_id=h3_ref2av_omni_imageonly_head8_legacy_10k", result.stdout)
        self.assertIn(str(CONFIG), result.stdout)

    def test_head_algorithm_incompatible_overrides_are_rejected(self):
        cases = (
            (("training", "dmd", "residual_head", "enabled"), False, "enabled calibrated HEAD"),
            (("training", "dmd", "residual_head", "gate_mode"), "full", "enabled calibrated HEAD"),
            (("model", "capabilities", "distribution_matching", "projected_dmd"), True, "no PDMD projection"),
            (("training", "dmd", "update_order"), "student_first", "fake-first HEAD updates"),
        )
        for path, value, message in cases:
            with self.subTest(setting=".".join(path)):
                config = self.custom_config()
                cursor = config
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                result = self.run_launcher(config)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_experiment_hyperparameters_remain_configurable_and_reported(self):
        config = self.custom_config()
        training = config["training"]
        training["max_train_iters"] = 17
        training["gradient_accumulation_iters"] = 2
        training["dmd"]["num_inference_steps"] = 4
        training["dmd"]["fake_update_ratio"] = 2
        training["dmd"]["residual_head"]["fit_steps"] = 3
        training["student"]["lora"].update(rank=16, alpha=2)
        training["student"]["optimizer"].update(learning_rate=7e-6, adam_beta2=0.95, weight_decay=0.02)
        training["fake"]["optimizer"]["learning_rate"] = 9e-7
        config["model"]["transformer_param_dtype"] = "bf16"
        config["data"]["train"].update(num_workers=0, pin_memory=False)
        result = self.run_launcher(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "steps=4, iters=17, grad_accum=2",
            "fake_update_ratio=2",
            "rank=16, alpha=2",
            '"learning_rate": 7e-06',
            '"learning_rate": 9e-07',
            '"adam_beta2": 0.95',
            '"weight_decay": 0.02',
            '"fit_steps": 3',
            "data_workers=0, pin_memory=False",
            "precision student: transformer_param_dtype=bf16",
        ):
            self.assertIn(expected, result.stdout)

    def test_gpu_layout_and_coverage_guards_remain_active(self):
        for change, message in (("fsdp", "FSDP32"), ("coverage", "no category/orientation undersampling")):
            with self.subTest(change=change):
                config = self.custom_config()
                if change == "fsdp":
                    config["distributed"]["fsdp2"]["size"] = 40
                else:
                    config["data"]["train"]["reference_cost_sampler"]["balance_image_counts"] = True
                result = self.run_launcher(config)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_role_precision_overrides_are_reported_without_restrictions(self):
        config = self.custom_config()
        config["model"]["fake"].update(transformer_param_dtype="bf16", running_dtype="fp32", use_autocast=True)
        config["model"]["fake"]["distributed"] = {"fsdp2": {"mixed_precision": {"param_dtype": "fp32"}}}
        result = self.run_launcher(config)
        self.assertEqual(result.returncode, 0, result.stderr)
        mixed = {**config["distributed"]["fsdp2"]["mixed_precision"], "param_dtype": "fp32"}
        self.assertIn(
            f"precision fake: transformer_param_dtype=bf16, running_dtype=fp32, use_autocast=True, fsdp={json.dumps(mixed, sort_keys=True)}\n",
            result.stdout,
        )

    def test_manifest_receipt_and_expected_row_checks_remain_active(self):
        result = self.run_launcher(overrides={"H3_REF2AV_EXPECTED_ROWS": "100000"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 100000, found 1", result.stderr)
        self.manifest.write_text('{"changed": true}\n', encoding="utf-8")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer matches its complete receipt", result.stderr)
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.receipt.rename(self.root / "receipt.backup.json")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing completed 32-shard merge receipt", result.stderr)


if __name__ == "__main__":
    unittest.main()
