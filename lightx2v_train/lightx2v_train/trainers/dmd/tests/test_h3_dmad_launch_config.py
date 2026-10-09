"""CPU-only checks for the independent DMAD8 recipe and ACP preflight."""

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
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_dmad8_fsdp32.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_dmad8_fsdp32_32gpu_acp.sh"
PREFLIGHT = TRAIN_ROOT / "scripts/check_minimax_h3_dmad_launch.py"


class H3DmadLaunchConfigTests(unittest.TestCase):
    def config(self, **overrides):
        environment = {
            "H3_MODEL_PATH": "/path/to/model",
            "H3_DMAD_CACHE": "/path/to/dmad/paired.jsonl",
            "H3_DMAD_OUTPUT": "/path/to/new/dmad-output",
            "H3_PDMD": "true",  # Must not turn this recipe into PDMD.
            **overrides,
        }
        with patch.dict(os.environ, environment):
            if "H3_DMAD_MAX_ITERS" not in overrides:
                os.environ.pop("H3_DMAD_MAX_ITERS", None)
            return load_config(str(CONFIG))

    def test_recipe_is_dmad_not_online_teacher_dmd(self):
        config = self.config()
        model, training = config["model"], config["training"]
        self.assertEqual(training["method"], "dmad")
        self.assertEqual(model["name"], "minimax_h3_ref2av")
        self.assertNotIn("teacher", model)
        self.assertNotIn("teacher", training)
        self.assertEqual(config["data"]["train"]["name"], "minimax_h3_dmad_dataset")
        self.assertEqual(int(training["max_train_iters"]), 800)
        self.assertEqual(int(self.config(H3_DMAD_MAX_ITERS="1000")["training"]["max_train_iters"]), 1000)
        self.assertEqual(training["gradient_accumulation_iters"], 1)
        self.assertEqual(training["max_grad_norm"], 0)
        dmd = training["dmd"]
        self.assertEqual((dmd["num_inference_steps"], dmd["fake_update_ratio"], dmd["update_order"]), (8, 1, "student_first"))
        self.assertFalse(dmd["residual_head"]["enabled"])
        matching = model["capabilities"]["distribution_matching"]
        self.assertFalse(matching["projected_dmd"])
        self.assertEqual((matching["video_flow_shift"], matching["audio_flow_shift"]), (12.0, 2.0))
        self.assertEqual(matching["fixed_num_frames"], 124)
        self.assertEqual(training["dmad"]["ema_gammas"], [6.94, 16.97])
        self.assertEqual(training["dmad"]["feature_block"], 49)
        self.assertTrue(training["dmad"]["gap_sync"])
        self.assertEqual((training["dmad"]["lambda_real"], training["dmad"]["lambda_teacher"]), (1.0, 1.0))
        self.assertEqual((training["dmad"]["renoise_sigma_min"], training["dmad"]["renoise_sigma_max"]), (0.02, 0.98))

    def test_optimizer_precision_and_ref_sampler_are_explicit(self):
        config = self.config()
        self.assertEqual(config["model"]["transformer_param_dtype"], "fp32")
        self.assertEqual(config["model"]["fake"]["transformer_param_dtype"], "fp32")
        fsdp = config["distributed"]["fsdp2"]
        self.assertEqual(fsdp["size"], 32)
        self.assertEqual(fsdp["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(fsdp["mixed_precision"]["reduce_dtype"], "fp32")
        self.assertEqual(config["distributed"]["sequence_parallel"], {"enabled": False, "size": 1})
        self.assertFalse(config["training"]["student"]["ema"]["enabled"])
        for role in ("student", "fake"):
            settings = config["training"][role]
            self.assertEqual(settings["train_type"], "lora")
            self.assertEqual((settings["lora"]["rank"], settings["lora"]["alpha"]), (128, 128))
            optimizer = settings["optimizer"]
            self.assertEqual(optimizer["learning_rate"], 4e-5)
            self.assertEqual((optimizer["adam_beta1"], optimizer["adam_beta2"], optimizer["weight_decay"]), (0.0, 0.99, 0.01))
        sampler = config["data"]["train"]["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "cost_local")
        self.assertEqual(sampler["image_counts"], [1, 2, 3, 4, 5, 6])
        self.assertTrue(sampler["balance_image_counts"])
        self.assertTrue(sampler["balance_orientation"])

    def fixture(self, root):
        model = root / "model/transformer_ref"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        rows = []
        for index in range(32):
            row = {
                "cache_fingerprint": f"exact-condition-{index}",
                "target_height": 768 if index < 16 else 1344,
                "target_width": 1344 if index < 16 else 768,
                "target_num_frames": 124,
                "reference_image_count": 1,
                "reference_video_count": 0,
                "reference_audio_count": 0,
                "packed_sequence_tokens_124": 16000,
                "dmad_schema_version": 1,
            }
            for role in ("condition", "real_latent", "teacher_latent"):
                path = root / f"{role}-{index}.pt"
                # These are deliberately NOT torch payloads: preflight must not deserialize.
                path.write_bytes(b"path-existence-check-only")
                row[f"{role}_path"] = path.name
            rows.append(row)
        manifest = root / "paired.jsonl"
        self.write_rows(manifest, rows)
        environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29597",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(root / "model"),
            "H3_DMAD_CACHE": str(manifest),
            "H3_DMAD_OUTPUT": str(root / "output"),
            "H3_CONFIG_PATH": "/stale/dmd/config.yaml",
            "H3_PDMD": "true",
            "H3_DMAD_MAX_ITERS": "801",
            "PYTHONPATH": str(TRAIN_ROOT),
        }
        for key in ("H3_DMAD_CONFIG", "KERNELS_CACHE", "H3_KERNEL_SNAPSHOT"):
            environment.pop(key, None)
        return rows, manifest, environment

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    @staticmethod
    def launch(environment):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=environment, capture_output=True, text=True)

    def test_dry_run_checks_pairs_without_loading_tensors_or_launching_torchrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            result = self.launch(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("paired_rows=32", result.stdout)
            self.assertIn("iters=801", result.stdout)
            self.assertIn("steps=8", result.stdout)
            self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            self.assertIn("No online teacher", result.stdout)
            self.assertNotIn("/stale/dmd/config.yaml", result.stdout)
            self.assertFalse((root / "output").exists())
            command = "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__'); assert 'torch' not in sys.modules"
            checked = subprocess.run([sys.executable, "-c", command, str(PREFLIGHT), str(CONFIG)], env=environment, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_preflight_rejects_condition_only_missing_paths_and_unbalanced_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, manifest, environment = self.fixture(root)
            for key in ("real_latent_path", "teacher_latent_path"):
                with self.subTest(missing=key):
                    rows = [dict(row) for row in original]
                    rows[0].pop(key)
                    self.write_rows(manifest, rows)
                    result = self.launch(environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("condition-only", result.stderr)
                    self.assertIn(key, result.stderr)
            rows = [dict(row) for row in original]
            rows[0]["teacher_latent_path"] = "does-not-exist.pt"
            self.write_rows(manifest, rows)
            result = self.launch(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing file", result.stderr)
            self.write_rows(manifest, original[:16])
            result = self.launch(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least 16 rows", result.stderr)

    def test_preflight_rejects_old_recipe_and_allows_custom_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, manifest, environment = self.fixture(root)
            with patch.dict(os.environ, environment):
                config = load_config(str(CONFIG))
            config["training"]["method"] = "dmd"
            custom = root / "custom.yaml"
            custom.write_text(yaml.safe_dump(config))
            environment["H3_DMAD_CONFIG"] = str(custom)
            result = self.launch(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("method=dmad", result.stderr)
            config["training"]["method"] = "dmad"
            config["data"]["train"].pop("reference_cost_sampler")
            custom.write_text(yaml.safe_dump(config))
            self.write_rows(manifest, rows[:1])
            result = self.launch(environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("paired_rows=1", result.stdout)

    def test_preflight_rejects_runtime_incompatible_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, environment = self.fixture(root)
            custom = root / "custom.yaml"
            environment["H3_DMAD_CONFIG"] = str(custom)
            cases = (
                (("training", "dmd", "num_inference_steps"), 4, "num_inference_steps=8"),
                (("training", "dmd", "update_order"), "fake_first", "student_first"),
                (("training", "dmd", "random_schedule"), {"enabled": True}, "re-noise schedule"),
                (("training", "student", "ema"), {"enabled": True}, "power-function EMA"),
                (("training", "dmad", "gap_sync"), False, "gap_sync=true"),
                (("inference", "infer_every_iters"), 100, "Euler inferencer"),
            )
            for keys, value, message in cases:
                with self.subTest(keys=keys), patch.dict(os.environ, environment):
                    config = load_config(str(CONFIG))
                    target = config
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = value
                    custom.write_text(yaml.safe_dump(config))
                    result = self.launch(environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)

    def test_preflight_checks_configured_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, manifest, environment = self.fixture(root)
            rows = [dict(row) for row in original]
            rows[0]["target_num_frames"] = 107
            self.write_rows(manifest, rows)
            result = self.launch(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fixed_num_frames=124", result.stderr)
            rows = [dict(row) for row in original]
            rows[0]["target_height"] = 512
            rows[0]["target_width"] = 896
            self.write_rows(manifest, rows)
            result = self.launch(environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allowed_resolutions", result.stderr)


if __name__ == "__main__":
    unittest.main()
