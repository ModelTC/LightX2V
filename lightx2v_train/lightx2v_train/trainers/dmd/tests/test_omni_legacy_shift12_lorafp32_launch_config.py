"""CPU checks for the independent 32-GPU legacy DMD8 FP32-LoRA entry point."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32.yaml"
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_fsdp32.yaml"
BASE_LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_fsdp32_32gpu_acp.sh"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_omni_imageonly_dmd8_legacy_shift12_lorafp32_fsdp32_32gpu_acp.sh"
HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("omegaconf", "torch", "loguru"))


class LegacyShift12LoRAFP32ConfigTests(unittest.TestCase):
    def test_only_student_lora_parameter_dtype_differs_from_existing_32_gpu_recipe(self):
        baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
        current = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertNotIn("param_dtype", baseline["training"]["student"]["lora"])
        without_override = copy.deepcopy(current)
        self.assertEqual(without_override["training"]["student"]["lora"].pop("param_dtype"), "fp32")
        self.assertEqual(without_override, baseline)
        self.assertEqual(current["model"]["transformer_param_dtype"], "bf16")
        self.assertEqual(current["model"]["running_dtype"], "bf16")
        self.assertEqual(current["model"]["teacher"]["transformer_param_dtype"], "bf16")
        self.assertEqual(current["model"]["fake"]["transformer_param_dtype"], "fp32")
        fsdp = current["distributed"]["fsdp2"]
        self.assertEqual(fsdp["size"], 32)
        self.assertEqual(fsdp["mixed_precision"]["param_dtype"], "bf16")
        self.assertEqual(fsdp["mixed_precision"]["reduce_dtype"], "fp32")
        self.assertEqual(current["training"]["method"], "dmd")
        self.assertEqual(current["training"]["dmd"]["num_inference_steps"], 8)
        self.assertEqual(current["training"]["max_train_iters"], 100000)
        self.assertTrue(current["model"]["capabilities"]["distribution_matching"]["legacy_numerics"])
        sampler = current["data"]["train"]["reference_cost_sampler"]
        self.assertEqual(sampler["batch_mode"], "count_random")
        self.assertTrue(sampler["balance_image_counts"])
        self.assertFalse(sampler["balance_orientation"])

    def test_wrapper_shell_syntax_and_dedicated_config_override(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("H3_LEGACY_SHIFT12_LORA_FP32_CONFIG", script)
        self.assertIn(BASE_LAUNCHER.name, script)


@unittest.skipUnless(HAS_RUNTIME, "Launcher dry-run needs torch, OmegaConf and loguru, but no GPU")
class LegacyShift12LoRAFP32LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        transformer = self.root / "model/transformer_ref"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text("{}", encoding="utf-8")
        self.manifest = self.root / "metadata.jsonl"
        rows = [{"reference_image_count": 1, "target_orientation": "landscape", "source_id": str(index)} for index in range(32)]
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.manifest.with_suffix(".complete.json").write_text(
            json.dumps(
                {
                    "completed_count": 32,
                    "failed_count": 0,
                    "input_total_rows": 32,
                    "preprocess_fingerprint": "test-lora-fp32",
                    "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                    "reference_image_counts": {"1": 32},
                }
            ),
            encoding="utf-8",
        )
        self.env = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29597",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_REF2AV_CACHE": str(self.manifest),
            "H3_REF2AV_DMD_OUTPUT": str(self.root / "output"),
            "H3_CONFIG_PATH": "/stale/pdmd.yaml",
            "H3_LEGACY_SHIFT12_CONFIG": "/stale/legacy-bf16.yaml",
            "H3_PDMD": "true",
            "H3_REF2AV_EXPECTED_ROWS": "13553",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        for name in ("H3_LEGACY_SHIFT12_LORA_FP32_CONFIG", "H3_RDZV_ID", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE"):
            self.env.pop(name, None)

    def run_launcher(self, *, environment=None, launcher=LAUNCHER):
        return subprocess.run(["bash", str(launcher), "--dry-run"], env=self.env if environment is None else environment, capture_output=True, text=True, timeout=30)

    def test_dry_run_uses_fp32_lora_with_unchanged_legacy_numerics_and_4x8_topology(self):
        before = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        for value in (
            str(CONFIG),
            "Verified cache: completed=32",
            "plain DMD, steps=8, iters=100000",
            "legacy_numerics=True",
            "raw x0 = xt + sigma * velocity, without explicit dtype casts",
            "model_mode=legacy_train, update_order=student_first, fake_update_ratio=5",
            "PDMD=false",
            "--nnodes=4 --nproc_per_node=8",
            "precision student: transformer_param_dtype=bf16",
            "precision fake: transformer_param_dtype=fp32",
            "precision teacher: transformer_param_dtype=bf16",
            '"param_dtype": "fp32"',
            '"batch_mode": "count_random"',
            "orientations sampled naturally within that count",
        ):
            self.assertIn(value, result.stdout)
        self.assertNotIn("/stale/", result.stdout)
        self.assertNotIn("16 landscape + 16 portrait", result.stdout)
        self.assertFalse((self.root / "output").exists())
        after = {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(after, before, "Dry-run must not mutate model, cache, receipt or create training artifacts.")

    def test_defaults_have_independent_output_and_rendezvous_names(self):
        environment = dict(self.env)
        environment.pop("H3_REF2AV_DMD_OUTPUT")
        result = self.run_launcher(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"output=[^\n]*lora_?fp32[^\n]*")
        self.assertRegex(result.stdout, r"--rdzv_id=[^\s]*lora_?fp32[^\s]*")
        self.assertNotIn("--rdzv_id=h3_ref2av_omni_imageonly_dmd8_legacy_shift12_100k ", result.stdout)

    def test_output_and_rendezvous_can_be_overridden(self):
        environment = {
            **self.env,
            "H3_REF2AV_DMD_OUTPUT": str(self.root / "custom-output"),
            "H3_RDZV_ID": "custom-fp32-lora-run",
        }
        result = self.run_launcher(environment=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root / "custom-output"), result.stdout)
        self.assertIn("--rdzv_id=custom-fp32-lora-run", result.stdout)
        self.assertFalse((self.root / "custom-output").exists())

    def test_dedicated_override_wins_over_stale_generic_and_legacy_overrides(self):
        custom = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        custom["data"]["train"]["reference_cost_sampler"]["seed"] = 123
        custom_path = self.root / "custom-fp32-lora.yaml"
        custom_path.write_text(yaml.safe_dump(custom), encoding="utf-8")
        result = self.run_launcher(environment={**self.env, "H3_LEGACY_SHIFT12_LORA_FP32_CONFIG": str(custom_path)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(custom_path), result.stdout)
        self.assertIn('"seed": 123', result.stdout)
        self.assertIn('"param_dtype": "fp32"', result.stdout)
        self.assertNotIn("/stale/", result.stdout)

    def test_existing_launcher_keeps_original_bf16_lora_recipe(self):
        environment = {**self.env, "H3_LEGACY_SHIFT12_LORA_FP32_CONFIG": str(CONFIG)}
        environment.pop("H3_LEGACY_SHIFT12_CONFIG")
        result = self.run_launcher(environment=environment, launcher=BASE_LAUNCHER)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(BASE_CONFIG), result.stdout)
        self.assertNotIn(str(CONFIG), result.stdout)
        student_line = next(line for line in result.stdout.splitlines() if line.startswith("student: "))
        self.assertNotIn("param_dtype", json.loads(student_line.removeprefix("student: "))["lora"])
        self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)

    def test_changed_cache_receipt_still_fails_before_launch(self):
        self.manifest.write_text("{}\n", encoding="utf-8")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no longer matches", result.stderr)
        self.assertNotIn("--nnodes=4 --nproc_per_node=8", result.stdout)


if __name__ == "__main__":
    unittest.main()
