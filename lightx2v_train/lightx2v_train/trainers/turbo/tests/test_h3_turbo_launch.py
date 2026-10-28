"""CPU-only recipe and read-only ACP launch checks for H3 Turbo8."""

import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIGS = {
    "ref2av": TRAIN_ROOT / "configs/train/turbo/minimax_h3_ref2av_omni_imageonly_turbo8_shift12_3_r128_a128_adaln16_fsdp32.yaml",
    "fl2av": TRAIN_ROOT / "configs/train/turbo/minimax_h3_fl2av_3task_turbo8_shift6_3_r128_a128_adaln16_fsdp32.yaml",
}
LAUNCHERS = {task: TRAIN_ROOT / f"scripts/run_minimax_h3_{task}_turbo8_adaln16_fsdp32_32gpu_acp.sh" for task in CONFIGS}
_spec = importlib.util.spec_from_file_location("_h3_turbo_config_test", TRAIN_ROOT / "lightx2v_train/runtime/config.py")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


class H3TurboLaunchTests(unittest.TestCase):
    @staticmethod
    def config(task, environment=None):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/model",
                "H3_REF2AV_CACHE": "/ref-cache/metadata.jsonl",
                "H3_FL2AV_CACHE": "/fl-cache",
                "H3_CM_CACHE": "/cm-cache/metadata.jsonl",
                "H3_REF2AV_TURBO_OUTPUT": "/ref-output",
                "H3_FL2AV_TURBO_OUTPUT": "/fl-output",
                **(environment or {}),
            },
        ):
            return _config.load_config(str(CONFIGS[task]))

    @staticmethod
    def cm_fixture(root):
        cache = root / "cm"
        cache.mkdir()
        rows = []
        for index in range(64):
            height, width = (768, 1344) if index < 32 else (1344, 768)
            row = {
                "task": "t2av",
                "source_id": index,
                "cache_fingerprint": hashlib.sha256(str(index).encode()).hexdigest(),
                "target_num_frames": 124,
                "target_height": height,
                "target_width": width,
                "video_latent_shape": [24, 37, height // 16, width // 16],
                "audio_latent_shape": [2, 32, 207],
                "vae_normalized": True,
            }
            for key in ("condition_path", "video_latent_path", "audio_latent_path"):
                path = cache / f"{key}_{index:05d}.pt"
                path.write_bytes(b"metadata-only; never deserialize this fixture")
                row[key] = path.name
            rows.append(row)
        manifest = cache / "metadata.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return manifest

    @staticmethod
    def fixture(root, task):
        model = root / "model"
        component = model / ("transformer_ref" if task == "ref2av" else "transformer")
        component.mkdir(parents=True)
        (component / "config.json").write_text("{}")
        if task == "ref2av":
            cache = root / "metadata.jsonl"
            for payload in ("condition.pt", "video.pt", "audio.pt"):
                (root / payload).write_bytes(b"metadata-only; never deserialize this fixture")
            rows = [
                {
                    "source_id": str(index),
                    "reference_image_count": 1 + index // 32,
                    "reference_video_count": 0,
                    "reference_audio_count": 0,
                    "target_num_frames": 124,
                    "target_height": 768,
                    "target_width": 1344,
                    "packed_sequence_tokens_124": 18000 + index,
                    "condition_path": "condition.pt",
                    "video_latent_path": "video.pt",
                    "audio_latent_path": "audio.pt",
                }
                for index in range(64)
            ]
            cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
            receipt = {
                "completed_count": 64,
                "failed_count": 0,
                "input_total_rows": 64,
                "preprocess_fingerprint": "h3-turbo-fixture",
                "manifest_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
            }
            cache.with_suffix(".complete.json").write_text(json.dumps(receipt))
        else:
            cache = root / "cache"
            for name in ("t2av", "i2av", "fl2av"):
                for bucket, height, width in (("landscape", 768, 1344), ("portrait", 1344, 768)):
                    directory = cache / name / bucket
                    directory.mkdir(parents=True)
                    (directory / "condition.pt").write_bytes(b"metadata-only; never deserialize this fixture")
                    rows = [
                        {
                            "task": name,
                            "target_height": height,
                            "target_width": width,
                            "target_num_frames": 124,
                            "condition_path": "condition.pt",
                        }
                        for _ in range(16)
                    ]
                    (directory / "metadata.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        prefix = "H3_REF2AV" if task == "ref2av" else "H3_FL2AV"
        environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29596",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(model),
            prefix + "_CACHE": str(cache),
            prefix + "_TURBO_OUTPUT": str(root / "output"),
            "H3_CONFIG_PATH": "/stale/legacy.yaml",
            "H3_PDMD8_OFFICIAL_CONFIG": "/stale/pdmd8.yaml",
            "H3_FL2AV_PDMD_CONFIG": "/stale/fl2av.yaml",
            "H3_REF2AV_DMD_OUTPUT": "/stale/ref-output",
            "H3_FL2AV_PDMD_OUTPUT": "/stale/fl-output",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if task == "fl2av":
            environment["H3_CM_CACHE"] = str(H3TurboLaunchTests.cm_fixture(root))
        for name in ("H3_TURBO_CONFIG", "H3_KERNEL_SNAPSHOT", "KERNELS_CACHE", "LOCAL_KERNELS", "H3_RDZV_ID"):
            environment.pop(name, None)
        return cache, environment

    @staticmethod
    def dry_run(task, environment):
        return subprocess.run(["bash", str(LAUNCHERS[task]), "--dry-run"], env=environment, capture_output=True, text=True)

    def test_recipes_preserve_data_precision_and_select_block_and_output_adaln(self):
        for task in CONFIGS:
            with self.subTest(task=task):
                config = self.config(task)
                shifts = "12_3" if task == "ref2av" else "6_3"
                name = "ref2av_omni_imageonly" if task == "ref2av" else "fl2av_3task"
                old = yaml.safe_load((TRAIN_ROOT / f"configs/train/dmd/minimax_h3_{name}_pdmd8_shift{shifts}_r128_a128_adaln16_fsdp32.yaml").read_text())
                new = yaml.safe_load(CONFIGS[task].read_text())
                self.assertEqual(new["data"]["train"], old["data"]["train"])
                if task == "ref2av":
                    self.assertNotIn("cm", new["data"])
                else:
                    self.assertEqual(new["data"]["cm"]["tasks"], ["t2av"])
                    self.assertEqual(new["data"]["cm"]["name"], "minimax_h3_turbo_cm_dataset")
                    self.assertFalse(new["data"]["cm"]["drop_last"])
                    self.assertEqual(new["data"]["cm"]["path_mapping"], {"/data/nvme6/gushiqiao": "/mnt/lm_data_afs/gushiqiao"})
                self.assertEqual(new["distributed"], old["distributed"])
                self.assertEqual(new["training"]["fake"]["train_type"], "full")
                for role, learning_rate in (("student", 2e-5), ("fake", 4e-6)):
                    optimizer = new["training"][role]["optimizer"]
                    self.assertEqual(optimizer["learning_rate"], learning_rate)
                    self.assertEqual(optimizer["adam_beta2"], 0.999)
                    self.assertEqual(optimizer["weight_decay"], 0.01)
                self.assertEqual(config["training"]["method"], "turbo")
                self.assertNotIn("dmd", config["training"])
                lora = config["training"]["student"]["lora"]
                self.assertEqual((lora["rank"], lora["alpha"], lora["param_dtype"]), (128, 128, "fp32"))
                self.assertEqual(lora["rank_pattern"], {"adaln_proj.linear": 16, "norm_out.linear": 16})
                self.assertEqual(lora["alpha_pattern"], lora["rank_pattern"])
                self.assertIn("norm_out.linear", lora["target_modules"])
                self.assertEqual(config["model"]["transformer_param_dtype"], "bf16")
                self.assertEqual(config["model"]["fake"]["transformer_param_dtype"], "fp32")
                self.assertEqual(config["model"]["teacher"]["transformer_param_dtype"], "bf16")
                matching = config["model"]["capabilities"]["distribution_matching"]
                self.assertNotIn("official_pdmd", matching)
                self.assertNotIn("projected_dmd", matching)
                self.assertEqual(config["training"]["teacher"]["guidance_scale"], 1.0)

    def test_curriculum_counts_student_updates_and_uses_ema(self):
        for task in CONFIGS:
            training = self.config(task)["training"]
            turbo = training["turbo"]
            self.assertEqual(turbo["num_inference_steps"], 8)
            self.assertEqual([turbo[key] for key in ("dcm_steps", "scm_steps", "joint_steps")], [2000, 500, 4500])
            self.assertEqual(training["max_train_iters"], 7000)
            self.assertEqual(turbo["fake_update_ratio"], 4)
            self.assertEqual(turbo["dcm_total_steps"], 48)
            self.assertEqual(turbo["dcm_skip_steps"], 1)
            self.assertEqual((turbo["generator_time_mean"], turbo["generator_time_std"]), (-0.8, 1.6))
            self.assertEqual((turbo["score_time_mean"], turbo["score_time_std"]), (0.0, 1.6))
            self.assertEqual(training["student"]["ema"], {"enabled": True, "decay": 0.999, "use_for_inference": True})

    def test_bash_syntax(self):
        for launcher in [*LAUNCHERS.values(), TRAIN_ROOT / "scripts/run_minimax_h3_turbo_fsdp32_32gpu_acp.sh"]:
            result = subprocess.run(["bash", "-n", str(launcher)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_dry_runs_are_read_only_and_ignore_pdmd_environment(self):
        for task in CONFIGS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, environment = self.fixture(root, task)
                before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                result = self.dry_run(task, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (
                    CONFIGS[task].name,
                    "--nnodes=4 --nproc_per_node=8",
                    "steps=8, global_batch=32",
                    "dCM=2000, sCM=500, joint=4500, total=7000",
                    "joint fake_update_ratio=4",
                    "max_train_iters counts STUDENT",
                    "block/output AdaLN rank=16 alpha=16",
                    "fake full FP32",
                    "direct conditional teacher, no CFG",
                ):
                    self.assertIn(expected, result.stdout)
                self.assertNotIn("/stale/", result.stdout)
                self.assertFalse((root / "output").exists())
                after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                self.assertEqual(before, after)
                if task == "ref2av":
                    self.assertIn("Verified Ref cache: completed=64", result.stdout)
                    self.assertIn("Student base=transformer_ref/ BF16", result.stdout)
                else:
                    self.assertIn('"t2av": {"landscape": 16, "portrait": 16}', result.stdout)
                    self.assertIn("Student base=transformer/ BF16", result.stdout)
                    self.assertIn("Verified CM cache: task=T2AV, rows=64", result.stdout)
                    self.assertIn("CM=T2AV clean AV (7000 global batches)", result.stdout)
                    self.assertIn("condition-only (22500 global batches)", result.stdout)

    def test_invalid_ref_receipts_fail_before_torchrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, environment = self.fixture(root, "ref2av")
            path = manifest.with_suffix(".complete.json")
            original = json.loads(path.read_text())
            for update, message in (
                ({"manifest_sha256": "0" * 64}, "matching its complete receipt"),
                ({"input_total_rows": 65}, "completed/failed/source accounting"),
                ({"preprocess_fingerprint": ""}, "preprocessing fingerprint"),
            ):
                with self.subTest(message=message):
                    path.write_text(json.dumps({**original, **update}))
                    result = self.dry_run("ref2av", environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("--nnodes=4", result.stdout)
            path.unlink()
            result = self.dry_run("ref2av", environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("completed cache merge receipt", result.stderr)

    def test_missing_fl_condition_and_task_routes_fail_before_torchrun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, environment = self.fixture(root, "fl2av")
            condition = cache / "i2av/portrait/condition.pt"
            condition.unlink()
            result = self.dry_run("fl2av", environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("readable cached condition_path", result.stderr)
            condition.write_bytes(b"metadata-only")
            (cache / "fl2av/portrait/metadata.jsonl").unlink()
            result = self.dry_run("fl2av", environment)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("both orientations", result.stderr)
            self.assertNotIn("--nnodes=4", result.stdout)

    def test_rejects_old_method_step_count_and_shift_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, environment = self.fixture(root, "ref2av")
            original = self.config("ref2av", environment)
            for change, expected in (
                (lambda c: c["training"].update(method="dmd"), "training.method=turbo"),
                (lambda c: c["training"]["turbo"].update(num_inference_steps=4), "8-step Turbo"),
                (lambda c: c["training"].update(max_train_iters=250000), "dcm_steps + scm_steps + joint_steps (STUDENT updates)"),
                (lambda c: c["training"]["turbo"].update(video_shift=6.0), "differs between training and model metadata"),
                (lambda c: c["training"]["student"]["lora"]["rank_pattern"].pop("norm_out.linear"), "block and output AdaLN"),
            ):
                with self.subTest(expected=expected):
                    config = copy.deepcopy(original)
                    change(config)
                    path = root / "override.yaml"
                    path.write_text(yaml.safe_dump(config))
                    result = self.dry_run("ref2av", {**environment, "H3_TURBO_CONFIG": str(path)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                    self.assertNotIn("--nnodes=4", result.stdout)

    def test_independent_output_and_rendezvous_overrides(self):
        for task in CONFIGS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, environment = self.fixture(root, task)
                prefix = "H3_REF2AV" if task == "ref2av" else "H3_FL2AV"
                environment.pop(prefix + "_TURBO_OUTPUT")
                result = self.dry_run(task, environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected_rdzv = "h3_ref2av_turbo8_adaln16_32gpu" if task == "ref2av" else "h3_fl2av_turbo8_adaln16_cm_t2av_9424_32gpu"
                self.assertIn(f"--rdzv_id={expected_rdzv}", result.stdout)
                self.assertIn("outputs/h3_", result.stdout)
                result = self.dry_run(task, {**environment, prefix + "_TURBO_OUTPUT": str(root / "custom"), "H3_RDZV_ID": "custom-turbo-job"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--rdzv_id=custom-turbo-job", result.stdout)
                self.assertIn(str(root / "custom"), result.stdout)
                self.assertFalse((root / "custom").exists())

    def test_relative_config_override_survives_training_workdir_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, environment = self.fixture(root, "ref2av")
            config = self.config("ref2av", environment)
            config["training"]["turbo"].update(dcm_steps=10, scm_steps=2, joint_steps=3)
            config["training"]["max_train_iters"] = 15
            path = root / "custom-turbo.yaml"
            path.write_text(yaml.safe_dump(config))
            relative = os.path.relpath(path)
            result = self.dry_run("ref2av", {**environment, "H3_TURBO_CONFIG": relative})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dCM=10, sCM=2, joint=3, total=15", result.stdout)
            command = shlex.split(result.stdout.splitlines()[-1])
            self.assertEqual(command[command.index("--config") + 1], str(path))

    def test_all_rows_require_readable_clean_video_and_audio(self):
        for task in CONFIGS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache, environment = self.fixture(root, task)
                manifest = cache if task == "ref2av" else Path(environment["H3_CM_CACHE"])
                original = [json.loads(line) for line in manifest.read_text().splitlines()]
                for key in ("video_latent_path", "audio_latent_path"):
                    rows = copy.deepcopy(original)
                    rows[-1][key] = "missing-clean-latent.pt"
                    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    if task == "ref2av":
                        receipt_path = manifest.with_suffix(".complete.json")
                        receipt = json.loads(receipt_path.read_text())
                        receipt["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
                        receipt_path.write_text(json.dumps(receipt))
                    result = self.dry_run(task, environment)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(key, result.stderr)
                    self.assertNotIn("--nnodes=4", result.stdout)

    def test_real_schema_preflight_does_not_import_torch(self):
        for task in CONFIGS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                _, environment = self.fixture(Path(directory), task)
                script = TRAIN_ROOT / "scripts/minimax_h3_turbo_preflight.py"
                source = f"import runpy,sys; sys.argv=[{str(script)!r}, {task!r}, {str(CONFIGS[task])!r}]; runpy.run_path({str(script)!r}, run_name='__main__'); assert 'torch' not in sys.modules"
                result = subprocess.run([sys.executable, "-c", source], env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_cm_absolute_paths_remap_without_rewriting_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, environment = self.fixture(root, "fl2av")
            manifest = Path(environment["H3_CM_CACHE"])
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            old_prefix = Path("/old-machine/cm")
            for row in rows:
                for key in ("condition_path", "video_latent_path", "audio_latent_path"):
                    row[key] = str(old_prefix / row[key])
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            config = self.config("fl2av", environment)
            config["data"]["cm"]["path_mapping"] = {str(old_prefix): str(manifest.parent)}
            override = root / "mapped-cm.yaml"
            override.write_text(yaml.safe_dump(config))
            before = manifest.read_bytes()
            result = self.dry_run("fl2av", {**environment, "H3_TURBO_CONFIG": str(override)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Verified CM cache: task=T2AV, rows=64", result.stdout)
            self.assertEqual(manifest.read_bytes(), before)
            self.assertFalse((root / "output").exists())

    def test_real_cm_data_builder_accepts_recipe_and_rejects_dropping_rows(self):
        from lightx2v_train.utils.registry import build_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, environment = self.fixture(root, "fl2av")
            config = self.config("fl2av", environment)
            loader = build_data(config, "cm")
            self.assertEqual(len(loader.dataset), 64)
            self.assertEqual(loader.batch_size, 1)
            self.assertFalse(loader.drop_last)
            loader.sampler.configure_from_microbatch_offset(start_microbatch=0)
            self.assertEqual(sorted(loader.sampler), list(range(64)))
            config["data"]["cm"]["drop_last"] = True
            with self.assertRaisesRegex(ValueError, "drop_last must be false"):
                build_data(config, "cm")
            override = root / "invalid-drop-last.yaml"
            override.write_text(yaml.safe_dump(config))
            result = self.dry_run("fl2av", {**environment, "H3_TURBO_CONFIG": str(override)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("drop_last", result.stderr)
            self.assertNotIn("--nnodes=4", result.stdout)


if __name__ == "__main__":
    unittest.main()
