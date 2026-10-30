"""CPU-only full-cache and two-stage ACP launch contracts."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_dcm_pdmd8_decfg_fsdp32_32gpu_acp.sh"


class H3DcmPdmdLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        model = self.root / "model"
        for component in ("transformer", "transformer_ref"):
            directory = model / component
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}")
        presets = self.root / "presets"
        presets.mkdir()
        for name in ("model_for_comfy_dit.safetensors", "model_ref2va_for_comfy_dit.safetensors"):
            (presets / name).write_bytes(b"preflight must not deserialize model tensors")
        self.rows = {}
        for subset in ("t2av", "ref2av"):
            geometries = ((768, 1344), (1184, 864), (1024, 992)) + (((1184, 864),) if subset == "ref2av" else ())
            rows = [self.row(subset, index, height=height, width=width, image_count=2 if subset == "ref2av" and index == 3 else None) for index, (height, width) in enumerate(geometries)]
            self.rows[subset] = rows
            self.manifest(f"new/{subset}/metadata.jsonl", rows, receipt=True)
            self.manifest(f"new/{subset}/condition_metadata.jsonl", [self.condition_only(row) for row in rows], receipt=True)
        image_only = [row for row in self.rows["ref2av"] if row["reference_image_count"] > 0 and row["reference_video_count"] == row["reference_audio_count"] == 0]
        self.rows["ref2av_imageonly"] = image_only
        self.manifest("new/ref2av_imageonly/metadata.jsonl", image_only, receipt=True)
        self.manifest("new/ref2av_imageonly/condition_metadata.jsonl", [self.condition_only(row) for row in image_only], receipt=True)
        for index, task in enumerate(("t2av", "i2av", "fl2av")):
            rows = [self.row(task, index + 10, height=768, width=1344), self.row(task, index + 20, height=1344, width=768)]
            self.manifest(f"old_fl/{task}/metadata.jsonl", [self.condition_only(row) for row in rows])
        old_ref = [self.row("ref2av", index + 30, height=768, width=1344, image_count=index + 1) for index in range(2)]
        self.manifest("old_ref/metadata.jsonl", [self.condition_only(row) for row in old_ref])
        self.environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29996",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(model),
            "H3_SELFGEN_CACHE": str(self.root / "new"),
            "H3_FL2AV_CACHE": str(self.root / "old_fl"),
            "H3_REF2AV_CACHE": str(self.root / "old_ref/metadata.jsonl"),
            "H3_TRAINING_ADAPTER_DIR": str(presets),
            # This dedicated route always enables DeCFG, regardless of an old toggle.
            "H3_TRAINING_ADAPTER_ENABLE": "false",
            "H3_CONFIG_PATH": "/stale/dmd.yaml",
            "H3_TURBO_CONFIG": "/stale/turbo.yaml",
            "H3_CURRICULUM_CONFIG": "/stale/curriculum.yaml",
            "H3_FL2AV_PDMD_CONFIG": "/stale/fl2av.yaml",
            "LOCAL_KERNELS": "",
            "H3_KERNEL_SNAPSHOT": "",
            "KERNELS_CACHE": "",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
        for name in ("H3_CM_CACHE", "H3_NEW_CACHE", "H3_STUDENT_INIT", "H3_TRAINING_ADAPTER_PATH", "H3_TRAINING_ADAPTER_SCALE", "H3_STAGED_CM_CONFIG", "H3_STAGED_PDMD_CONFIG", "H3_PDMD_MASTER_PORT"):
            self.environment.pop(name, None)

    def row(self, task, index, *, height, width, image_count=None):
        directory = self.root / "payloads"
        directory.mkdir(exist_ok=True)
        identity = f"{task}-{index}"
        payload = {}
        for key in ("condition_path", "video_latent_path", "audio_latent_path"):
            path = directory / f"{identity}.{key}.pt"
            path.write_bytes(b"metadata-only; no torch load in dry-run")
            payload[key] = str(path)
        row = {
            "task": task,
            "source_id": identity,
            "cache_fingerprint": f"fingerprint-{identity}",
            "target_height": height,
            "target_width": width,
            "target_num_frames": 124,
            "video_latent_shape": [24, 37, height // 16, width // 16],
            "audio_latent_shape": [2, 32, 207],
            "vae_normalized": True,
            **payload,
        }
        if task == "ref2av":
            images = image_count if image_count is not None else (1 if index == 0 else 0)
            row.update(
                reference_image_count=images, reference_video_count=0 if images else 1, reference_audio_count=1 if image_count is None and index == 2 else 0, packed_sequence_tokens_124=19000 + index
            )
        return row

    @staticmethod
    def condition_only(row):
        return {key: value for key, value in row.items() if key not in {"video_latent_path", "audio_latent_path"}}

    def manifest(self, relative, rows, *, receipt=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if receipt:
            record = {
                "schema_version": 1,
                "completed_count": len(rows),
                "failed_count": 0,
                "input_total_rows": len(rows),
                "preprocess_fingerprint": "test-native-cache",
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            path.with_suffix(".complete.json").write_text(json.dumps(record))
        return path

    def launch(self, task="fl2av", **environment):
        output = self.root / "output" / task
        return subprocess.run(
            ["bash", str(LAUNCHER), task, "--dry-run"],
            env={**self.environment, "H3_STAGED_OUTPUT": str(output), "H3_RDZV_ID": f"test_h3_{task}", **environment},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_both_tasks_keep_full_new_targets_decfg_and_two_independent_32gpu_groups(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                result = self.launch(task)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("dCM1500", result.stdout)
                self.assertIn("student_updates=250000, fake_updates=1250000", result.stdout)
                self.assertIn(f"NEW full AV rows={3 if task == 'fl2av' else 2}", result.stdout)
                self.assertIn("CM student EMA decay=0.999", result.stdout)
                if task == "ref2av":
                    self.assertIn("image-only references for both dCM and PDMD", result.stdout)
                    self.assertIn("new/ref2av_imageonly/metadata.jsonl", result.stdout)
                    self.assertIn("NEW=2 -> OLD=2", result.stdout)
                self.assertIn("shifts12/3", result.stdout)
                self.assertIn("DeCFG ON", result.stdout)
                self.assertIn("fake/teacher scores only", result.stdout)
                self.assertIn("student rollout and inference exclude DeCFG", result.stdout)
                self.assertRegex(result.stdout, r"--nnodes[= ]4")
                self.assertRegex(result.stdout, r"--nproc_per_node[= ]8")
                self.assertIn("127.0.0.1:29996", result.stdout)
                self.assertIn("127.0.0.1:29997", result.stdout)
                self.assertIn(f"test_h3_{task}_cm", result.stdout)
                self.assertIn(f"test_h3_{task}_pdmd", result.stdout)
                self.assertIn("checkpoint-000001500/student_ema", result.stdout)
                self.assertNotIn("/stale/", result.stdout)
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / "output").exists())

    def test_missing_clean_targets_fail_even_when_pdmd_condition_view_is_complete(self):
        Path(self.rows["t2av"][1]["video_latent_path"]).unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("video_latent_path", result.stderr)
        self.assertNotIn("torch.distributed.run", result.stdout)

    def test_both_tasks_restore_continuation_learning_rates_and_keep_save_limit_two(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                cm = yaml.safe_load((TRAIN_ROOT / f"configs/train/turbo/minimax_h3_{task}_dcm1500_decfg8_fsdp32.yaml").read_text())["training"]
                pdmd = yaml.safe_load((TRAIN_ROOT / f"configs/train/dmd/minimax_h3_{task}_pdmd1500000_from_cm_decfg8_fsdp32.yaml").read_text())["training"]
                for training in (cm, pdmd):
                    self.assertEqual(float(training["student"]["optimizer"]["learning_rate"]), 2e-5)
                    self.assertEqual(float(training["fake"]["optimizer"]["learning_rate"]), 4e-6)
                    self.assertEqual(training["save_total_limit"], 2)
                self.assertEqual(cm["student"]["optimizer"]["adam_beta2"], 0.999)
                self.assertEqual(pdmd["student"]["optimizer"]["adam_beta2"], 0.9)
                self.assertEqual(cm["student"]["ema"]["decay"], 0.999)

    def ref_preflight(self, cm_rows, new_rows):
        cm_manifest = self.manifest("strict_ref/metadata.jsonl", cm_rows, receipt=True)
        condition_manifest = self.manifest("strict_ref/condition_metadata.jsonl", [self.condition_only(row) for row in new_rows], receipt=True)
        output = self.root / "strict_output"
        environment = {
            **self.environment,
            "H3_CM_CACHE": str(cm_manifest),
            "H3_NEW_CACHE": str(condition_manifest),
            "H3_CM_OUTPUT": str(output / "cm/train"),
            "H3_PDMD_OUTPUT": str(output / "pdmd/train"),
            "H3_STUDENT_INIT": str(output / "cm/train/checkpoint-000001500/student_ema"),
            "H3_TRAINING_ADAPTER_PATH": str(self.root / "presets/model_ref2va_for_comfy_dit.safetensors"),
        }
        return subprocess.run(
            [
                sys.executable,
                str(TRAIN_ROOT / "scripts/minimax_h3_cm_pdmd_preflight.py"),
                "ref2av",
                str(TRAIN_ROOT / "configs/train/turbo/minimax_h3_ref2av_dcm1500_decfg8_fsdp32.yaml"),
                str(TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml"),
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_ref_imageonly_keeps_native_geometry_and_rejects_video_or_audio_in_both_new_streams(self):
        valid = self.rows["ref2av_imageonly"]
        self.assertIn((1184, 864), {(row["target_height"], row["target_width"]) for row in valid})
        result = self.ref_preflight(valid, valid)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("NEW=2 -> OLD=2", result.stdout)
        for phase in ("cm", "pdmd"):
            for videos, audios in ((1, 0), (0, 1)):
                with self.subTest(phase=phase, videos=videos, audios=audios):
                    bad = [{**row, "reference_video_count": videos, "reference_audio_count": audios} for row in valid]
                    result = self.ref_preflight(bad if phase == "cm" else valid, bad if phase == "pdmd" else valid)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("image-only references", result.stderr)

    def test_ref_imageonly_rejects_nonimage_reference_descriptors_even_with_imageonly_counts(self):
        valid = self.rows["ref2av_imageonly"]
        bad = [{**row, "references": [{"type": "video"}]} for row in valid]
        result = self.ref_preflight(bad, valid)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("image-only references", result.stderr)

    def test_rejects_the_previous_cm_ema_decay_for_both_tasks(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                cm_path = TRAIN_ROOT / f"configs/train/turbo/minimax_h3_{task}_dcm1500_decfg8_fsdp32.yaml"
                config = yaml.safe_load(cm_path.read_text())
                config["training"]["student"]["ema"]["decay"] = 0.99
                changed = self.root / f"old_ema_{task}.yaml"
                changed.write_text(yaml.safe_dump(config))
                result = self.launch(task, H3_STAGED_CM_CONFIG=str(changed))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("decay=0.999", result.stderr)

    def test_pdmd_only_dry_run_prints_only_the_second_stage_command(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                result = subprocess.run(
                    ["bash", str(LAUNCHER), task, "--pdmd-only", "--dry-run"],
                    env={**self.environment, "H3_STAGED_OUTPUT": str(self.root / "output" / task), "H3_RDZV_ID": f"test_h3_{task}"},
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("[h3-stages][cm]", result.stdout)
                self.assertIn("[h3-stages][pdmd]", result.stdout)
                self.assertIn('"pdmd_only": true', result.stdout)
                self.assertFalse((self.root / "output").exists())

    def test_old_student_side_preset_is_rejected_and_default_outputs_use_score_namespace(self):
        result = self.launch(H3_STAGED_OUTPUT="", H3_RDZV_ID="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pdmd1500000_scoredecfg8", result.stdout)
        paths = {}
        for phase, directory, filename in (
            ("CM", "turbo", "minimax_h3_fl2av_dcm1500_decfg8_fsdp32.yaml"),
            ("PDMD", "dmd", "minimax_h3_fl2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml"),
        ):
            value = yaml.safe_load((TRAIN_ROOT / "configs/train" / directory / filename).read_text())
            self.assertEqual(value["model"]["training_adapter"]["roles"], ["fake", "teacher"])
            value["model"]["training_adapter"]["roles"] = ["student", "fake"]
            path = self.root / f"old_student_{phase}.yaml"
            path.write_text(yaml.safe_dump(value))
            paths[f"H3_STAGED_{phase}_CONFIG"] = str(path)
        result = self.launch(**paths)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fake/teacher scores only", result.stderr)
        self.assertNotIn("torch.distributed.run", result.stdout)

    def test_cm_and_pdmd_new_views_must_cover_the_same_rows(self):
        self.manifest("new/t2av/condition_metadata.jsonl", [self.condition_only(self.rows["t2av"][0])], receipt=True)
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly the same complete rows", result.stderr)

    def test_missing_decfg_and_stale_student_initialization_fail_before_training(self):
        (self.root / "presets/model_ref2va_for_comfy_dit.safetensors").unlink()
        result = self.launch("ref2av")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing DeCFG preset", result.stderr)
        result = self.launch("ref2av", H3_TRAINING_ADAPTER_PATH=str(self.root / "presets/model_for_comfy_dit.safetensors"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("task-specific Comfy DeCFG preset", result.stderr)
        result = self.launch(H3_STUDENT_INIT="/stale/checkpoint/student_ema")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dCM-1500 EMA", result.stderr)

    def test_new_completion_receipts_are_required_and_bound_to_manifest_content(self):
        path = self.root / "new/t2av/metadata.jsonl"
        path.with_suffix(".complete.json").unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not fully merged", result.stderr)
        self.manifest("new/t2av/metadata.jsonl", self.rows["t2av"], receipt=True)
        path.write_text(path.read_text() + "\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from its completion receipt", result.stderr)

    def test_rejects_an_accidentally_changed_pdmd_budget_or_shift(self):
        original = TRAIN_ROOT / "configs/train/dmd/minimax_h3_fl2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml"
        for key, value, message in (("max_train_iters", 15000, "1500000 role updates"), ("video_shift", 6.0, "video12/audio3")):
            with self.subTest(key=key):
                config = yaml.safe_load(original.read_text())
                if key == "max_train_iters":
                    config["training"][key] = value
                else:
                    config["model"]["capabilities"]["distribution_matching"]["video_flow_shift"] = value
                path = self.root / "incorrect_pdmd.yaml"
                path.write_text(yaml.safe_dump(config))
                result = self.launch(H3_STAGED_PDMD_CONFIG=str(path))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_separate_pdmd_port_override_and_no_torch_import_in_dry_run(self):
        no_torch = self.root / "no_torch"
        no_torch.mkdir()
        (no_torch / "torch.py").write_text("raise RuntimeError('metadata preflight must not import torch')\n")
        result = self.launch(H3_PDMD_MASTER_PORT="30110", PYTHONPATH=str(no_torch))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("127.0.0.1:30110", result.stdout)
        result = self.launch(H3_PDMD_MASTER_PORT="29996")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("port", result.stderr.lower())

    def test_launcher_bash_syntax_and_wrong_task_rejection(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode, 0)
        result = subprocess.run(["bash", str(LAUNCHER), "t2av", "--dry-run"], env=self.environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
