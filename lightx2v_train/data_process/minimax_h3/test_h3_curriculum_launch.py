"""CPU-only config, ordered-cache preflight and 32-GPU launch contracts."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_new_then_old_fsdp32_32gpu_acp.sh"
DOWNLOAD = TRAIN_ROOT / "scripts/download_minimax_h3_training_adapter.sh"


class CurriculumLaunchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for component in ("transformer", "transformer_ref"):
            path = self.root / "model" / component
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
        self.condition = self.root / "condition.pt"
        self.condition.write_bytes(b"metadata-only; preflight must not deserialize")
        self.new_fl = self.write_manifest("selfgen/t2av/condition_metadata.jsonl", [self.row("t2av"), self.row("t2av", height=1184, width=864)])
        self.new_ref = self.write_manifest(
            "selfgen/ref2av/condition_metadata.jsonl",
            [self.row("ref2av", image_count=1), self.row("ref2av", image_count=0, video_count=1), self.row("ref2av", height=1024, width=992, image_count=0, video_count=1)],
        )
        for task in ("t2av", "i2av", "fl2av"):
            self.write_manifest(f"old_fl/{task}/metadata.jsonl", [self.row(task), self.row(task, height=1344, width=768)])
        self.old_ref = self.write_manifest("old_ref/metadata.jsonl", [self.row("ref2av", image_count=1), self.row("ref2av", image_count=2)])
        self.environment = {
            **os.environ,
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29996",
            "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
            "H3_PYTHON": sys.executable,
            "H3_MODEL_PATH": str(self.root / "model"),
            "H3_SELFGEN_CACHE": str(self.root / "selfgen"),
            "H3_FL2AV_CACHE": str(self.root / "old_fl"),
            "H3_REF2AV_CACHE": str(self.old_ref),
            "H3_TRAINING_ADAPTER_DIR": str(self.root / "adapter-not-downloaded"),
            "H3_CONFIG_PATH": "/stale/turbo.yaml",
            "H3_FL2AV_PDMD_CONFIG": "/stale/old_pdmd.yaml",
            "LOCAL_KERNELS": "",
            "H3_KERNEL_SNAPSHOT": "",
            "KERNELS_CACHE": "",
            "H3_TRAINING_ADAPTER_ENABLE": "false",
        }
        for name in ("H3_CURRICULUM_OUTPUT", "H3_CURRICULUM_CONFIG", "H3_NEW_CACHE", "H3_TRAINING_ADAPTER_PATH", "H3_STUDENT_INIT", "H3_RDZV_ID", "H3_TRAINING_ADAPTER_SCALE"):
            self.environment.pop(name, None)

    def row(self, task, *, height=768, width=1344, image_count=0, video_count=0):
        return {
            "task": task,
            "target_height": height,
            "target_width": width,
            "target_num_frames": 124,
            "condition_path": str(self.condition),
            "reference_image_count": image_count,
            "reference_video_count": video_count,
            "reference_audio_count": 0,
        }

    def write_manifest(self, relative, rows):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = {
            "completed_count": len(rows),
            "failed_count": 0,
            "input_total_rows": len(rows),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "preprocess_fingerprint": "test-only",
        }
        path.with_suffix(".complete.json").write_text(json.dumps(receipt))
        return path

    def launch(self, task="fl2av", method="pdmd", **environment):
        return subprocess.run(["bash", str(LAUNCHER), task, method, "--dry-run"], env={**self.environment, **environment}, capture_output=True, text=True, timeout=30)

    def test_all_four_configs_are_read_only_and_preserve_explicit_task_method(self):
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for task in ("fl2av", "ref2av"):
            for method in ("dmd", "pdmd"):
                with self.subTest(task=task, method=method):
                    result = self.launch(task, method)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
                    self.assertIn(f"minimax_h3_{task}_{method}8_new_then_old_adaln16_fsdp32.yaml", result.stdout)
                    self.assertIn("NEW -> OLD -> repeat", result.stdout)
                    self.assertIn("student_updates=2500, fake_updates=12500", result.stdout)
                    self.assertIn("TrainingAdapter enabled=False", result.stdout)
                    self.assertIn("_ta0_fake5_32gpu", result.stdout)
                    self.assertIn("shifts=6.0/3.0" if task == "fl2av" else "shifts=12.0/3.0", result.stdout)
                    self.assertNotIn("/stale/", result.stdout)
        after = {str(path.relative_to(self.root)): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_new_native_aspect_ratios_and_video_references_are_accepted(self):
        result = self.launch("ref2av", "dmd")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("new_rows=3, old_rows=2", result.stdout)
        result = self.launch("fl2av", "dmd")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("new_rows=2, old_rows=6", result.stdout)

    def test_optional_adapter_toggle_and_separate_output(self):
        result = self.launch(H3_TRAINING_ADAPTER_ENABLE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing TrainingAdapter", result.stderr)
        adapter = self.root / "adapter.safetensors"
        adapter.write_bytes(b"header checks occur in the strict model adapter loader")
        result = self.launch(H3_TRAINING_ADAPTER_ENABLE="1", H3_TRAINING_ADAPTER_PATH=str(adapter))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TrainingAdapter enabled=True", result.stdout)
        self.assertIn("_ta1_scoredecfg_fake5_32gpu", result.stdout)
        self.assertIn("fake/teacher only", result.stdout)
        self.assertIn("student rollout and inference exclude DeCFG", result.stdout)

    def test_legacy_student_side_preset_cannot_select_the_corrected_recipe(self):
        for method in ("dmd", "pdmd"):
            with self.subTest(method=method):
                path = TRAIN_ROOT / f"configs/train/dmd/minimax_h3_fl2av_{method}8_new_then_old_adaln16_fsdp32.yaml"
                value = yaml.safe_load(path.read_text())
                self.assertEqual(value["model"]["training_adapter"]["roles"], ["fake", "teacher"])
                value["model"]["training_adapter"]["roles"] = ["student", "fake"]
                custom = self.root / f"old_student_{method}.yaml"
                custom.write_text(yaml.safe_dump(value))
                result = self.launch(method=method, H3_CURRICULUM_CONFIG=str(custom))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fake/teacher scores only", result.stderr)
                self.assertNotIn("torch.distributed.run", result.stdout)

    def test_missing_or_modified_new_completion_receipt_fails_before_torchrun(self):
        receipt = self.new_fl.with_suffix(".complete.json")
        receipt.unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not fully merged", result.stderr)
        self.assertNotIn("torch.distributed.run", result.stdout)
        self.write_manifest("selfgen/t2av/condition_metadata.jsonl", [self.row("t2av")])
        self.new_fl.write_text(self.new_fl.read_text() + "\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from its completion receipt", result.stderr)

    def test_rejects_wrong_stage_task_and_invalid_native_geometry(self):
        self.write_manifest("selfgen/t2av/condition_metadata.jsonl", [self.row("i2av")])
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unexpected task", result.stderr)
        self.write_manifest("selfgen/t2av/condition_metadata.jsonl", [self.row("t2av", height=769)])
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("32-aligned", result.stderr)

    def test_optional_student_only_warm_start_is_checked(self):
        result = self.launch(H3_STUDENT_INIT=str(self.root / "missing_student"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing student-only warm start", result.stderr)

    def test_bash_and_download_dry_run(self):
        for script in (LAUNCHER, DOWNLOAD):
            self.assertEqual(subprocess.run(["bash", "-n", str(script)]).returncode, 0)
        output = self.root / "download-not-created"
        result = subprocess.run(["bash", str(DOWNLOAD), "--dry-run"], env={**self.environment, "H3_TRAINING_ADAPTER_DIR": str(output)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("download_minimax_h3_training_adapter.py", result.stdout)
        self.assertFalse(output.exists())

    def test_downloader_selects_correct_two_exports_and_emits_hash_receipt(self):
        path = TRAIN_ROOT / "scripts/download_minimax_h3_training_adapter.py"
        spec = importlib.util.spec_from_file_location("_adapter_download_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        destination = self.root / "download"
        sdk = ModuleType("modelscope.hub.snapshot_download")

        def snapshot(**kwargs):
            destination.mkdir()
            for name in module.FILES:
                (destination / name).write_bytes(b"test verified SDK payload" * 2)
            return str(destination)

        sdk.snapshot_download = Mock(side_effect=snapshot)
        with patch.dict(sys.modules, {"modelscope.hub.snapshot_download": sdk}):
            receipt = module.download(destination, "pinned-revision", 2)
        sdk.snapshot_download.assert_called_once_with(model_id=module.MODEL_ID, revision="pinned-revision", local_dir=str(destination), allow_file_pattern=[*module.FILES, "README.md"], max_workers=2)
        self.assertEqual(receipt["qkv_layout"], "comfy")
        self.assertEqual(len(receipt["files"]), 2)
        self.assertEqual(json.loads((destination / "download.complete.json").read_text()), receipt)


if __name__ == "__main__":
    unittest.main()
