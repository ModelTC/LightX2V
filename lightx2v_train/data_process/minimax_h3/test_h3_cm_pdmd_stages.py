"""Stage handoff identity, failure isolation and completed-stage resume."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lightx2v_train.data.minimax_h3_turbo_cm_dataset import MiniMaxH3TurboCMDataset
from lightx2v_train.runtime.config import load_config
from lightx2v_train.trainers.turbo.trainer import TurboTrainer

TRAIN_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("h3_cm_pdmd_stages_tested", TRAIN_ROOT / "scripts/run_minimax_h3_cm_pdmd_stages.py")
STAGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGES)
PREFLIGHT = STAGES.leaf(TRAIN_ROOT / "scripts/minimax_h3_cm_pdmd_preflight.py", "h3_cm_pdmd_test_preflight")


class StageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29911", "H3_RDZV_ID": "test_stages", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        self.environment_patch = patch.dict(os.environ, self.environment)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        os.environ.pop("H3_PDMD_MASTER_PORT", None)

    def configs(self, task="fl2av"):
        output = self.root / task
        metadata = self.root / f"{task}.jsonl"
        paths = {}
        for key in ("condition_path", "video_latent_path", "audio_latent_path"):
            path = self.root / f"{task}.{key}.pt"
            path.write_bytes(b"metadata-only target fixture")
            paths[key] = str(path)
        row = {
            "task": "t2av" if task == "fl2av" else "ref2av",
            "source_id": task,
            "cache_fingerprint": task,
            "target_height": 1184,
            "target_width": 864,
            "target_num_frames": 124,
            "vae_normalized": True,
            **paths,
        }
        metadata.write_text(json.dumps(row) + "\n")
        preset = self.root / ("model_for_comfy_dit.safetensors" if task == "fl2av" else "model_ref2va_for_comfy_dit.safetensors")
        preset.write_bytes(b"preset identity fixture")
        environment = {
            "H3_CM_CACHE": str(metadata),
            "H3_NEW_CACHE": str(metadata),
            "H3_FL2AV_CACHE": str(self.root / "old_fl"),
            "H3_REF2AV_CACHE": str(self.root / "old_ref/metadata.jsonl"),
            "H3_CM_OUTPUT": str(output / "cm/train"),
            "H3_PDMD_OUTPUT": str(output / "pdmd/train"),
            "H3_STUDENT_INIT": str(output / "cm/train/checkpoint-000001500/student_ema"),
            "H3_TRAINING_ADAPTER_PATH": str(preset),
        }
        cm_path = TRAIN_ROOT / f"configs/train/turbo/minimax_h3_{task}_dcm1500_decfg8_fsdp32.yaml"
        pdmd_path = TRAIN_ROOT / f"configs/train/dmd/minimax_h3_{task}_pdmd1500000_from_cm_decfg8_fsdp32.yaml"
        with patch.dict(os.environ, environment):
            cm, pdmd = load_config(str(cm_path)), load_config(str(pdmd_path))
        summary = PREFLIGHT.validate(task, cm, pdmd, check_files=False)
        return cm_path, pdmd_path, cm, pdmd, summary

    def make_checkpoint(self, config, summary):
        checkpoint = Path(summary["cm_output"]) / "checkpoint-000001500"
        checkpoint.mkdir(parents=True)
        common = {"version": 1, "iteration": 1500, "world_size": 32}
        (checkpoint / "turbo.complete.json").write_text(json.dumps(common))
        recipe = STAGES.expected_cm_recipe(config)
        for rank in range(32):
            value = {**common, "rank": rank, "recipe": recipe, "fake_updates": 0, "consumed_microbatches": 1500}
            (checkpoint / f"turbo.rank{rank:05d}.json").write_text(json.dumps(value))
        export = checkpoint / "student_ema"
        export.mkdir()
        (export / "pytorch_lora_weights.safetensors").write_bytes(b"exported learned adapter identity fixture")
        (export / "adapter_config.json").write_text(json.dumps({"r": 128, "lora_alpha": 128}))
        preset = STAGES.preset_metadata(config)
        (export / "training_adapter.json").write_text(
            json.dumps(
                {
                    "enabled": False,
                    "schema_version": 1,
                    "inference_requires_preset": False,
                    "export_semantics": "learned_lora_only",
                    "export_role": "student",
                    "model_name": config["model"]["name"],
                    "score_training_adapter": preset,
                }
            )
        )
        return checkpoint, export, preset

    def test_expected_recipe_matches_real_turbo_dataset_fingerprint_for_both_tasks(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                _, _, config, _, _ = self.configs(task)
                dataset = MiniMaxH3TurboCMDataset(config["data"]["train"]["data_path"], tasks=config["data"]["train"]["tasks"], native_geometry=True)
                trainer = TurboTrainer.__new__(TurboTrainer)
                trainer.public_config = config
                trainer.dataloader_train = SimpleNamespace(dataset=dataset)
                self.assertEqual(STAGES.expected_cm_recipe(config), trainer._recipe())

    def test_real_receipt_and_preset_validation_reject_tampered_task_rank_and_targets(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                _, _, cm, _, summary = self.configs(task)
                checkpoint, export, preset = self.make_checkpoint(cm, summary)
                value = STAGES.verify_handoff(cm, summary, preset)
                self.assertEqual(value["student_init"], str(export))
                self.assertFalse(value["student_uses_decfg"])
                self.assertFalse(value["inference_requires_preset"])
                self.assertEqual(value["student_adapter_sha256"], hashlib.sha256((export / "pytorch_lora_weights.safetensors").read_bytes()).hexdigest())
                rank = checkpoint / "turbo.rank00007.json"
                original = rank.read_text()
                state = json.loads(original)
                state["consumed_microbatches"] = 1499
                rank.write_text(json.dumps(state))
                with self.assertRaisesRegex(RuntimeError, "cursor mismatch"):
                    STAGES.verify_handoff(cm, summary, preset)
                rank.write_text(original)
                sidecar = export / "training_adapter.json"
                original_sidecar = sidecar.read_text()
                state = json.loads(original_sidecar)
                state["model_name"] = "minimax_h3_ref2av" if task == "fl2av" else "minimax_h3_t2av"
                sidecar.write_text(json.dumps(state))
                with self.assertRaisesRegex(RuntimeError, "model/task mismatch"):
                    STAGES.verify_handoff(cm, summary, preset)
                sidecar.write_text(original_sidecar)
                old = {
                    **preset,
                    "roles": ["student", "fake"],
                    "export_semantics": "learned_lora_only_requires_preset",
                    "inference_requires_preset": True,
                    "export_role": "student",
                    "model_name": cm["model"]["name"],
                }
                sidecar.write_text(json.dumps(old))
                with self.assertRaisesRegex(RuntimeError, "preset differs"):
                    STAGES.verify_handoff(cm, summary, preset)
                sidecar.write_text(original_sidecar)
                manifest = Path(cm["data"]["train"]["data_path"])
                row = json.loads(manifest.read_text())
                row["cache_fingerprint"] = "changed-target-cache"
                manifest.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(RuntimeError, "recipe, rank or cursor mismatch"):
                    STAGES.verify_handoff(cm, summary, preset)

    def test_shared_plan_is_atomic_identical_and_never_overwrites_a_different_recipe(self):
        path = self.root / "plan.json"
        value = {"task": "ref2av", "preset": "first"}
        STAGES.publish_once(path, value)
        STAGES.publish_once(path, copy.deepcopy(value))
        with self.assertRaisesRegex(RuntimeError, "recipe differs"):
            STAGES.publish_once(path, {**value, "preset": "other"})
        self.assertEqual(json.loads(path.read_text()), value)
        self.assertEqual(list(self.root.glob(".plan.json.*")), [])

    def test_ref_actual_launch_automatically_prepares_the_native_imageonly_view(self):
        fixtures = STAGES.leaf(TRAIN_ROOT / "data_process/minimax_h3/test_h3_dcm_pdmd_launch.py", "_h3_imageonly_launch_fixtures")
        fixture = fixtures.H3DcmPdmdLaunchTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        view = fixture.root / "new/ref2av_imageonly"
        shutil.rmtree(view)
        source = fixture.root / "new/ref2av/metadata.jsonl"
        original = source.read_bytes()
        output = fixture.root / "restart_ref"
        environment = {
            **fixture.environment,
            "H3_CM_CACHE": str(view / "metadata.jsonl"),
            "H3_NEW_CACHE": str(view / "condition_metadata.jsonl"),
            "H3_CM_OUTPUT": str(output / "cm/train"),
            "H3_PDMD_OUTPUT": str(output / "pdmd/train"),
            "H3_STUDENT_INIT": str(output / "cm/train/checkpoint-000001500/student_ema"),
            "H3_TRAINING_ADAPTER_PATH": str(fixture.root / "presets/model_ref2va_for_comfy_dit.safetensors"),
            "H3_RDZV_ID": "restart_ref",
        }
        cm_path = TRAIN_ROOT / "configs/train/turbo/minimax_h3_ref2av_dcm1500_decfg8_fsdp32.yaml"
        pdmd_path = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_pdmd1500000_from_cm_decfg8_fsdp32.yaml"
        with patch.dict(os.environ, environment), patch.object(STAGES.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "stop-before-GPU-training")) as run:
            with self.assertRaisesRegex(RuntimeError, "image-only view is not prepared"):
                STAGES.execute("ref2av", cm_path, pdmd_path, dry_run=True)
            self.assertFalse(view.exists())
            run.assert_not_called()
            with self.assertRaises(subprocess.CalledProcessError):
                STAGES.execute("ref2av", cm_path, pdmd_path)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][-1], str(cm_path))
        rows = [json.loads(line) for line in (view / "metadata.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["reference_image_count"] > 0 and row["reference_video_count"] == row["reference_audio_count"] == 0 for row in rows))
        self.assertIn((1184, 864), {(row["target_height"], row["target_width"]) for row in rows})
        self.assertTrue(all(Path(row[key]).is_file() for row in rows for key in ("condition_path", "video_latent_path", "audio_latent_path")))
        self.assertEqual(source.read_bytes(), original)

    def test_pdmd_only_uses_verified_existing_ema_and_preserves_the_original_plan(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                cm_path, pdmd_path, cm, pdmd, summary = self.configs(task)
                self.make_checkpoint(cm, summary)
                directory = Path(summary["cm_output"]).parent.parent / "stages"
                directory.mkdir(parents=True)
                old_pdmd = copy.deepcopy(pdmd)
                old_pdmd["training"]["student"]["optimizer"]["learning_rate"] = 5e-5
                old_pdmd["training"]["fake"]["optimizer"]["learning_rate"] = 1e-5
                old_plan = directory / "plan.json"
                old_plan.write_text(json.dumps({"pdmd": old_pdmd, "started_before_lr_restore": True}))
                original = old_plan.read_bytes()
                helper = SimpleNamespace(load_config=lambda path: cm if Path(path) == cm_path else pdmd, validate=Mock(return_value=summary))
                real_leaf = STAGES.leaf
                with (
                    patch.object(STAGES, "leaf", side_effect=lambda path, name: helper if Path(path).name == "minimax_h3_cm_pdmd_preflight.py" else real_leaf(path, name)),
                    patch.object(STAGES.subprocess, "run") as run,
                ):
                    STAGES.execute(task, cm_path, pdmd_path, pdmd_only=True)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][-1], str(pdmd_path))
                self.assertIn("--rdzv_id=test_stages_pdmd", run.call_args.args[0])
                self.assertEqual(old_plan.read_bytes(), original)
                new_plan = json.loads((directory / "pdmd_only/plan.json").read_text())
                self.assertEqual(new_plan["pdmd"]["training"]["student"]["optimizer"]["learning_rate"], 2e-5)
                self.assertEqual(new_plan["pdmd"]["training"]["fake"]["optimizer"]["learning_rate"], 4e-6)
                receipt = json.loads((directory / "pdmd_only/cm_handoff.complete.json").read_text())
                self.assertEqual(receipt["student_init"], summary["student_init"])
                self.assertEqual(receipt["plan_sha256"], new_plan["plan_sha256"])
                self.assertFalse(receipt["inference_requires_preset"])

    def test_pdmd_only_refuses_an_incomplete_cm_checkpoint_without_launching_training(self):
        cm_path, pdmd_path, cm, pdmd, summary = self.configs()
        checkpoint, _, _ = self.make_checkpoint(cm, summary)
        (checkpoint / "turbo.rank00007.json").unlink()
        helper = SimpleNamespace(load_config=lambda path: cm if Path(path) == cm_path else pdmd, validate=Mock(return_value=summary))
        real_leaf = STAGES.leaf
        with (
            patch.object(STAGES, "leaf", side_effect=lambda path, name: helper if Path(path).name == "minimax_h3_cm_pdmd_preflight.py" else real_leaf(path, name)),
            patch.object(STAGES.subprocess, "run") as run,
        ):
            with self.assertRaises(FileNotFoundError):
                STAGES.execute("fl2av", cm_path, pdmd_path, pdmd_only=True)
            run.assert_not_called()

    def test_stage_failure_never_starts_pdmd_and_completed_stage_is_skipped_on_resume(self):
        cm_path, pdmd_path, cm, pdmd, summary = self.configs()
        helper = SimpleNamespace(load_config=lambda path: cm if Path(path) == cm_path else pdmd, validate=Mock(return_value=summary))
        handoff = {"student_init": summary["student_init"], "student_adapter_sha256": "unchanged"}
        with (
            patch.object(STAGES, "leaf", return_value=helper),
            patch.object(STAGES, "preset_metadata", return_value={"enabled": True}),
            patch.object(STAGES, "verify_handoff", return_value=handoff) as verify,
            patch.object(STAGES.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cm")) as run,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                STAGES.execute("fl2av", cm_path, pdmd_path)
            self.assertEqual(run.call_count, 1)
            verify.assert_not_called()
        with (
            patch.object(STAGES, "leaf", return_value=helper),
            patch.object(STAGES, "preset_metadata", return_value={"enabled": True}),
            patch.object(STAGES, "verify_handoff", side_effect=RuntimeError("invalid export")),
            patch.object(STAGES.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid export"):
                STAGES.execute("fl2av", cm_path, pdmd_path)
            self.assertEqual(run.call_count, 1)
        with (
            patch.object(STAGES, "leaf", return_value=helper),
            patch.object(STAGES, "preset_metadata", return_value={"enabled": True}),
            patch.object(STAGES, "verify_handoff", return_value=handoff),
            patch.object(STAGES.subprocess, "run") as run,
        ):
            STAGES.execute("fl2av", cm_path, pdmd_path)
            self.assertEqual(run.call_count, 2)
            self.assertIn("--rdzv_endpoint=127.0.0.1:29911", run.call_args_list[0].args[0])
            self.assertIn("--rdzv_endpoint=127.0.0.1:29912", run.call_args_list[1].args[0])
            run.reset_mock()
            STAGES.execute("fl2av", cm_path, pdmd_path)
            self.assertEqual(run.call_count, 1)
            self.assertIn("--rdzv_id=test_stages_pdmd", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
