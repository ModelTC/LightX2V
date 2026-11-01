"""Completed-CM handoff, score-only premerge and 32-GPU PDMD launch contracts."""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TRAIN_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("h3_premerged_launch_fixtures", Path(__file__).with_name("test_h3_dcm_pdmd_launch.py"))
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)
with patch.object(sys, "path", [str(TRAIN_ROOT / "scripts"), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("h3_premerged_launch_tested", TRAIN_ROOT / "scripts/run_minimax_h3_pdmd_from_cm_premerged.py")
    LAUNCH = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(LAUNCH)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PremergedLaunchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = FIXTURES.H3DcmPdmdLaunchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.environment = {**self.fixture.environment, "H3_SCORE_MODEL_PATH": str(self.root / "scores")}
        self.loader = LAUNCH.leaf(TRAIN_ROOT / "lightx2v_train/runtime/config.py", "_h3_merged_test_config")

    def config(self, task):
        export = self.root / task / "cm/train/checkpoint-000001500/student_ema"
        environment = {**self.environment, "H3_STUDENT_INIT": str(export), "H3_PDMD_OUTPUT": str(self.root / task / "pdmd_new/train"), "H3_RDZV_ID": f"premerged_{task}"}
        environment["H3_NEW_CACHE"] = str(self.root / f"new/{'t2av' if task == 'fl2av' else 'ref2av_imageonly'}/condition_metadata.jsonl")
        path = TRAIN_ROOT / f"configs/train/dmd/minimax_h3_{task}_pdmd1500000_from_cm_premerged_decfg8_save100_fsdp32.yaml"
        with patch.dict(os.environ, environment):
            config = self.loader.load_config(str(path))
        return config, export, environment, path

    def prepare(self, task):
        config, export, environment, path = self.config(task)
        component = "transformer" if task == "fl2av" else "transformer_ref"
        directory = self.root / "scores" / component
        directory.mkdir(parents=True)
        (directory / "config.json").write_text("{}")
        shard = directory / "diffusion_pytorch_model-00001-of-00001.safetensors"
        shard.write_bytes(b"CPU preflight only reads sizes and hashes, not tensor data")
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
        adapter = {"enabled": True, "schema_version": 1, "sha256": "a" * 64, "scale": 1.0, "roles": ["fake", "teacher"], "qkv_layout": "comfy", "inference_requires_preset": False}
        manifest = {
            "schema_version": 1,
            "format": "lightx2v_h3_score_decfg",
            "complete": True,
            "source_model_path": config["model"]["pretrained_model_name_or_path"],
            "precision": "fp32_merged_targets",
            "student_uses_decfg": False,
            "components": {
                component: {
                    "model_name": config["model"]["name"],
                    "training_adapter": adapter,
                    "config_sha256": digest(directory / "config.json"),
                    "index_sha256": digest(index),
                    "shards": [{"path": shard.name, "sha256": digest(shard), "size_bytes": shard.stat().st_size}],
                    "adapted_weights": 1,
                }
            },
        }
        (directory.parent / "decfg_merged.json").write_text(json.dumps(manifest))
        export.mkdir(parents=True)
        (export / "pytorch_lora_weights.safetensors").write_bytes(b"learned CM EMA adapter fixture")
        (export / "adapter_config.json").write_text(json.dumps({"r": 128, "lora_alpha": 128}))
        (export / "training_adapter.json").write_text(
            json.dumps(
                {
                    "enabled": False,
                    "schema_version": 1,
                    "inference_requires_preset": False,
                    "export_semantics": "learned_lora_only",
                    "export_role": "student",
                    "model_name": config["model"]["name"],
                    "score_training_adapter": adapter,
                }
            )
        )
        marker = {"version": 1, "iteration": 1500, "world_size": 32}
        (export.parent / "turbo.complete.json").write_text(json.dumps(marker))
        recipe = {
            "model": {"name": config["model"]["name"], "pretrained_model_name_or_path": config["model"]["pretrained_model_name_or_path"]},
            "training": {"student": {"lora": config["training"]["student"]["lora"]}},
        }
        for rank in range(32):
            (export.parent / f"turbo.rank{rank:05d}.json").write_text(json.dumps({**marker, "rank": rank, "fake_updates": 0, "consumed_microbatches": 1500, "recipe": recipe}))
        return config, export, environment, path

    def test_both_shell_dry_runs_are_read_only_and_select_only_pdmd(self):
        for task in ("fl2av", "ref2av"):
            with self.subTest(task=task):
                config, _, environment, _ = self.prepare(task)
                before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
                result = subprocess.run(
                    ["bash", str(TRAIN_ROOT / "scripts/run_minimax_h3_pdmd_from_cm_premerged_fsdp32_32gpu_acp.sh"), task, "--dry-run"], env=environment, capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"save_every_role_updates": 100', result.stdout)
                self.assertIn('"online_decfg_merge": false', result.stdout)
                self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
                self.assertIn("premerged_decfg8_save100_fsdp32.yaml", result.stdout)
                self.assertNotIn("[h3-stages][cm]", result.stdout)
                after = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
                self.assertEqual(before, after)
                self.assertFalse(Path(config["training"]["output_dir"]).exists())

    def test_actual_entry_launches_once_then_enforces_immutable_handoff(self):
        config, export, environment, path = self.prepare("fl2av")
        with patch.dict(os.environ, environment), patch.object(LAUNCH.subprocess, "run") as run:
            LAUNCH.execute("fl2av", path)
            self.assertEqual(run.call_count, 1)
            self.assertIn("--rdzv_id=premerged_fl2av", run.call_args.args[0])
            receipt = Path(config["training"]["output_dir"]) / "cm_premerged_handoff.json"
            original = receipt.read_bytes()
            LAUNCH.execute("fl2av", path)
            self.assertEqual(receipt.read_bytes(), original)
            (export / "pytorch_lora_weights.safetensors").write_bytes(b"changed weights")
            with self.assertRaisesRegex(RuntimeError, "recipe differs"):
                LAUNCH.execute("fl2av", path)
            self.assertEqual(run.call_count, 2)

    def test_rejects_incomplete_cm_wrong_task_student_decfg_and_double_merge(self):
        config, export, _, _ = self.prepare("ref2av")
        rank = export.parent / "turbo.rank00017.json"
        original = rank.read_text()
        state = json.loads(original)
        state["consumed_microbatches"] = 1499
        rank.write_text(json.dumps(state))
        with self.assertRaisesRegex(RuntimeError, "rank/cursor"):
            LAUNCH.validate(config, "ref2av")
        rank.write_text(original)
        with self.assertRaisesRegex(RuntimeError, "task differs"):
            LAUNCH.validate(config, "fl2av")
        sidecar = export / "training_adapter.json"
        original = sidecar.read_text()
        source = json.loads(original)
        source["enabled"] = True
        sidecar.write_text(json.dumps(source))
        with self.assertRaisesRegex(RuntimeError, "without DeCFG"):
            LAUNCH.validate(config, "ref2av")
        sidecar.write_text(original)
        bad = copy.deepcopy(config)
        bad["model"]["teacher"]["training_adapter"]["enabled"] = True
        with self.assertRaisesRegex(RuntimeError, "applied twice"):
            LAUNCH.validate(bad, "ref2av")
        score = self.root / "scores/decfg_merged.json"
        manifest = json.loads(score.read_text())
        manifest["components"]["transformer_ref"]["training_adapter"]["sha256"] = "b" * 64
        score.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "differs from the CM"):
            LAUNCH.validate(config, "ref2av")

    def test_ref_still_rejects_video_references_without_needing_cm_latents(self):
        config, _, _, _ = self.prepare("ref2av")
        # PDMD needs conditioning, not the old clean AV files or a CM YAML.
        for row in self.fixture.rows["ref2av_imageonly"]:
            Path(row["video_latent_path"]).unlink()
            Path(row["audio_latent_path"]).unlink()
        LAUNCH.validate(config, "ref2av")
        manifest = Path(config["data"]["train"]["sources"]["new"]["data_path"])
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        rows[0]["reference_video_count"] = 1
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "image-only"):
            LAUNCH.validate(config, "ref2av")


if __name__ == "__main__":
    unittest.main()
