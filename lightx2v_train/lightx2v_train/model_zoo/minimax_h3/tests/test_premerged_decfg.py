"""Offline DeCFG score bases cannot leak into the deployable student."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lightx2v_train.model_zoo.minimax_h3.premerged_decfg import (
    INDEX_NAME,
    MANIFEST_NAME,
    validate_premerged_score_base,
    validate_premerged_score_role,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PremergedDeCFGTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.component_dir = self.root / "transformer"
        self.component_dir.mkdir()
        (self.component_dir / "config.json").write_text('{"_class_name":"MiniMaxH3Transformer3DModel"}\n')
        self.shard = self.component_dir / "diffusion_pytorch_model-00001-of-00001.safetensors"
        self.shard.write_bytes(b"fixture-for-provenance-checks")
        index = self.component_dir / INDEX_NAME
        index.write_text(json.dumps({"weight_map": {"transformer_blocks.0.attn.to_q.weight": self.shard.name}}))
        self.preset = {
            "enabled": True,
            "sha256": "a" * 64,
            "path": "/original/machine/preset.safetensors",
            "scale": 1.0,
            "roles": ["fake", "teacher"],
            "qkv_layout": "comfy",
            "inference_requires_preset": False,
        }
        self.manifest = {
            "schema_version": 1,
            "format": "lightx2v_h3_score_decfg",
            "complete": True,
            "source_model_path": "/original/MiniMax-H3",
            "precision": "fp32_merged_targets",
            "student_uses_decfg": False,
            "components": {
                "transformer": {
                    "model_name": "minimax_h3_t2av",
                    "source_transformer_dir": "/original/MiniMax-H3/transformer",
                    "training_adapter": copy.deepcopy(self.preset),
                    "config_sha256": _sha(self.component_dir / "config.json"),
                    "index_sha256": _sha(index),
                    "adapted_weights": 1,
                    "shards": [{"path": self.shard.name, "sha256": _sha(self.shard), "size_bytes": self.shard.stat().st_size}],
                }
            },
        }
        self._publish()

    def _publish(self, manifest=None):
        text = json.dumps(self.manifest if manifest is None else manifest)
        (self.root / MANIFEST_NAME).write_text(text)
        (self.component_dir / MANIFEST_NAME).write_text(text)

    def _validate(self, **kwargs):
        return validate_premerged_score_base(self.root, component_name="transformer", **kwargs)

    def test_root_and_component_load_and_relocated_adapter_identity(self):
        expected = {**self.preset, "path": "/new/machine/preset.safetensors"}
        root = self._validate(expected_adapter=expected, verify_weights=True)
        component = validate_premerged_score_base(self.component_dir, component_name="transformer", expected_adapter=expected)
        self.assertEqual(root["component"], component["component"])
        self.assertEqual(root["transformer_dir"], str(self.component_dir))
        for role in ("fake", "teacher"):
            validate_premerged_score_role(root, role)
        with self.assertRaisesRegex(RuntimeError, "only for fake/teacher"):
            validate_premerged_score_role(root, "student")

    def test_original_base_is_allowed_but_required_merged_base_is_not(self):
        for path in (self.root / MANIFEST_NAME, self.component_dir / MANIFEST_NAME):
            path.unlink()
        self.assertIsNone(self._validate())
        with self.assertRaisesRegex(RuntimeError, "Missing premerged"):
            self._validate(required=True)
        validate_premerged_score_role(None, "student")

    def test_task_and_score_preset_must_match(self):
        with self.assertRaisesRegex(RuntimeError, "model/task mismatch"):
            self._validate(model_name="minimax_h3_ref2av")
        with self.assertRaisesRegex(RuntimeError, "no matching component"):
            validate_premerged_score_base(self.component_dir, component_name="transformer_ref")
        for change in ({"sha256": "b" * 64}, {"scale": 0.5}, {"qkv_layout": "official"}):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "differs from the CM"):
                self._validate(expected_adapter={**self.preset, **change})

    def test_incomplete_or_different_precision_is_rejected(self):
        for change in ({"complete": False}, {"precision": "bf16"}, {"student_uses_decfg": True}, {"schema_version": 2}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self._publish({**self.manifest, **change})
                self._validate()

    def test_root_and_component_manifests_must_agree(self):
        altered = copy.deepcopy(self.manifest)
        altered["components"]["transformer"]["training_adapter"]["scale"] = 0.5
        (self.component_dir / MANIFEST_NAME).write_text(json.dumps(altered))
        with self.assertRaisesRegex(RuntimeError, "manifests disagree"):
            self._validate()

    def test_small_file_hashes_and_shard_size_are_checked(self):
        for filename in ("config.json", INDEX_NAME):
            path = self.component_dir / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(filename=filename), self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                self._validate()
            path.write_bytes(original)
        self.shard.write_bytes(b"truncated")
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            self._validate()

    def test_full_shard_verification_detects_same_size_corruption(self):
        self.shard.write_bytes(b"x" * self.shard.stat().st_size)
        self._validate()
        with self.assertRaisesRegex(RuntimeError, "shard SHA256 mismatch"):
            self._validate(verify_weights=True)

    def test_shard_paths_and_index_membership_are_strict(self):
        component = self.manifest["components"]["transformer"]
        original = copy.deepcopy(component["shards"])
        for path in ("../outside.safetensors", "/absolute.safetensors"):
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "relative|escapes"):
                component["shards"][0]["path"] = path
                self._publish()
                self._validate()
        component["shards"] = original
        component["shards"].append(copy.deepcopy(original[0]))
        self._publish()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self._validate()

    def test_runtime_rejects_double_merge_and_student_role(self):
        import torch

        from lightx2v_train.model_zoo.minimax_h3.minimax_h3_t2av import MiniMaxH3T2AVModel

        def owner(adapter):
            result = object.__new__(MiniMaxH3T2AVModel)
            result.config = {"model": {"name": "minimax_h3_t2av", "pretrained_model_name_or_path": str(self.root), "training_adapter": adapter}}
            return result

        preset = self.root / "preset.safetensors"
        preset.write_bytes(b"not-read-when-already-merged")
        doubled = owner({"enabled": True, "path": str(preset)})
        with self.assertRaisesRegex(ValueError, "merge DeCFG twice"):
            doubled.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)
        score = owner({"enabled": False})
        score.load_components(load_transformer=False, load_vae=False, load_condition_encoder=False)
        score.transformer = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, "only for fake/teacher"):
            score.prepare_training_adapter_role("student")
        score.prepare_training_adapter_role("fake")
        self.assertFalse(score._training_adapter_applied)
        score.prepare_training_adapter_role("teacher")
        self.assertFalse(any(parameter.requires_grad for parameter in score.transformer.parameters()))


if __name__ == "__main__":
    unittest.main()
