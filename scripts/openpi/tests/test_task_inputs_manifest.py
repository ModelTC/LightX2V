#!/usr/bin/env python3
"""Focused tests for the LIBERO task-input manifest migration."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _namespace_package(name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


# Import only the evaluator protocol modules. Importing the public lightx2v
# package initializes the selected accelerator, which is unrelated to these
# CPU-only filesystem tests.
_namespace_package("lightx2v", PROJECT_ROOT / "lightx2v")
_namespace_package("lightx2v.models", PROJECT_ROOT / "lightx2v/models")
_namespace_package("lightx2v.models.runners", PROJECT_ROOT / "lightx2v/models/runners")
_namespace_package("lightx2v.models.runners.openpi", PROJECT_ROOT / "lightx2v/models/runners/openpi")

from lightx2v.models.runners.openpi.libero_protocol import (  # noqa: E402
    EvaluationConfig,
    TaskSpec,
    build_task_inputs_manifest,
    ensure_task_inputs_manifest,
    resolved_protocol,
)


class TaskInputsManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.bddl_path = self.root / "libero/libero/bddl_files/suite/task.bddl"
        self.init_states_path = self.root / "libero/libero/init_files/suite/task.pruned_init"
        self.bddl_path.parent.mkdir(parents=True)
        self.init_states_path.parent.mkdir(parents=True)
        self.bddl_path.write_bytes(b"bddl-v1")
        self.init_states_path.write_bytes(b"init-v1")
        self.spec = TaskSpec(
            benchmark="suite",
            task_id=0,
            suite=None,
            task=None,
            bddl_path=self.bddl_path.resolve(),
            init_states_path=self.init_states_path.resolve(),
        )
        self.task_specs = {"suite": [self.spec]}
        self.manifest_config = SimpleNamespace(benchmarks=("suite",), libero_root=self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _legacy_record(self, **updates: object) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": 1,
            "bddl_file": str(self.bddl_path),
            "init_states_file": str(self.init_states_path),
            "init_states_loader": "direct_file",
        }
        record.update(updates)
        return record

    def test_new_manifest_is_stable_and_content_sensitive(self) -> None:
        first = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        second = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 1)
        self.assertEqual(first["input_count"], 2)

        self.bddl_path.write_bytes(b"bddl-v2")
        changed = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        self.assertNotEqual(first["manifest_sha256"], changed["manifest_sha256"])

    def test_existing_manifest_is_verified_strictly(self) -> None:
        output_dir = self.root / "new-output"
        output_dir.mkdir()
        manifest = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        path = ensure_task_inputs_manifest(output_dir, manifest, {}, self.task_specs)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)
        self.assertEqual(ensure_task_inputs_manifest(output_dir, manifest, {}, self.task_specs), path)

        self.init_states_path.write_bytes(b"init-v2")
        changed = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        with self.assertRaisesRegex(RuntimeError, "content differs"):
            ensure_task_inputs_manifest(output_dir, changed, {}, self.task_specs)

    def test_schema_one_records_are_adopted_after_path_validation(self) -> None:
        output_dir = self.root / "legacy-output"
        output_dir.mkdir()
        manifest = build_task_inputs_manifest(self.task_specs, self.manifest_config)
        records = {("suite", 0, 0): self._legacy_record()}
        with self.assertLogs("lightx2v.models.runners.openpi.libero_protocol", level="WARNING"):
            path = ensure_task_inputs_manifest(output_dir, manifest, records, self.task_specs)
        self.assertTrue(path.is_file())

        invalid_output = self.root / "invalid-legacy-output"
        invalid_output.mkdir()
        invalid_records = {("suite", 0, 0): self._legacy_record(bddl_file=str(self.root / "other.bddl"))}
        with self.assertRaisesRegex(RuntimeError, "legacy episode"):
            ensure_task_inputs_manifest(invalid_output, manifest, invalid_records, self.task_specs)
        self.assertFalse((invalid_output / "task_inputs_manifest.json").exists())

    def test_protocol_id_retains_the_schema_one_field_set(self) -> None:
        model_path = self.root / "model"
        norm_path = model_path / "assets/physical-intelligence/libero/norm_stats.json"
        tokenizer_path = model_path / "assets/paligemma_tokenizer.model"
        norm_path.parent.mkdir(parents=True)
        (model_path / "model.safetensors").write_bytes(b"model")
        norm_path.write_bytes(b"norm")
        tokenizer_path.write_bytes(b"tokenizer")
        config_json = self.root / "model.json"
        config_json.write_bytes(b"{}")
        config = EvaluationConfig(
            benchmarks=("suite",),
            task_ids={"suite": (0,)},
            num_trials_per_task=1,
            env_seed=7,
            policy_seed=0,
            actions_per_plan=5,
            num_steps_wait=10,
            render_size=256,
            video_fps=10,
            video_policy="none",
            save_actions=False,
            fail_fast=False,
            resume=True,
            max_steps={"suite": 2},
            libero_root=self.root,
            libero_config_dir=self.root / "runtime",
        )
        resolved, protocol_id = resolved_protocol(config, model_path=model_path, config_json=config_json)
        protocol_fields = {
            key: value
            for key, value in resolved.items()
            if key
            not in {
                "protocol_id",
                "protocol_name",
                "official_protocol",
                "libero_config_dir",
                "video_fps",
                "video_policy",
                "save_actions",
                "fail_fast",
                "resume",
            }
        }
        expected = hashlib.sha256(json.dumps(protocol_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        self.assertEqual(protocol_id, expected)
        self.assertNotIn("task_inputs_manifest", resolved)


if __name__ == "__main__":
    unittest.main()
