"""CPU-only selection, request, resume, and launch contract tests."""

import argparse
import ast
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import completed, request_for
from common import atomic_json, atomic_jsonl, file_digest, records
from launch import smoke_command
from prepare_selection import ordered_images, prepare


class TeacherWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.cache, self.output = self.root / "raw.jsonl", self.root / "cache.jsonl", self.root / "output"
        self.image = self.root / "reference.jpg"
        self.image.write_bytes(b"fixture-image")
        self.raw_rows, self.cache_rows = [], []
        for count in range(2, 7):
            for item in range(6):
                index = len(self.raw_rows)
                source_id = f"source-{count}-{item}"
                self.raw_rows.append(
                    {
                        "sample_id": source_id,
                        "task": "ref2v",
                        "enhanced_prompt": f"  prompt {source_id}  ",
                        "target_height": 3840,
                        "target_width": 2160,
                        "ref_image_count": 99,
                        "references": [{"modality": "image", "rel_path": self.image.name, "order": i + 1} for i in range(count)],
                    }
                )
                self.cache_rows.append(
                    {
                        "source_index": index,
                        "source_id": source_id,
                        "task": "ref2av",
                        "target_height": 768,
                        "target_width": 1344,
                        "num_frames": 124,
                        "condition_path": f"conditions/{index}.pt",
                        "reference_image_count": count,
                        "ref_image_count": count,
                        "prompt_source": "enhanced_prompt",
                    }
                )
        self.write_fixture()

    def write_fixture(self):
        atomic_jsonl(self.raw, self.raw_rows)
        atomic_jsonl(self.cache, self.cache_rows)

    def select(self, node=0):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare(self.raw, self.cache, self.root, self.output, node, quota=4)

    def test_selection_balanced_stable_cached_geometry_and_no_stale_count(self):
        first, second = self.select(0), self.select(1)
        self.assertEqual(first, second)
        self.assertEqual(first["total_selected"], 20)
        for node in (0, 1):
            rows = [row for _, row in records(self.output / "selection" / f"node{node}" / f"node{node}.jsonl")]
            self.assertEqual(len(rows), 10)
            self.assertEqual({count: sum(row["ref_image_count"] == count for row in rows) for count in range(2, 7)}, dict.fromkeys(range(2, 7), 2))
            self.assertTrue(all((row["target_height"], row["target_width"]) == (768, 1344) for row in rows))
        self.assertEqual(self.select(0)["selection_sha256"], first["selection_sha256"])

    def test_missing_enhanced_prompt_and_mixed_modalities_cannot_fill_quota(self):
        for row in self.raw_rows[:3]:
            row["enhanced_prompt"] = " "
            row["prompt_en"] = "must not fall back"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            self.select()

    def test_exact_source_index_and_identity_join(self):
        for row in self.cache_rows[:3]:
            row["source_id"] = "wrong-id"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
            self.select()

    def test_missing_selected_image_is_fatal(self):
        self.image.unlink()
        with self.assertRaises(FileNotFoundError):
            self.select()

    def test_wrong_cache_reference_count_is_fatal(self):
        self.cache_rows[0]["reference_image_count"] = 99
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "actual ordered references"):
            self.select()

    def test_node_selection_mismatch_is_fatal(self):
        self.select(0)
        self.raw_rows[0]["enhanced_prompt"] = "changed prompt"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "Nodes disagree"):
            self.select(1)

    def test_reference_order_aliases_fallback_and_blank_lines(self):
        row = {"actual_ordered_references": [{"kind": "reference_image", "path": "b.jpg", "order": 0}, {"kind": "image_url", "path": "a.jpg", "order": 1}]}
        self.assertEqual(ordered_images(row, self.root), [str(self.root / "b.jpg"), str(self.root / "a.jpg")])
        row["actual_ordered_references"].reverse()
        with self.assertRaisesRegex(ValueError, "refusing to reorder"):
            ordered_images(row, self.root)
        self.assertEqual(ordered_images({"reference_images": ["b.jpg", "a.jpg"]}, self.root), [str(self.root / "b.jpg"), str(self.root / "a.jpg")])
        with self.assertRaisesRegex(ValueError, "non-image"):
            ordered_images({"references": [{"modality": "video", "path": "a.mp4"}]}, self.root)
        self.raw.write_text('\n{"id": 1}\n\n{"id": 2}\n')
        self.assertEqual([index for index, _ in records(self.raw)], [0, 1])

    def example(self):
        self.select()
        return next(records(self.output / "selection/node0/node0.jsonl"))[1]

    def test_http_request_matches_actual_server_schema(self):
        # AST inspection uses the actual schema without importing GPU/platform dependencies.
        repo = Path(__file__).resolve().parents[3]
        tree = ast.parse((repo / "lightx2v/server/schema.py").read_text())
        fields = {
            statement.target.id
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name in {"BaseTaskRequest", "VideoTaskRequest"}
            for statement in cls.body
            if isinstance(statement, ast.AnnAssign)
        }
        payload = request_for(self.example(), self.root / "trajectory")
        self.assertFalse(set(payload) - fields)
        self.assertTrue(payload["teacher_latents_only"])
        self.assertNotIn("fps", payload)

    def write_export(self, row):
        directory = self.root / "export"
        directory.mkdir()
        manifest = {
            "schema": "h3_teacher_trajectory_v1",
            "task": "ref2av",
            "dtype": "float32",
            "normalized": True,
            "num_steps": 29,
            "num_states": 30,
            "num_frames": 124,
            "height": row["target_height"],
            "width": row["target_width"],
            "fps": 24,
            "seed": row["seed"],
            "prompt": row["prompt"],
            "source_id": row["source_id"],
            "image_paths": row["reference_images"],
            "scheduler_type": "reference_blend",
            "model_variant": "ref2av",
            "cache_fingerprint": "runtime-fingerprint",
            "state_paths": [f"state_{index:03d}.pt" for index in range(30)],
            "condition_path": "condition.pt",
            "video_latents_path": "video_latents.pt",
            "audio_latents_path": "audio_latents.pt",
        }
        for name in manifest["state_paths"] + ["condition.pt", "video_latents.pt", "audio_latents.pt"]:
            (directory / name).write_bytes(b"tensor-fixture")
        atomic_json(directory / "manifest.json", manifest)
        return directory, manifest

    def test_resume_requires_success_manifest_matching_fingerprint_and_every_state(self):
        row = self.example()
        directory, manifest = self.write_export(row)
        self.assertIsNone(completed(directory, row, "request-A"))
        atomic_json(
            directory / "completion.json", {"request_fingerprint": "request-A", "cache_fingerprint": manifest["cache_fingerprint"], "manifest_sha256": file_digest(directory / "manifest.json")}
        )
        self.assertEqual(completed(directory, row, "request-A"), manifest)
        self.assertIsNone(completed(directory, row, "request-B"))
        (directory / "state_029.pt").unlink()
        self.assertIsNone(completed(directory, row, "request-A"))

    def test_smoke_is_offline_eight_gpus_with_decode_and_source_identity(self):
        row = self.example()
        args = types.SimpleNamespace(model_path="merged", config_json="teacher.json", no_export=False)
        command = smoke_command(row, args, self.root / "trajectory", self.root / "sample.mp4")
        self.assertIn("--standalone", command)
        self.assertIn("--nproc_per_node=8", command)
        self.assertIn("lightx2v.infer", command)
        self.assertNotIn("--teacher_latents_only", command)
        self.assertNotIn("--no-teacher_latents_only", command)
        self.assertEqual(command[command.index("--teacher_sample_id") + 1], row["source_id"])
        self.assertNotIn("lightx2v.server", command)
        tree = ast.parse((Path(__file__).resolve().parents[3] / "lightx2v/infer.py").read_text())
        parse_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_parser")
        namespace = {"argparse": argparse, "json": json, "RUNNER_MODULES": {"minimax_h3": None}}
        exec(compile(ast.Module(body=[parse_function], type_ignores=[]), "infer.py", "exec"), namespace)
        from unittest.mock import patch

        inference_args = command[command.index("lightx2v.infer") + 1 :]
        with patch.object(sys, "argv", ["lightx2v.infer", *inference_args]):
            parsed = namespace["build_parser"]().parse_args()
        self.assertFalse(parsed.teacher_latents_only)
        self.assertEqual(parsed.teacher_sample_id, row["source_id"])


if __name__ == "__main__":
    unittest.main()
