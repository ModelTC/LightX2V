"""Metadata preparation tests intentionally require only the standard library."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "minimax_h3_dmad_manifest.py"
SPEC = importlib.util.spec_from_file_location("dmad_manifest_stdlib", MODULE_PATH)
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


class DMADManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = {
            "condition_path": "condition.pt",
            "cache_fingerprint": "fp-1",
            "source_id": "video-1",
            "target_height": 32,
            "target_width": 64,
            "target_num_frames": 107,
            "target_orientation": "landscape",
            "reference_image_count": 3,
            "packed_sequence_tokens_124": 100,
        }
        for path in ("condition.pt", "real.pt", "teacher.pt", "negative_condition.pt"):
            (self.root / path).touch()
        self.condition = self.write("condition.jsonl", [self.row])
        self.real_row = {**self.row, "normalized": True, "real_latent_path": "real.pt"}
        self.teacher_row = {**self.row, "normalized": True, "teacher_latent_path": "teacher.pt"}
        self.real = self.write("real.jsonl", [self.real_row])
        self.teacher = self.write("teacher.jsonl", [self.teacher_row])

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def join(self):
        return manifest.build_joined_rows([self.condition], [self.real], [self.teacher])

    def test_join_preserves_sampler_metadata_and_absolutizes_paths(self):
        row = self.join()[0]
        self.assertEqual(row["reference_image_count"], 3)
        self.assertEqual(row["packed_sequence_tokens_124"], 100)
        self.assertEqual(row["real_latent_path"], str(self.root / "real.pt"))
        self.assertEqual(row["negative_condition_path"], str(self.root / "negative_condition.pt"))
        self.assertEqual(row["dmad_schema_version"], 1)
        output = self.root / "elsewhere" / "paired.jsonl"
        manifest.write_manifest_atomic([row], output)
        self.assertEqual(manifest.validate_manifest(output), [row])

    def test_rejects_fingerprint_and_source_id_and_geometry_mismatch(self):
        for key, value in (("cache_fingerprint", "wrong"), ("source_id", "other"), ("target_width", 96)):
            with self.subTest(key=key):
                self.write("teacher.jsonl", [{**self.teacher_row, key: value}])
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.join()

    def test_rejects_missing_source_id_and_normalization(self):
        row = dict(self.teacher_row)
        del row["source_id"]
        self.write("teacher.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "source_id"):
            self.join()
        self.write("teacher.jsonl", [{**self.teacher_row, "normalized": False}])
        with self.assertRaisesRegex(ValueError, "normalized=true"):
            self.join()

    def test_rejects_source_only_pairing(self):
        row = dict(self.real_row)
        del row["cache_fingerprint"]
        self.write("real.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "cache_fingerprint"):
            self.join()

    def test_rejects_missing_target_and_condition_only_cache(self):
        (self.root / "teacher.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "teacher_latent_path"):
            self.join()
        with self.assertRaisesRegex(ValueError, "condition-only caches are insufficient"):
            manifest.validate_manifest(self.condition)

    def test_rejects_duplicate_condition_and_duplicate_fingerprint(self):
        self.write("real.jsonl", [self.real_row, self.real_row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.join()

    def test_rejects_extraneous_target_rows(self):
        (self.root / "other.pt").touch()
        self.write("real.jsonl", [self.real_row, {**self.real_row, "condition_path": "other.pt", "cache_fingerprint": "other"}])
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self.join()

    def test_rejects_real_teacher_aliasing(self):
        self.write("teacher.jsonl", [{**self.teacher_row, "teacher_latent_path": "real.pt"}])
        with self.assertRaisesRegex(ValueError, "distinct files"):
            self.join()

    def test_rejects_invalid_and_conflicting_geometry(self):
        for changes in ({"target_height": 33}, {"target_num_frames": 106}, {"target_num_frames": 107.5}, {"num_frames": 124}, {"target_orientation": "portrait"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                manifest.target_geometry({**self.row, **changes})

    def test_atomic_writer_never_overwrites(self):
        output = self.root / "paired.jsonl"
        rows = self.join()
        manifest.write_manifest_atomic(rows, output)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            manifest.write_manifest_atomic([], output)
        self.assertEqual(output.read_bytes(), original)
        self.assertFalse(list(self.root.glob(".paired.jsonl.*.tmp")))

    def test_cli_runs_without_site_packages(self):
        output = self.root / "paired.jsonl"
        result = subprocess.run(
            [sys.executable, "-S", str(MODULE_PATH), "--conditions", str(self.condition), "--real", str(self.real), "--teacher", str(self.teacher), "--output", str(output)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Joined 1", result.stdout)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
