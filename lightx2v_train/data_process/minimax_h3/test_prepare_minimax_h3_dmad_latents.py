"""CPU-only real-target preparation preflight tests; no encoder is invoked."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("prepare_minimax_h3_dmad_latents.py")
SPEC = importlib.util.spec_from_file_location("dmad_real_preparation", MODULE_PATH)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class DMADRealPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.condition_rows, self.target_rows = [], []
        for index in range(4):
            (self.root / f"condition_{index}.pt").touch()
            (self.root / f"video_{index}.mp4").touch()
            self.condition_rows.append(
                {"source_id": str(index), "cache_fingerprint": f"fp-{index}", "condition_path": f"condition_{index}.pt", "target_height": 32, "target_width": 64, "num_frames": 107}
            )
            self.target_rows.append({"source_id": str(index), "video_path": f"video_{index}.mp4"})
        self.conditions = self.write("conditions.jsonl", self.condition_rows)
        self.targets = self.write("targets.jsonl", self.target_rows)

    def write(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def jobs(self, **kwargs):
        return prepare.prepare_jobs(self.conditions, self.targets, **kwargs)

    def test_binds_unique_real_sources_and_preserves_exact_condition_identity(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual(jobs[0]["condition"]["cache_fingerprint"], "fp-0")
        self.assertEqual(jobs[0]["condition"]["target_num_frames"], 107)
        self.assertEqual(jobs[0]["video_path"], str(self.root / "video_0.mp4"))
        self.assertEqual(jobs[0]["audio_path"], jobs[0]["video_path"])

    def test_custom_identity_column_separate_audio_and_target_video_alias(self):
        (self.root / "sound.wav").touch()
        rows = [{"sample_id": row["source_id"], "target_video_path": row["video_path"], "audio_path": "sound.wav"} for row in self.target_rows]
        self.write("targets.jsonl", rows)
        jobs = self.jobs(identity_field="sample_id")
        self.assertEqual(jobs[0]["audio_path"], str(self.root / "sound.wav"))

    def test_rank_partition_is_disjoint_and_global_max_samples_applies_first(self):
        first = self.jobs(rank=0, world_size=2, max_samples=3)
        second = self.jobs(rank=1, world_size=2, max_samples=3)
        self.assertEqual([job["condition"]["source_id"] for job in first], ["0", "2"])
        self.assertEqual([job["condition"]["source_id"] for job in second], ["1"])

    def test_rejects_duplicate_real_source_and_ambiguous_condition_source(self):
        self.write("targets.jsonl", self.target_rows + [self.target_rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate real target"):
            self.jobs()
        self.write("targets.jsonl", self.target_rows)
        self.write("conditions.jsonl", self.condition_rows + [{**self.condition_rows[0], "cache_fingerprint": "another"}])
        with self.assertRaisesRegex(ValueError, "Ambiguous/duplicate"):
            self.jobs()

    def test_rejects_missing_video_and_source_without_line_number_fallback(self):
        self.write("targets.jsonl", self.target_rows[1:])
        with self.assertRaisesRegex(ValueError, "Missing ground-truth"):
            self.jobs()
        self.write("targets.jsonl", [{"video_path": "video_0.mp4"}])
        with self.assertRaisesRegex(ValueError, "row numbers are not identities"):
            self.jobs()

    def test_rejects_conflicting_declared_condition_identity_and_geometry(self):
        for changes in ({"cache_fingerprint": "wrong"}, {"condition_path": "condition_1.pt"}, {"target_height": 32, "target_width": 96, "num_frames": 107}):
            with self.subTest(changes=changes):
                self.write("targets.jsonl", [{**self.target_rows[0], **changes}, *self.target_rows[1:]])
                with self.assertRaises(ValueError):
                    self.jobs()

    def test_cli_dry_run_is_stdlib_only_and_does_not_create_outputs(self):
        model = self.root / "model"
        for component in ("vae", "audio_vae"):
            directory = model / component
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
        output = self.root / "new_output"
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(MODULE_PATH),
                "--conditions",
                str(self.conditions),
                "--targets",
                str(self.targets),
                "--model-path",
                str(model),
                "--output-dir",
                str(output),
                "--max-samples",
                "2",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["samples"], 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
