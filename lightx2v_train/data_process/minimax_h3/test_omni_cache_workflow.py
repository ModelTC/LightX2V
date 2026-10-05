#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("merge_omni_ref2av_cache.py")
SPEC = importlib.util.spec_from_file_location("merge_omni", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REPO = SCRIPT.parents[3]


class OmniCacheWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "conditions").mkdir()
        (self.root / "preprocess_config.json").write_text(
            json.dumps(
                {
                    "preprocess_fingerprint": "fixture",
                    "preprocess_config": {
                        "selection": {"num_shards": 32, "start_index": 0, "max_samples": None},
                        "dtype": "bf16",
                        "reference_latent_dtype": "bf16",
                        "reference_image_resize_mode": "match",
                        "target_policy": "fixed-768p",
                        "prompt_policy": "enhanced-or-original",
                        "image_only": True,
                    },
                }
            )
        )
        for index in range(32):
            condition = self.root / "conditions" / f"condition_{index:08d}.pt"
            condition.write_bytes(b"already-validated-by-encoder")
            stem = f"metadata.shard-{index:03d}-of-032"
            manifest = self.root / f"{stem}.jsonl"
            failed = self.root / f"{stem}.failed.jsonl"
            rows = [
                {
                    "condition_path": str(condition.relative_to(self.root)),
                    "source_index": index,
                    "ref_image_count": 1 + index % 5,
                    "reference_image_count": 1 + index % 5,
                    "reference_video_count": 0,
                    "reference_audio_count": 0,
                    "target_height": 768,
                    "target_width": 1344,
                    "num_frames": 124,
                    "prompt_source": "prompt_en" if index == 0 else "enhanced_prompt",
                }
            ]
            MODULE.atomic_lines(manifest, rows)
            MODULE.atomic_lines(failed, [])
            MODULE.atomic_lines(
                self.root / f"{stem}.complete.json",
                [
                    {
                        "preprocess_fingerprint": "fixture",
                        "stage": "all",
                        "num_shards": 32,
                        "shard_index": index,
                        "input_total_rows": 32,
                        "selected_count": 1,
                        "completed_count": 1,
                        "failed_count": 0,
                        "manifest_sha256": MODULE.digest(manifest),
                        "failures_sha256": MODULE.digest(failed),
                    }
                ],
            )

    def merge(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return MODULE.merge(self.root)

    def test_merge_publishes_verified_full_manifest(self):
        summary = self.merge()
        self.assertEqual(summary["completed_count"], 32)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["prompt_sources"]["prompt_en"], 1)
        self.assertEqual(summary["manifest_sha256"], MODULE.digest(self.root / "metadata.jsonl"))
        self.assertEqual(len(list(MODULE.records(self.root / "metadata.jsonl"))), 32)

    def test_missing_receipt_prevents_publication(self):
        (self.root / "metadata.shard-007-of-032.complete.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "not complete"):
            self.merge()
        self.assertFalse((self.root / "metadata.jsonl").exists())

    def test_manifest_changed_after_completion_is_rejected(self):
        with (self.root / "metadata.shard-003-of-032.jsonl").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "Changed shard file"):
            self.merge()

    def test_missing_cache_is_rejected(self):
        (self.root / "conditions" / "condition_00000002.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.merge()

    def test_failed_rows_are_accounted_but_excluded_from_training(self):
        stem = "metadata.shard-005-of-032"
        manifest, failures = self.root / f"{stem}.jsonl", self.root / f"{stem}.failed.jsonl"
        MODULE.atomic_lines(manifest, [])
        MODULE.atomic_lines(failures, [{"source_index": 5, "error": "missing image"}])
        receipt_path = self.root / f"{stem}.complete.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(completed_count=0, failed_count=1, manifest_sha256=MODULE.digest(manifest), failures_sha256=MODULE.digest(failures))
        MODULE.atomic_lines(receipt_path, [receipt])
        summary = self.merge()
        self.assertEqual(summary["completed_count"], 31)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(list(MODULE.records(self.root / "failed.jsonl"))[0]["source_index"], 5)

    def test_wrong_shard_ownership_is_rejected(self):
        stem = "metadata.shard-005-of-032"
        manifest = self.root / f"{stem}.jsonl"
        row = next(MODULE.records(manifest))
        row["source_index"] = 6
        MODULE.atomic_lines(manifest, [row])
        receipt_path = self.root / f"{stem}.complete.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["manifest_sha256"] = MODULE.digest(manifest)
        MODULE.atomic_lines(receipt_path, [receipt])
        with self.assertRaisesRegex(ValueError, "source_index"):
            self.merge()

    def test_four_shell_entrypoints_select_32_distinct_workers(self):
        env = {**os.environ, "H3_CODE_ROOT": str(REPO), "H3_PYTHON": "python3", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
        indices = []
        for node in range(4):
            script = REPO / "lightx2v_train/scripts" / f"cache_omni_ref2av_node{node}.sh"
            result = subprocess.run(["bash", str(script), "--dry-run"], env=env, text=True, capture_output=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 8)
            for line in lines:
                self.assertIn("--reference-latent-dtype bf16", line)
                self.assertIn("--reference-resize-mode match", line)
                self.assertIn("--prompt-policy enhanced-or-original", line)
                indices.append(int(line.split("--shard-index ", 1)[1]))
        self.assertEqual(indices, list(range(32)))


if __name__ == "__main__":
    unittest.main()
