"""CPU contracts for raw public metadata, four-worker accounting and views."""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

_spec = importlib.util.spec_from_file_location("_selfgen_prepare", Path(__file__).with_name("prepare_modelscope_h3_dataset.py"))
helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helper)


class SelfGeneratedPreparationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw, self.output = self.root / "raw", self.root / "encoded"
        self.raw.mkdir()
        self.t2av = [self.raw_row("videos/000001.mp4"), self.raw_row("videos/000002.mp4", height=1024, width=992)]
        self.refs = []
        for index, (kind, labels, extension) in enumerate((("image", "Picture 1", "jpg"), ("video", "Video 1", "mp4"), ("video", "Audio 1 + Video 1", "mp4")), 1):
            self.refs.append({**self.raw_row(f"ref2va_videos/{index:06d}.mp4"), "reference_kind": kind, "reference_labels": labels, "reference_path": f"references/{index:06d}.{extension}"})
        self.write_raw()

    @staticmethod
    def raw_row(video, *, height=768, width=1344):
        return {"prompt": "  A structured H3 prompt.\n\noverall_soundscape: dialogue.  ", "video_path": video, "height": height, "width": width, "num_frames": 124, "num_inference_steps": 30}

    def write_raw(self):
        helper.atomic_jsonl(self.raw / "metadata.jsonl", self.t2av)
        helper.atomic_jsonl(self.raw / "ref2va_metadata.jsonl", self.refs)
        for row in self.t2av + self.refs:
            for key in ("video_path", "reference_path"):
                if key in row:
                    path = self.raw / row[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"metadata-only fixture, never decode")

    def prepare(self, **options):
        args = dict(dataset_dir=self.raw, output_dir=self.output, subsets="all", max_samples_per_subset=None, allow_missing_media=False)
        helper.prepare_dataset(SimpleNamespace(**{**args, **options}))
        return helper.read_jsonl(self.output / "prepared.jsonl")

    def make_shards(self, rows, *, failure=None):
        for shard in range(4):
            selected = [row for index, row in enumerate(rows) if index % 4 == shard]
            complete, failed = [], []
            for row in selected:
                if row["source_id"] == failure:
                    failed.append({"source_id": failure, "error": "intentional sample failure"})
                    continue
                record = {
                    **row,
                    "cache_fingerprint": "test:" + row["source_id"],
                    "reference_image_count": int(row.get("reference_kind") == "image"),
                    "reference_video_count": int(row.get("reference_kind") == "video"),
                    "reference_audio_count": 0,
                }
                for field in ("condition_path", "video_latent_path", "audio_latent_path"):
                    target = self.output / "payload" / f"{row['source_index']}_{field}.pt"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"complete synthetic tensor file")
                    record[field] = str(target)
                complete.append(record)
            suffix = f"{shard:05d}-of-00004"
            helper.atomic_jsonl(self.output / f"metadata.shard-{suffix}.jsonl", complete)
            helper.atomic_jsonl(self.output / f"failed.shard-{suffix}.jsonl", failed)
            helper.atomic_json(
                self.output / f"receipt.shard-{suffix}.json",
                {
                    "schema_version": 1,
                    "prepared_sha256": helper.file_sha256(self.output / "prepared.jsonl"),
                    "shard_index": shard,
                    "num_shards": 4,
                    "expected_count": len(selected),
                    "completed_count": len(complete),
                    "failed_count": len(failed),
                },
            )

    def merge(self, **options):
        helper.merge_dataset(SimpleNamespace(output_dir=self.output, num_shards=4, allow_failures=options.get("allow_failures", False)))

    def test_preparation_preserves_all_native_geometries_prompts_and_reference_audio_semantics(self):
        rows = self.prepare()
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["caption"], self.t2av[0]["prompt"])
        self.assertEqual((rows[1]["target_height"], rows[1]["target_width"]), (1024, 992))
        self.assertNotIn("include_embedded_audio", rows[2]["references"][0])
        self.assertFalse(rows[3]["references"][0]["include_embedded_audio"])
        self.assertTrue(rows[4]["references"][0]["include_embedded_audio"])
        self.assertEqual(len({row["source_id"] for row in rows}), 5)
        self.assertTrue(all(Path(row["video_path"]).is_absolute() for row in rows))
        previous = (self.output / "prepared.jsonl").read_bytes()
        self.prepare()
        self.assertEqual(previous, (self.output / "prepared.jsonl").read_bytes())

    def test_missing_published_reference_is_reported_and_not_replaced_with_an_unconditional_sample(self):
        (self.raw / self.refs[0]["reference_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        rows = self.prepare(allow_missing_media=True)
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["source_index"] for row in rows], [0, 1, 3, 4])
        self.assertEqual(len(helper.read_jsonl(self.output / "prepare_excluded_missing_media.jsonl")), 1)
        self.make_shards(rows)
        self.merge()
        self.assertEqual(len(helper.read_jsonl(self.output / "all_metadata.jsonl")), 4)

    def test_bad_geometry_path_traversal_and_duplicate_media_fail(self):
        for changes in ({"height": 769}, {"video_path": "../outside.mp4"}, {"num_frames": 121}):
            row = {**self.t2av[0], **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                helper.prepared_row(row, "t2av", self.raw, 0, "hash")
        self.t2av.append(copy.deepcopy(self.t2av[0]))
        helper.atomic_jsonl(self.raw / "metadata.jsonl", self.t2av)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_smoke_limit_applies_to_each_subset(self):
        rows = self.prepare(max_samples_per_subset=1)
        self.assertEqual([row["task"] for row in rows], ["t2av", "ref2av"])

    def test_merge_checks_complete_coverage_then_publishes_compatible_training_views(self):
        rows = self.prepare()
        self.make_shards(rows)
        self.merge()
        expected_counts = {"all_metadata.jsonl": 5, "metadata.jsonl": 1, "t2av/metadata.jsonl": 2, "t2av_768p/metadata.jsonl": 1, "ref2av/metadata.jsonl": 3, "ref2av_imageonly_768p/metadata.jsonl": 1}
        for name, count in expected_counts.items():
            self.assertEqual(len(helper.read_jsonl(self.output / name)), count, name)
            path = self.output / name
            receipt = json.loads(path.with_suffix(".complete.json").read_text())
            self.assertEqual(receipt["completed_count"], count)
            self.assertEqual(receipt["manifest_sha256"], helper.file_sha256(path))
            self.assertEqual(receipt["completed_count"] + receipt["failed_count"], receipt["input_total_rows"])
        for row in helper.read_jsonl(self.output / "condition_metadata.jsonl"):
            self.assertNotIn("video_latent_path", row)
            self.assertNotIn("audio_latent_path", row)
            self.assertIn("condition_path", row)

    def test_merge_rejects_wrong_shard_identity_missing_payload_and_stale_receipts(self):
        rows = self.prepare()
        self.make_shards(rows)
        receipt = self.output / "receipt.shard-00000-of-00004.json"
        saved = json.loads(receipt.read_text())
        helper.atomic_json(receipt, {**saved, "prepared_sha256": "different"})
        with self.assertRaisesRegex(ValueError, "receipt"):
            self.merge()
        helper.atomic_json(receipt, saved)
        manifest = self.output / "metadata.shard-00000-of-00004.jsonl"
        valid = helper.read_jsonl(manifest)
        helper.atomic_jsonl(manifest, [{**valid[0], "source_id": rows[1]["source_id"]}, *valid[1:]])
        with self.assertRaisesRegex(ValueError, "identity"):
            self.merge()
        helper.atomic_jsonl(manifest, valid)
        Path(valid[0]["audio_latent_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "audio_latent_path"):
            self.merge()
        self.assertFalse((self.output / "cache.complete.json").exists())

    def test_failures_require_explicit_opt_in_for_success_only_merge(self):
        rows = self.prepare()
        self.make_shards(rows, failure=rows[2]["source_id"])
        with self.assertRaisesRegex(RuntimeError, "Encoding failed"):
            self.merge()
        self.merge(allow_failures=True)
        self.assertEqual(len(helper.read_jsonl(self.output / "all_metadata.jsonl")), 4)
        self.assertEqual(len(helper.read_jsonl(self.output / "failed.jsonl")), 1)

    def test_preparation_does_not_replace_input_namespace_after_encoding_started(self):
        rows = self.prepare()
        self.make_shards(rows)
        before = (self.output / "prepared.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "new --output-dir"):
            self.prepare(max_samples_per_subset=1)
        self.assertEqual(before, (self.output / "prepared.jsonl").read_bytes())

    def test_download_uses_resumable_sdk_without_loading_gpu_models(self):
        sdk = ModuleType("modelscope.hub.snapshot_download")
        sdk.dataset_snapshot_download = Mock(return_value=str(self.raw))
        with patch.dict(sys.modules, {"modelscope.hub.snapshot_download": sdk}):
            helper.download_dataset(SimpleNamespace(dataset_dir=self.raw, revision="fixed-revision", download_workers=4))
        sdk.dataset_snapshot_download.assert_called_once_with(dataset_id=helper.DATASET_ID, revision="fixed-revision", local_dir=str(self.raw), max_workers=4)
        self.assertTrue((self.raw / "download.complete.json").is_file())


if __name__ == "__main__":
    unittest.main()
