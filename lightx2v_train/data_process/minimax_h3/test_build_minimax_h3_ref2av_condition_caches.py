#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).with_name("build_minimax_h3_ref2av_condition_caches.py")
SPEC = importlib.util.spec_from_file_location("ref2av_cache_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Ref2AVConditionCacheSchemaTest(unittest.TestCase):
    def test_merge_does_not_require_model_paths(self):
        args = MODULE.parse_args(["--output-dir", "/path/to/cache", "--merge-shards"])
        self.assertTrue(args.merge_shards)
        self.assertIsNone(args.model_path)
        self.assertIsNone(args.source_model_path)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.media = []
        for name, contents in (
            ("image.jpg", b"image"),
            ("video.mp4", b"video"),
            ("audio.mp3", b"audio"),
        ):
            path = self.root / name
            path.write_bytes(contents)
            self.media.append(path)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _row(self):
        return {
            "id": "sample-7",
            "task": "ref2va",
            "enhanced_prompt": "An enhanced multimodal reference prompt.",
            "duration": 4,
            "target_num_frames": 107,
            "target_height": 768,
            "target_width": 1344,
            "references": [
                {
                    "order": 1,
                    "type": "image",
                    "local_path": self.media[0].name,
                    "sha256": self._sha(self.media[0]),
                },
                {
                    "order": 2,
                    "role": "reference_video",
                    "path": self.media[1].name,
                    "sha256": self._sha(self.media[1]),
                    "include_embedded_audio": False,
                },
                {
                    "order": 3,
                    "kind": "audio",
                    "path": self.media[2].name,
                    "sha256": self._sha(self.media[2]),
                },
            ],
        }

    def _sample(self):
        return MODULE.normalize_sample(
            self._row(),
            source_index=7,
            media_root=self.root,
            dtype_name="bf16",
            model_descriptor={"identity": "mock-ref2va"},
        )

    def test_canonical_order_hash_and_geometry_are_fingerprinted(self):
        sample = self._sample()
        self.assertEqual([reference.kind for reference in sample.references], ["image", "video", "audio"])
        self.assertEqual(sample.target_num_frames, 107)
        self.assertEqual(
            [reference["sha256"] for reference in sample.descriptor["references"]],
            [self._sha(path) for path in self.media],
        )
        reversed_descriptor = dict(sample.descriptor)
        reversed_descriptor["references"] = list(reversed(sample.descriptor["references"]))
        self.assertNotEqual(sample.fingerprint, MODULE.digest_object(reversed_descriptor))

    def test_match_resize_and_fixed_target_override_are_fingerprinted(self):
        sample = MODULE.normalize_sample(
            self._row(),
            source_index=7,
            media_root=self.root,
            dtype_name="bf16",
            model_descriptor={"identity": "mock-ref2va"},
            reference_resize_mode="match",
            target_num_frames_override=124,
        )
        self.assertEqual(sample.target_num_frames, 124)
        self.assertEqual(sample.descriptor["source_target_num_frames"], 107)
        self.assertEqual(sample.descriptor["reference_image_resize_mode"], "match")
        self.assertEqual(
            MODULE.resolve_reference_image_size(
                1920,
                1080,
                target_width=1344,
                target_height=768,
                mode="match",
            ),
            (768, 1344),
        )
        self.assertEqual(
            MODULE.resolve_reference_image_size(
                640,
                360,
                target_width=1344,
                target_height=768,
                mode="match",
            ),
            (352, 640),
        )

    def _omni_row(self):
        row = self._row()
        row.update(task="ref2v", target_height=1066, target_width=1906, sample_id="omni-image-7")
        row.pop("enhanced_prompt")
        row.pop("duration")
        row.pop("target_num_frames")
        row["prompt_en"] = " <Figure 1> shows a person. <Subject 2> walks away. "
        row["references"] = [{"modality": "image", "rel_path": self.media[0].name}]
        row["ref_image_count"] = 99  # Recomputed, never trust stale annotations.
        return row

    def _omni_sample(self, row=None):
        return MODULE.normalize_sample(
            self._omni_row() if row is None else row,
            7,
            self.root,
            "bf16",
            {"identity": "mock"},
            prompt_policy="enhanced-or-original",
            target_policy="fixed-768p",
            reference_resize_mode="match",
            reference_latent_dtype="bf16",
            image_only=True,
        )

    def test_omni_prompt_fallback_and_fixed_geometry_need_no_target_video(self):
        sample = self._omni_sample()
        self.assertEqual(sample.prompt, "<Picture 1> shows a person. <Subject 2> walks away.")
        self.assertEqual(sample.descriptor["prompt_source"], "prompt_en")
        self.assertEqual(sample.descriptor["ref_image_count"], 1)
        self.assertEqual(sample.descriptor["reference_latent_dtype"], "bf16")
        self.assertEqual((sample.target_height, sample.target_width, sample.target_num_frames), (768, 1344, 124))
        self.assertIsNone(sample.descriptor["source_target_num_frames"])
        row = self._omni_row()
        row.update(target_height=3840, target_width=2160, duration=999, target_num_frames=1000)
        portrait = self._omni_sample(row)
        self.assertEqual((portrait.target_height, portrait.target_width), (1344, 768))
        self.assertEqual(portrait.target_num_frames, 124)
        self.assertEqual(portrait.descriptor["source_target_num_frames"], 1000)
        self.assertEqual(portrait.descriptor["source_target_geometry"]["duration"], 999)

    def test_prompt_priority_original_aliases_and_empty_rejection(self):
        row = self._omni_row()
        row.update(enhanced_prompt=" Keep <Figure 1> unchanged in enhanced. ", prompt_cn="中文")
        self.assertEqual(self._omni_sample(row).prompt, "Keep <Figure 1> unchanged in enhanced.")
        row["enhanced_prompt"] = " "
        row["prompt_en"] = None
        row["prompt_en_original"] = "@Image1 English"
        self.assertEqual(self._omni_sample(row).descriptor["prompt_source"], "prompt_en_original")
        row["prompt_en_original"] = ""
        row["prompt_cn"] = "@图片1人物"
        self.assertEqual(self._omni_sample(row).prompt, "<Picture 1>人物")
        row["prompt_cn"] = ""
        row["prompt_cn_original"] = "@图1中文"
        self.assertEqual(self._omni_sample(row).prompt, "<Picture 1>中文")
        row["prompt_cn_original"] = ""
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self._omni_sample(row)
        with self.assertRaisesRegex(ValueError, "enhanced_prompt"):
            MODULE.normalize_sample(self._omni_row(), 7, self.root, "bf16", {})

    def test_fixed_geometry_orientation_fallback_and_image_only_rejection(self):
        row = self._omni_row()
        row.pop("target_width")
        row.pop("target_height")
        row["target_orientation"] = "portrait"
        self.assertEqual(self._omni_sample(row).target_height, 1344)
        row.pop("target_orientation")
        with self.assertRaisesRegex(ValueError, "target_orientation"):
            self._omni_sample(row)
        row = self._omni_row()
        row["references"].append({"kind": "video", "path": str(self.media[1])})
        with self.assertRaisesRegex(ValueError, "not image-only"):
            self._omni_sample(row)

    def _text_payload(self, sample):
        return {
            "cache_schema_version": 1,
            "cache_fingerprint": sample.fingerprint,
            "cache_metadata": sample.descriptor,
            "cache_stage": "text",
            "conditioning": {
                "positive": {
                    "task": "ref2av",
                    "target_height": sample.target_height,
                    "target_width": sample.target_width,
                    "target_num_frames": sample.target_num_frames,
                    "prompt_embeds": torch.zeros(1, 2, 5120, dtype=torch.bfloat16),
                    "text_token_tags": torch.tensor([0, 1], dtype=torch.long),
                }
            },
        }

    def test_bf16_storage_keeps_integer_tags_and_all_float_tensors_bf16(self):
        sample = self._omni_sample()
        prepared = [SimpleNamespace(num_latent_frames=1, latent_height=4, latent_width=4)]
        components = SimpleNamespace(audio_vae=SimpleNamespace(config=SimpleNamespace(sampling_rate=32000)), _execution_device="cpu")
        runtime = {
            "MiniMaxH3Ref2VAReferenceEncoderStep": SimpleNamespace(
                encode_references=lambda *args, **kwargs: (torch.ones(4, 96, dtype=torch.float32), None),
            ),
            "MiniMaxH3Ref2VATextEncoderStep": SimpleNamespace(
                encode_prompt=lambda *args, **kwargs: (torch.ones(1, 2, 5120, dtype=torch.float32), torch.tensor([0, 1])),
            ),
        }
        with patch.object(MODULE, "prepare_official_references", return_value=prepared):
            text_payload = MODULE.encode_text_stage(sample, runtime, components, torch.bfloat16)
            payload = MODULE.encode_reference_stage(sample, text_payload, runtime, components, torch.bfloat16)
        positive = payload["conditioning"]["positive"]
        self.assertEqual(positive["prompt_embeds"].dtype, torch.bfloat16)
        self.assertEqual(positive["references"][0]["video_latents"].dtype, torch.bfloat16)
        self.assertEqual(positive["text_token_tags"].dtype, torch.long)
        manifest = MODULE.manifest_row(sample, Path("conditions/a.pt"), payload)
        self.assertEqual(manifest["ref_image_count"], 1)
        self.assertEqual(manifest["source_index"], 7)
        self.assertEqual(manifest["source_id"], "omni-image-7")
        self.assertEqual(manifest["prompt_source"], "prompt_en")
        self.assertEqual(manifest["cache_fingerprint"], sample.fingerprint)
        fp32_sample = MODULE.normalize_sample(
            self._omni_row(),
            7,
            self.root,
            "bf16",
            {"identity": "mock"},
            prompt_policy="enhanced-or-original",
            target_policy="fixed-768p",
            image_only=True,
            reference_resize_mode="match",
        )
        self.assertNotEqual(sample.fingerprint, fp32_sample.fingerprint)

    def test_streamed_sharding_matches_old_selection_without_full_parse(self):
        rows = [{"id": index} for index in range(11)]
        path = self.root / "input.jsonl"
        path.write_text("\n".join(json.dumps(row) + "\n" for row in rows))
        for shard in range(4):
            args = SimpleNamespace(start_index=2, max_samples=7, num_shards=4, shard_index=shard)
            actual, total = MODULE.read_selected_rows(path, args)
            self.assertEqual(actual, MODULE._selected_rows(rows, args))
            self.assertEqual(total, 11)

    def test_skip_invalid_receipt_coverage_and_resume(self):
        source = self.root / "input.jsonl"
        valid = self._omni_row()
        invalid = {**valid, "sample_id": "missing", "references": [{"kind": "image", "path": "missing.jpg"}]}
        MODULE.atomic_write_jsonl(source, [valid, invalid])
        output = self.root / "output"
        argv = [
            str(source),
            "--output-dir",
            str(output),
            "--model-path",
            str(self.root),
            "--source-model-path",
            str(self.root),
            "--prompt-policy",
            "enhanced-or-original",
            "--target-policy",
            "fixed-768p",
            "--reference-resize-mode",
            "match",
            "--reference-latent-dtype",
            "bf16",
            "--image-only",
            "--skip-invalid",
            "--num-shards",
            "2",
        ]

        # Shard zero contains the valid record, shard one the missing record.
        def complete(sample, payload, *args):
            payload["cache_stage"] = "complete"
            payload["conditioning"]["positive"]["references"] = [
                {
                    "kind": "image",
                    "normalized": True,
                    "video_latents": torch.ones(4, 96, dtype=torch.bfloat16),
                    "num_latent_frames": 1,
                    "latent_height": 4,
                    "latent_width": 4,
                }
            ]
            return payload

        with (
            patch.object(MODULE, "model_identity", return_value={"model": "mock"}),
            patch.object(MODULE, "import_official_runtime", return_value={}),
            patch.object(MODULE, "load_conditioner", return_value=object()),
            patch.object(MODULE, "load_vaes", return_value=object()),
            patch.object(MODULE, "encode_text_stage", side_effect=lambda sample, *args: self._text_payload(sample)) as text_encoder,
            patch.object(MODULE, "encode_reference_stage", side_effect=complete),
        ):
            self.assertEqual(MODULE.main(argv), 0)
            self.assertEqual(MODULE.main(argv), 0)
            self.assertEqual(text_encoder.call_count, 1)  # Valid existing caches are reused.
            self.assertEqual(MODULE.main(argv + ["--shard-index", "1"]), 0)
            receipt_zero = output / "metadata.shard-000-of-002.complete.json"
            with patch.object(MODULE, "encode_text_stage", side_effect=RuntimeError("CUDA out of memory")):
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    MODULE.main(argv + ["--overwrite"])
            self.assertFalse(receipt_zero.exists())  # --skip-invalid must not hide GPU failures.
            self.assertEqual(MODULE.main(argv), 0)
            with self.assertRaisesRegex(RuntimeError, "different input/model"):
                MODULE.main(argv + ["--reference-latent-dtype", "fp32"])
            self.assertTrue(receipt_zero.exists())  # Wrong namespace must not damage a completed run.
        for shard, completed, failed in [(0, 1, 0), (1, 0, 1)]:
            receipt = json.loads((output / f"metadata.shard-{shard:03d}-of-002.complete.json").read_text())
            self.assertEqual(receipt["input_total_rows"], 2)
            self.assertEqual(receipt["selected_count"], 1)
            self.assertEqual(receipt["completed_count"], completed)
            self.assertEqual(receipt["failed_count"], failed)
            self.assertEqual(receipt["manifest_sha256"], MODULE.sha256_file(output / receipt["manifest"]))
            self.assertEqual(receipt["failures_sha256"], MODULE.sha256_file(output / receipt["failures"]))
        failures = MODULE.read_jsonl(output / "metadata.shard-001-of-002.failed.jsonl")
        self.assertEqual(failures[0]["source_index"], 1)

    def test_refuses_silent_reorder_or_stale_declared_hash(self):
        row = self._row()
        row["references"][1]["order"] = 3
        with self.assertRaisesRegex(ValueError, "Refusing to silently reorder"):
            MODULE.normalize_sample(row, 0, self.root, "bf16", {"identity": "mock"})

        row = self._row()
        row["references"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            MODULE.normalize_sample(row, 0, self.root, "bf16", {"identity": "mock"})

    def test_complete_payload_matches_training_schema(self):
        sample = self._sample()
        positive = {
            "task": "ref2av",
            "prompt_embeds": torch.zeros(1, 4, 5120, dtype=torch.bfloat16),
            "text_token_tags": torch.tensor([1, 0, 0, 1], dtype=torch.long),
            "target_height": 768,
            "target_width": 1344,
            "target_num_frames": 107,
            "references": [
                {
                    "kind": "image",
                    "normalized": True,
                    "video_latents": torch.zeros(4, 96),
                    "num_latent_frames": 1,
                    "latent_height": 4,
                    "latent_width": 4,
                },
                {
                    "kind": "video",
                    "normalized": True,
                    "video_latents": torch.zeros(12, 96),
                    "num_latent_frames": 2,
                    "latent_height": 4,
                    "latent_width": 6,
                },
                {
                    "kind": "audio",
                    "normalized": True,
                    "audio_latents": torch.zeros(6, 32),
                    "num_audio_latents": 3,
                },
            ],
        }
        payload = {
            "cache_schema_version": 1,
            "cache_stage": "complete",
            "cache_fingerprint": sample.fingerprint,
            "conditioning": {"positive": positive},
        }
        stage = MODULE.validate_cache_payload(
            payload,
            sample,
            self.root / "mock.pt",
            torch.bfloat16,
            require_complete=True,
        )
        self.assertEqual(stage, "complete")
        row = MODULE.manifest_row(sample, Path("conditions/condition_00000007.pt"))
        self.assertEqual(
            set(row),
            {"condition_path", "target_height", "target_width", "num_frames"},
        )
        enriched = MODULE.manifest_row(
            sample,
            Path("conditions/condition_00000007.pt"),
            payload,
        )
        self.assertEqual(enriched["reference_image_count"], 1)
        self.assertEqual(enriched["reference_video_count"], 1)
        self.assertEqual(enriched["reference_audio_count"], 1)
        self.assertEqual(enriched["prompt_token_count"], 4)
        self.assertEqual(enriched["reference_video_rows"], 16)
        self.assertEqual(enriched["reference_audio_rows"], 6)
        self.assertEqual(enriched["reference_compute_cost"], 26)
        self.assertEqual(
            enriched["packed_sequence_tokens_124"],
            MODULE.FIXED_DMD_TARGET_ROWS + 26,
        )

    def test_namespace_publication_is_no_clobber(self):
        path = self.root / "preprocess_config.json"
        MODULE.atomic_create_json(path, {"owner": "first"})
        MODULE.atomic_create_json(path, {"owner": "second"})
        import json

        self.assertEqual(json.loads(path.read_text())["owner"], "first")

    def test_shard_merge_refuses_incomplete_cache(self):
        condition_dir = self.root / "conditions"
        condition_dir.mkdir()
        condition = condition_dir / "condition_00000000.pt"
        torch.save(
            {
                "cache_schema_version": 1,
                "cache_stage": "text",
                "conditioning": {"positive": {"references": []}},
            },
            condition,
        )
        row = {
            "condition_path": "conditions/condition_00000000.pt",
            "target_height": 768,
            "target_width": 1344,
            "num_frames": 107,
        }
        MODULE.atomic_write_jsonl(
            self.root / "metadata.shard-000-of-001.jsonl",
            [row],
        )
        args = SimpleNamespace(output_dir=self.root, num_shards=1)
        with self.assertRaisesRegex(ValueError, "incomplete cache"):
            MODULE.merge_shards(args)
        self.assertFalse((self.root / "metadata.jsonl").exists())

        torch.save(
            {
                "cache_schema_version": 1,
                "cache_stage": "complete",
                "conditioning": {"positive": {"references": [{"kind": "image"}]}},
            },
            condition,
        )
        MODULE.merge_shards(args)
        self.assertEqual(MODULE.read_jsonl(self.root / "metadata.jsonl"), [row])


if __name__ == "__main__":
    unittest.main()
