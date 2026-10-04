#!/usr/bin/env python3

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
