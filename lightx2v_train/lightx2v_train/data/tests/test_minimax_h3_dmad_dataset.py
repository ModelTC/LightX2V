import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from lightx2v_train.data.minimax_h3_dmad_dataset import MiniMaxH3DMADDataset, build_minimax_h3_dmad_dataset, pack_target_payload
from lightx2v_train.utils.utils import is_train_cache_dataset


class DMADDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = {
            "condition_path": str(self.root / "condition.pt"),
            "cache_fingerprint": "fp-1",
            "source_id": "source-1",
            "target_height": 32,
            "target_width": 64,
            "target_num_frames": 107,
            "target_orientation": "landscape",
            "caption": "test",
            "real_latent_path": str(self.root / "real.pt"),
            "teacher_latent_path": str(self.root / "teacher.pt"),
            "reference_image_count": 1,
            "reference_video_count": 0,
            "reference_audio_count": 0,
            "packed_sequence_tokens_124": 123,
        }
        self.condition = {
            "cache_fingerprint": "fp-1",
            "source_id": "source-1",
            "conditioning": {
                "positive": {
                    "task": "ref2av",
                    "target_height": 32,
                    "target_width": 64,
                    "target_num_frames": 107,
                    "prompt_embeds": torch.ones(1, 3, 4),
                    "text_token_tags": torch.tensor([0, 1, 1]),
                    "references": [{"kind": "image", "normalized": True, "video_latents": torch.ones(2, 96)}],
                }
            },
        }
        self.payload = {
            "cache_fingerprint": "fp-1",
            "source_id": "source-1",
            "normalized": True,
            "target_height": 32,
            "target_width": 64,
            "target_num_frames": 107,
            "video": torch.arange(64 * 96, dtype=torch.float32).reshape(64, 96),
            "audio": torch.zeros(356, 32),
        }
        torch.save(self.condition, self.row["condition_path"])
        torch.save(self.payload, self.row["real_latent_path"])
        torch.save({**self.payload, "audio": torch.ones(356, 32)}, self.row["teacher_latent_path"])
        self.manifest = self.root / "metadata.jsonl"
        self.write_rows([self.row])

    def write_rows(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def dataset(self):
        return MiniMaxH3DMADDataset([self.manifest])

    def pack(self, payload):
        return pack_target_payload(payload, self.row, label="test", base_dir=self.root)

    def test_preserves_condition_contract_and_paired_collation(self):
        dataset = self.dataset()
        row = dataset[0]
        self.assertEqual(row["conditioning"]["positive"]["task"], "ref2av")
        self.assertEqual(row["meta"]["num_frames"], 107)
        self.assertEqual(row["meta"]["cache_fingerprint"], "fp-1")
        self.assertEqual(tuple(row["dmad_real"]["video"].shape), (64, 96))
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        self.assertEqual(tuple(batch["dmad_teacher"]["audio"].shape), (1, 356, 32))
        self.assertEqual(batch["dmad_real"]["audio"].sum().item(), 0)
        self.assertEqual(batch["dmad_teacher"]["audio"].sum().item(), 356 * 32)

    def test_packs_canonical_axes_and_batch_axes(self):
        video = torch.arange(24 * 32 * 2 * 4, dtype=torch.float32).reshape(1, 24, 32, 2, 4)
        audio = torch.arange(2 * 178 * 32, dtype=torch.float32).reshape(1, 2, 178, 32)
        actual = self.pack({**self.payload, "video": video, "audio": audio})
        expected_first = video[0, :, 0, :2, :2].reshape(-1)
        torch.testing.assert_close(actual["video"][0], expected_first)
        torch.testing.assert_close(actual["audio"], audio.reshape(-1, 32))
        packed = self.pack({**self.payload, "video": actual["video"].unsqueeze(0), "audio": actual["audio"].unsqueeze(0)})
        torch.testing.assert_close(packed["video"], actual["video"])
        native_audio = self.pack({**self.payload, "audio": audio.transpose(2, 3)})
        torch.testing.assert_close(native_audio["audio"], actual["audio"])

    def test_aliases_and_float_conversion(self):
        payload = {key: value for key, value in self.payload.items() if key not in ("video", "audio")}
        payload.update(video_latents=self.payload["video"].to(torch.bfloat16), audio_latents=self.payload["audio"])
        self.assertEqual(self.pack(payload)["video"].dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "both video"):
            self.pack({**payload, "video": self.payload["video"]})

    def test_rejects_condition_fingerprint_geometry_and_nonfinite(self):
        variants = [
            {**self.condition, "cache_fingerprint": "wrong"},
            {**self.condition, "conditioning": {"positive": {**self.condition["conditioning"]["positive"], "target_width": 96}}},
            {**self.condition, "conditioning": {"positive": {**self.condition["conditioning"]["positive"], "prompt_embeds": torch.tensor([float("nan")])}}},
        ]
        for condition in variants:
            with self.subTest(condition=condition["cache_fingerprint"]):
                torch.save(condition, self.row["condition_path"])
                with self.assertRaises(ValueError):
                    self.dataset()[0]

    def test_rejects_wrong_target_identity_geometry_and_normalization(self):
        for changes in ({"cache_fingerprint": "wrong"}, {"source_id": "other"}, {"target_width": 96}, {"normalized": False}, {"normalized": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.pack({**self.payload, **changes})

    def test_rejects_wrong_shapes_nonfloating_and_nonfinite(self):
        for key, value in (
            ("video", torch.zeros(63, 96)),
            ("video", torch.zeros(2, 64, 96)),
            ("audio", torch.zeros(355, 32)),
            ("audio", torch.zeros(356, 32, dtype=torch.int32)),
            ("video", torch.full((64, 96), float("nan"))),
            ("audio", torch.full((356, 32), float("inf"))),
            ("audio", torch.full((356, 32), 1e300, dtype=torch.float64)),
        ):
            with self.subTest(key=key, shape=value.shape), self.assertRaises((ValueError, TypeError)):
                self.pack({**self.payload, key: value})

    def test_rejects_missing_paths_and_condition_only_cache(self):
        row = dict(self.row)
        del row["real_latent_path"]
        self.write_rows([row])
        with self.assertRaisesRegex(ValueError, "condition-only caches"):
            self.dataset()

    def test_rejects_duplicate_across_manifests(self):
        with self.assertRaisesRegex(ValueError, "Duplicate DMAD condition"):
            MiniMaxH3DMADDataset([self.manifest, self.manifest])

    def test_manifest_digest_is_stable_for_equivalent_selected_rows(self):
        baseline = self.dataset().manifest_digest
        self.assertEqual(len(baseline), 64)
        # Resolved paths, sorted keys and canonical geometry avoid accidental
        # changes caused by moving/reformatting a manifest or using aliases.
        equivalent = {**dict(reversed(list(self.row.items()))), "condition_path": "condition.pt", "real_latent_path": "real.pt", "teacher_latent_path": "teacher.pt"}
        equivalent["num_frames"] = equivalent.pop("target_num_frames")
        self.write_rows([equivalent])
        self.assertEqual(self.dataset().manifest_digest, baseline)
        self.assertEqual(MiniMaxH3DMADDataset([self.manifest], dataset_repeat=7, max_samples=1).manifest_digest, baseline)
        self.assertEqual(MiniMaxH3DMADDataset([self.manifest], max_samples=100).manifest_digest, baseline)

    def test_manifest_digest_tracks_target_identity_paths_and_geometry(self):
        baseline = self.dataset().manifest_digest
        (self.root / "another_teacher.pt").touch()
        for changes in (
            {"cache_fingerprint": "changed"},
            {"source_id": "changed"},
            {"sample_id": "new"},
            {"teacher_latent_path": str(self.root / "another_teacher.pt")},
            {"target_width": 96},
            {"target_num_frames": 124},
        ):
            with self.subTest(changes=changes):
                self.write_rows([{**self.row, **changes}])
                self.assertNotEqual(self.dataset().manifest_digest, baseline)

    def test_manifest_digest_tracks_order_and_applies_max_samples_before_hashing(self):
        baseline = self.dataset().manifest_digest
        second = {**self.row, "cache_fingerprint": "fp-2", "source_id": "source-2", "condition_path": str(self.root / "condition2.pt")}
        (self.root / "condition2.pt").touch()
        self.write_rows([self.row, second])
        both = self.dataset().manifest_digest
        self.assertNotEqual(both, baseline)
        self.assertEqual(MiniMaxH3DMADDataset([self.manifest], max_samples=1).manifest_digest, baseline)
        self.write_rows([second, self.row])
        self.assertNotEqual(self.dataset().manifest_digest, both)
        self.assertNotEqual(MiniMaxH3DMADDataset([self.manifest], max_samples=1).manifest_digest, baseline)

    def test_reuses_cost_sampler_and_cache_component_mode(self):
        second = {**self.row, "cache_fingerprint": "fp-2", "condition_path": str(self.root / "other.pt"), "target_height": 64, "target_width": 32, "target_orientation": "portrait"}
        (self.root / "other.pt").touch()
        self.write_rows([self.row, second])
        config = {"data_path": str(self.manifest), "reference_cost_sampler": {"image_counts": [1]}, "num_workers": 0}
        module = "lightx2v_train.data.minimax_h3_dmad_dataset"
        with patch(f"{module}.get_data_parallel_world_size", return_value=2), patch(f"{module}.get_data_parallel_rank", return_value=0):
            loader = build_minimax_h3_dmad_dataset(config)
        self.assertTrue(loader.sampler.is_minimax_h3_ref_cost_sampler)
        self.assertEqual(loader.sampler.global_batch_summary(0)["orientations"], {"landscape": 1, "portrait": 1})
        self.assertTrue(is_train_cache_dataset({"data": {"train": {"name": "minimax_h3_dmad_dataset"}}}))
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            build_minimax_h3_dmad_dataset({**config, "batch_size": 2})


if __name__ == "__main__":
    unittest.main()
