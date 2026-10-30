"""Portable CM paths, full-target identity, and exact independent data resume."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from lightx2v_train.data.minimax_h3_turbo_cm_dataset import MiniMaxH3TurboCMDataset, MiniMaxH3TurboCMSampler, build_minimax_h3_turbo_cm_dataset
from lightx2v_train.data.minimax_h3_turbo_paths import load_cm_records, normalize_path_mapping, resolve_cache_path
from lightx2v_train.utils.registry import build_data
from lightx2v_train.utils.utils import is_train_cache_dataset


class _IndexDataset:
    metadata_fingerprint = "immutable-fixture"

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count


def make_manifest(root, count=3):
    cache, conditions = root / "cache", root / "conditions"
    (cache / "video_latents").mkdir(parents=True)
    (cache / "audio_latents").mkdir()
    conditions.mkdir()
    rows = []
    for index in range(count):
        height, width = (768, 1344) if index % 2 == 0 else (1344, 768)
        fingerprint = str(index).zfill(64)
        row = {
            "source_id": index,
            "task": "t2av",
            "caption": f"source {index}",
            "cache_fingerprint": fingerprint,
            "target_height": height,
            "target_width": width,
            "num_frames": 124,
            "vae_normalized": True,
            "video_latent_shape": [24, 37, height // 16, width // 16],
            "audio_latent_shape": [2, 32, 207],
            "condition_path": f"/old/conditions/{index}.pt",
            "video_latent_path": f"video_latents/{index}.pt",
            "audio_latent_path": f"audio_latents/{index}.pt",
            "video_path": f"/old/raw_video/{index}.mp4",
        }
        # Invalid tensor bytes prove defer_latent_loading never reads target
        # tensors in DataLoader workers; the trainer owns their loading.
        (cache / row["video_latent_path"]).write_bytes(b"deferred video")
        (cache / row["audio_latent_path"]).write_bytes(b"deferred audio")
        torch.save(
            {
                "source_id": index,
                "task": "t2av",
                "cache_fingerprint": fingerprint,
                "conditioning": {"positive": {"task": "t2av", "prompt_embeds": torch.zeros(1, 2, 4), "text_token_tags": torch.ones(2, dtype=torch.long)}},
            },
            conditions / f"{index}.pt",
        )
        rows.append(row)
    path = cache / "metadata.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows, {"/old/conditions": str(conditions)}


def collect_epochs(sampler, count):
    result = []
    for epoch in range(count):
        sampler.set_epoch(epoch)
        result.extend(sampler)
    return result


def make_native_manifest(root, task):
    path, rows, mapping = make_manifest(root)
    for index, (row, geometry) in enumerate(zip(rows, ((704, 1280), (1280, 704), (1024, 1024)), strict=True)):
        height, width = geometry
        row.update(task=task, target_height=height, target_width=width, target_num_frames=124, video_latent_shape=[24, 37, height // 16, width // 16])
        payload_path = Path(mapping["/old/conditions"]) / f"{index}.pt"
        payload = torch.load(payload_path, weights_only=True)
        payload.update(task=task, cache_stage="complete")
        positive = payload["conditioning"]["positive"]
        positive.update(task=task, target_height=height, target_width=width, target_num_frames=124)
        if task == "ref2av":
            frames = 1 if index == 0 else 3
            reference = {
                "kind": "image" if index == 0 else "video",
                "normalized": True,
                "video_latents": torch.zeros(frames * 2, 96),
                "num_latent_frames": frames,
                "latent_height": 2,
                "latent_width": 4,
            }
            if index == 2:
                reference.update(audio_latents=torch.zeros(4, 32), num_audio_latents=2)
            positive.update(text_token_tags=torch.tensor([0, 1]), references=[reference])
            row.update(reference_image_count=int(index == 0), reference_video_count=int(index != 0), reference_audio_count=0)
        torch.save(payload, payload_path)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows, mapping


class TurboCMPathsAndDatasetTests(unittest.TestCase):
    def test_cm_and_curriculum_cached_training_do_not_require_gpu_encoders(self):
        for name in ("minimax_h3_turbo_cm_dataset", "minimax_h3_curriculum_cache_dataset"):
            self.assertTrue(is_train_cache_dataset({"data": {"train": {"name": name}}}))
        for name in ("video_dataset", "prompt_dataset"):
            self.assertFalse(is_train_cache_dataset({"data": {"train": {"name": name}}}))

    def test_longest_component_prefix_and_relative_latents(self):
        mappings = {"/old": "/new", "/old/conditions": "/special"}
        self.assertEqual(normalize_path_mapping(mappings)[0], ("/old/conditions", "/special"))
        self.assertEqual(resolve_cache_path("/old/conditions/a.pt", "/root/cache/metadata.jsonl", mappings), Path("/special/a.pt"))
        self.assertEqual(resolve_cache_path("/older/a.pt", "/root/cache/metadata.jsonl", mappings), Path("/older/a.pt"))
        self.assertEqual(resolve_cache_path("video/a.pt", "/old/cache/metadata.jsonl", mappings), Path("/old/cache/video/a.pt"))
        with self.assertRaisesRegex(ValueError, "absolute"):
            normalize_path_mapping({"relative": "/new"})

    def test_mapped_deferred_full_target_loader_preserves_identity_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows, mapping = make_manifest(Path(directory))
            original = path.read_bytes()
            _, mapped, fingerprint = load_cm_records(path, mapping)
            self.assertEqual(mapped[0]["video_path"], rows[0]["video_path"])
            self.assertEqual(mapped[0]["video_latent_path"], str(path.parent / rows[0]["video_latent_path"]))
            loader = build_data({"data": {"cm": {"name": "minimax_h3_turbo_cm_dataset", "data_path": str(path), "path_mapping": mapping, "tasks": ["t2av"], "num_workers": 0}}}, "cm")
            loader.sampler.configure_from_microbatch_offset(0, 1, 4, unused_role_metadata="accepted")
            sample = next(iter(loader))
            self.assertEqual(sample["inputs"], {})
            self.assertIn("video_latent_path", sample["meta"])
            self.assertIn("audio_latent_path", sample["meta"])
            self.assertEqual(loader.sampler.checkpoint_metadata()["metadata_fingerprint"], fingerprint)
            self.assertEqual(loader.sampler.cursor_microbatch_loss_scale(100), 1.0)
            self.assertEqual(path.read_bytes(), original)

    def test_metadata_and_mapping_change_resume_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows, mapping = make_manifest(Path(directory))
            first = load_cm_records(path, mapping)[2]
            self.assertEqual(first, load_cm_records(path, mapping, check_files=False)[2])
            rows[0]["caption"] += " changed"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertNotEqual(first, load_cm_records(path, mapping)[2])
            self.assertNotEqual(load_cm_records(path, mapping)[2], load_cm_records(path, {"/old/conditions": "/different"}, check_files=False)[2])

    def test_condition_fingerprint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, mapping = make_manifest(Path(directory))
            payload_path = Path(mapping["/old/conditions"]) / "0.pt"
            payload = torch.load(payload_path, weights_only=True)
            payload["cache_fingerprint"] = "wrong"
            torch.save(payload, payload_path)
            dataset = MiniMaxH3TurboCMDataset(path, path_mapping=mapping)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                dataset[0]

    def test_missing_targets_wrong_geometry_and_duplicate_identity_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows, mapping = make_manifest(Path(directory))
            (path.parent / rows[0]["audio_latent_path"]).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "audio_latent_path"):
                load_cm_records(path, mapping)
            rows[0]["video_latent_shape"] = [24, 1, 48, 84]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "full target geometry"):
                load_cm_records(path, mapping, check_files=False)
            rows[0]["video_latent_shape"] = [24, 37, 48, 84]
            rows[1]["source_id"] = rows[0]["source_id"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cm_records(path, mapping, check_files=False)

    def test_builder_rejects_dropping_or_non_singleton_batches(self):
        for config in ({"batch_size": 2}, {"drop_last": True}, {"dataset_repeat": 2}):
            with self.assertRaises(ValueError):
                build_minimax_h3_turbo_cm_dataset({"data_path": "/unused", **config})

    def test_native_t2av_preserves_every_geometry_and_default_stays_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows, mapping = make_native_manifest(Path(directory), "t2av")
            with self.assertRaisesRegex(ValueError, "native_geometry=true"):
                load_cm_records(path, mapping)
            _, accepted, _ = load_cm_records(path, mapping, native_geometry=True)
            self.assertEqual([(row["target_height"], row["target_width"]) for row in accepted], [(704, 1280), (1280, 704), (1024, 1024)])
            dataset = MiniMaxH3TurboCMDataset(path, path_mapping=mapping, native_geometry=True)
            self.assertEqual(len(dataset), len(rows))
            for index in range(len(dataset)):
                self.assertEqual(dataset[index]["meta"]["target_height"], rows[index]["target_height"])
            rows[0]["target_height"] = 705
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "32-aligned"):
                load_cm_records(path, mapping, native_geometry=True)

    def test_native_ref_image_video_and_av_conditions_are_all_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows, mapping = make_native_manifest(Path(directory), "ref2av")
            with self.assertRaisesRegex(ValueError, "task=t2av"):
                load_cm_records(path, mapping, native_geometry=True)
            loader = build_data(
                {
                    "data": {
                        "train": {
                            "name": "minimax_h3_turbo_cm_dataset",
                            "data_path": str(path),
                            "path_mapping": mapping,
                            "tasks": ["ref2av"],
                            "native_geometry": True,
                            "num_workers": 0,
                            "shuffle": False,
                        }
                    }
                },
                "train",
            )
            self.assertEqual(len(loader.dataset), len(rows))
            loader.sampler.configure_from_microbatch_offset(0)
            samples = list(loader)
            self.assertEqual([sample["conditioning"]["positive"]["references"][0]["kind"][0] for sample in samples], ["image", "video", "video"])
            self.assertIn("audio_latents", samples[2]["conditioning"]["positive"]["references"][0])
            self.assertEqual([sample["meta"]["source_id"].item() for sample in samples], [0, 1, 2])
            self.assertTrue(all(sample["inputs"] == {} for sample in samples))

    def test_ref_conditions_reject_wrong_geometry_reference_rows_and_count(self):
        for corruption in ("target_geometry", "reference_rows", "reference_count", "cache_stage"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                path, rows, mapping = make_native_manifest(Path(directory), "ref2av")
                condition = Path(mapping["/old/conditions"]) / "0.pt"
                payload = torch.load(condition, weights_only=True)
                if corruption == "target_geometry":
                    payload["conditioning"]["positive"]["target_height"] = 1024
                elif corruption == "reference_rows":
                    payload["conditioning"]["positive"]["references"][0]["video_latents"] = torch.zeros(3, 96)
                elif corruption == "reference_count":
                    rows[0]["reference_image_count"] = 0
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                else:
                    payload["cache_stage"] = "text"
                torch.save(payload, condition)
                dataset = MiniMaxH3TurboCMDataset(path, path_mapping=mapping, tasks=["ref2av"], native_geometry=True)
                with self.assertRaises(ValueError):
                    dataset[0]

    def test_one_row_reference_is_validated_before_singleton_batch_stripping(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, mapping = make_native_manifest(Path(directory), "ref2av")
            condition = Path(mapping["/old/conditions"]) / "0.pt"
            payload = torch.load(condition, weights_only=True)
            reference = payload["conditioning"]["positive"]["references"][0]
            reference.update(video_latents=torch.zeros(1, 96), latent_width=2)
            torch.save(payload, condition)
            dataset = MiniMaxH3TurboCMDataset(path, path_mapping=mapping, tasks=["ref2av"], native_geometry=True)
            self.assertEqual(dataset[0]["conditioning"]["positive"]["references"][0]["kind"], "image")

    def test_legacy_fingerprint_stays_compatible_and_native_policy_is_bound(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path, _, mapping = make_manifest(Path(directory))
            _, resolved, fingerprint = load_cm_records(path, mapping)
            old_identity = {
                "schema": "h3_turbo_cm_full_av_v1",
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "path_mapping": normalize_path_mapping(mapping),
                "resolved_cache_paths": [[row[key] for key in ("condition_path", "video_latent_path", "audio_latent_path")] for row in resolved],
            }
            self.assertEqual(fingerprint, hashlib.sha256(json.dumps(old_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            self.assertNotEqual(fingerprint, load_cm_records(path, mapping, native_geometry=True)[2])


class TurboCMSamplerTests(unittest.TestCase):
    def sampler(self, count=11, rank=0, world=4, seed=42):
        sampler = MiniMaxH3TurboCMSampler(_IndexDataset(count), num_replicas=world, rank=rank, seed=seed)
        sampler.configure_from_microbatch_offset(0, 1, 4)
        return sampler

    def test_exact_9424_examples_are_not_dropped_or_padded_at_32_cards(self):
        samplers = [self.sampler(9424, rank, 32) for rank in range(32)]
        self.assertTrue(all(len(sampler) == 295 for sampler in samplers))
        streams = [list(sampler) for sampler in samplers]
        global_stream = [streams[rank][microbatch] for microbatch in range(295) for rank in range(32)]
        self.assertEqual(sorted(global_stream[:9424]), list(range(9424)))
        next_permutation = torch.randperm(9424, generator=torch.Generator().manual_seed(43)).tolist()
        self.assertEqual(global_stream[9424:], next_permutation[:16])
        self.assertEqual(sum(index < 4820 for index in global_stream[:9424]), 4820)

    def test_resume_matches_uninterrupted_suffix_at_arbitrary_microbatches(self):
        for rank in range(4):
            full = collect_epochs(self.sampler(rank=rank), 20)
            for offset in (0, 1, 2, 3, 4, 11, 17):
                resumed = self.sampler(rank=rank)
                resumed.configure_from_microbatch_offset(offset, 3, 4)
                self.assertEqual(collect_epochs(resumed, 3), full[offset : offset + 3 * len(resumed)])

    def test_shuffle_is_deterministic_and_fingerprint_metadata_is_static(self):
        first, repeat, other = self.sampler(), self.sampler(), self.sampler(seed=43)
        self.assertEqual(collect_epochs(first, 3), collect_epochs(repeat, 3))
        self.assertNotEqual(collect_epochs(self.sampler(), 3), collect_epochs(other, 3))
        metadata = first.checkpoint_metadata()
        first.configure_from_microbatch_offset(123, 1, 4)
        first.set_epoch(7)
        self.assertEqual(metadata, first.checkpoint_metadata())
        self.assertEqual(metadata["remainder_policy"], "continue_next_shuffled_data_epoch")


if __name__ == "__main__":
    unittest.main()
