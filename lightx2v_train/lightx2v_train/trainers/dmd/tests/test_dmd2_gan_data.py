"""Real GAN data identity, own-caption pairing and exact critic-stream resume."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lightx2v_train.trainers.dmd.dmd2_gan_data import DMD2GanRealStream, load_gan_real_records


def make_manifest(root, count=5):
    rows = []
    for index in range(count):
        row = {
            "task": "t2av",
            "source_id": index,
            "caption": f"real caption {index}",
            "cache_fingerprint": f"cache-{index}",
            "target_num_frames": 124,
            "target_height": 768,
            "target_width": 1344,
            "fps": 24,
            "video_latent_shape": [24, 37, 48, 84],
            "audio_latent_shape": [2, 32, 207],
            "vae_normalized": True,
            "condition_path": f"condition-{index}.pt",
            "video_latent_path": f"video-{index}.pt",
            "audio_latent_path": f"audio-{index}.pt",
        }
        (root / row["video_latent_path"]).write_bytes(b"target loading must be deferred")
        (root / row["audio_latent_path"]).write_bytes(b"target loading must be deferred")
        torch.save(
            {
                "source_id": index,
                "cache_fingerprint": row["cache_fingerprint"],
                "conditioning": {
                    "positive": {
                        "task": "t2av",
                        "prompt_embeds": torch.full((1, 2, 3), float(index)),
                        "text_token_tags": torch.ones(2, dtype=torch.long),
                        "target_num_frames": 124,
                        "target_height": 768,
                        "target_width": 1344,
                    }
                },
            },
            root / row["condition_path"],
        )
        rows.append(row)
    path = root / "metadata.jsonl"
    write_rows(path, rows)
    return path, rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class GanRealDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path, self.rows = make_manifest(self.root)

    def test_preflight_is_stdlib_only_and_never_loads_tensor_payloads(self):
        # The shared trainers package eagerly imports torch via its registry;
        # standalone launcher preflight loads this stdlib helper by file.
        script = "import runpy,sys; module=runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules; assert len(module['load_gan_real_records'](sys.argv[2])[1]) == 5"
        module_path = Path(__file__).parents[1] / "dmd2_gan_data.py"
        subprocess.run([sys.executable, "-c", script, str(module_path), str(self.path)], check=True)
        with patch("torch.load", side_effect=AssertionError("no payload reads in preflight")):
            manifest, rows, fingerprint = load_gan_real_records(self.root)
            stream = DMD2GanRealStream(self.path, seed=123)
        self.assertEqual(manifest, self.path)
        self.assertEqual(stream.recipe()["metadata_fingerprint"], fingerprint)
        self.assertTrue(all(Path(row[key]).is_absolute() for row in rows for key in ("condition_path", "video_latent_path", "audio_latent_path")))

    def test_filters_native_geometry_portrait_other_task_and_wrong_frame_rate(self):
        for index, updates in enumerate(({"target_height": 704, "target_width": 1280}, {"target_height": 1344, "target_width": 768}, {"task": "ref2av"}, {"fps": 30})):
            self.rows[index].update(updates)
            self.rows[index]["video_latent_path"] = "excluded_missing_file.pt"
        write_rows(self.path, self.rows)
        rows = load_gan_real_records(self.path)[1]
        self.assertEqual([row["source_id"] for row in rows], [4])
        self.rows[4]["target_num_frames"] = 125
        write_rows(self.path, self.rows)
        with self.assertRaisesRegex(ValueError, "no eligible"):
            load_gan_real_records(self.path)

    def test_eligible_rows_must_have_full_av_caption_fps_and_normalized_geometry(self):
        for field, value, message in (
            ("video_latent_path", "", "video_latent_path"),
            ("audio_latent_path", None, "audio_latent_path"),
            ("condition_path", "", "condition_path"),
            ("caption", "", "caption"),
            ("vae_normalized", False, "normalized"),
            ("video_latent_shape", [24, 1, 48, 84], "geometry"),
            ("audio_latent_shape", [2, 32, 1], "geometry"),
            ("references", [{"kind": "image"}], "references"),
        ):
            with self.subTest(field=field):
                rows = [dict(self.rows[0], **{field: value})]
                write_rows(self.path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    load_gan_real_records(self.path)
        row = dict(self.rows[0])
        del row["fps"]
        write_rows(self.path, [row])
        with self.assertRaisesRegex(ValueError, "explicitly declare fps"):
            load_gan_real_records(self.path)
        write_rows(self.path, [dict(self.rows[0], target_num_frames=124.5)])
        with self.assertRaisesRegex(ValueError, "target geometry"):
            load_gan_real_records(self.path)

    def test_missing_file_fails_and_metadata_changes_change_recipe_identity(self):
        initial = load_gan_real_records(self.path)[2]
        self.assertEqual(initial, load_gan_real_records(self.path, check_files=False)[2])
        self.rows[0]["caption"] += " changed"
        write_rows(self.path, self.rows)
        self.assertNotEqual(initial, load_gan_real_records(self.path)[2])
        original = load_gan_real_records(self.path)[2]
        self.rows[0]["condition_path"] = str(self.root / "alternate.pt")
        write_rows(self.path, self.rows)
        self.assertNotEqual(original, load_gan_real_records(self.path, check_files=False)[2])
        with self.assertRaisesRegex(FileNotFoundError, "condition_path"):
            load_gan_real_records(self.path)

    def test_all_ranks_partition_permutations_and_resume_exactly_across_epochs(self):
        streams = [DMD2GanRealStream(self.path, seed=879, rank=rank, world_size=3) for rank in range(3)]
        global_sequence = [stream.index_at(cursor) for cursor in range(20) for stream in streams]
        for offset in range(0, len(global_sequence), len(self.rows)):
            self.assertEqual(sorted(global_sequence[offset : offset + len(self.rows)]), list(range(5)))
        uninterrupted = [streams[1].index_at(cursor) for cursor in range(100)]
        resumed = DMD2GanRealStream(self.path, seed=879, rank=1, world_size=3)
        self.assertEqual(uninterrupted[7:], [resumed.index_at(cursor) for cursor in range(7, 100)])
        # An arbitrary old cursor remains reproducible after memoized epochs expire.
        self.assertEqual(uninterrupted[0], resumed.index_at(0))
        self.assertLessEqual(len(resumed._permutations), 2)
        changed_seed = DMD2GanRealStream(self.path, seed=880, rank=1, world_size=3)
        self.assertNotEqual(uninterrupted, [changed_seed.index_at(cursor) for cursor in range(100)])

    def test_world_size_larger_than_dataset_still_has_exact_resume(self):
        stream = DMD2GanRealStream(self.path, seed=8, rank=9, world_size=32)
        expected = [stream.index_at(cursor) for cursor in range(16)]
        resumed = DMD2GanRealStream(self.path, seed=8, rank=9, world_size=32)
        self.assertEqual(expected[3:], [resumed.index_at(cursor) for cursor in range(3, 16)])
        for cursor in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                stream.index_at(cursor)
        with self.assertRaises(ValueError):
            DMD2GanRealStream(self.path, seed=0, rank=32, world_size=32)

    def test_consumption_loads_only_own_condition_and_preserves_deferred_target_paths(self):
        stream = DMD2GanRealStream(self.path, seed=879)
        index = stream.index_at(12)
        with patch("torch.load", wraps=torch.load) as loader:
            sample = stream.sample(12)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(Path(loader.call_args.args[0]), self.root / self.rows[index]["condition_path"])
        self.assertEqual(sample["inputs"], {})
        self.assertEqual(sample["conditioning"]["prompt"], f"real caption {index}")
        self.assertTrue(torch.equal(sample["conditioning"]["positive"]["prompt_embeds"], torch.full((2, 3), float(index))))
        self.assertEqual(sample["meta"]["video_latent_path"], str(self.root / self.rows[index]["video_latent_path"]))
        self.assertTrue(stream._dataset.defer_latent_loading)

    def test_runtime_rejects_wrong_pair_and_negative_or_reference_conditions(self):
        for update, message in (({"cache_fingerprint": "wrong"}, "fingerprint"), ({"source_id": -1}, "source_id"), ({"negative": {"prompt_embeds": torch.zeros(2, 3)}}, "negative")):
            with self.subTest(update=tuple(update)):
                path, _ = make_manifest(self.root, count=1)
                condition_path = self.root / "condition-0.pt"
                payload = torch.load(condition_path, weights_only=True)
                payload.update(update)
                torch.save(payload, condition_path)
                with self.assertRaisesRegex(ValueError, message):
                    DMD2GanRealStream(path, seed=0).sample(0)
        for update, message in (
            ({"task": "fl2av"}, "T2AV"),
            ({"references": [{"kind": "image"}]}, "references"),
            ({"keyframe_anchors": [0]}, "keyframe"),
            ({"target_height": 704}, "target_height"),
            ({"text_token_tags": torch.tensor([0, 1])}, "non-text"),
        ):
            with self.subTest(update=tuple(update)):
                path, _ = make_manifest(self.root, count=1)
                condition_path = self.root / "condition-0.pt"
                payload = torch.load(condition_path, weights_only=True)
                payload["conditioning"]["positive"].update(update)
                torch.save(payload, condition_path)
                with self.assertRaisesRegex(ValueError, message):
                    DMD2GanRealStream(path, seed=0).sample(0)

    def test_load_uses_capability_with_own_caption_and_matching_shape(self):
        stream = DMD2GanRealStream(self.path, seed=44)
        owner = self

        class Capability:
            def latent_shape(self, sample, generation_shapes, broadcast):
                owner.assertEqual(generation_shapes, [[124, 768, 1344]])
                return broadcast("shape")

            def encode_conditions(self, sample, negative_prompt, guidance_scale, broadcast):
                owner.assertEqual((negative_prompt, guidance_scale), ("", 1.0))
                self.caption = sample["conditioning"]["prompt"]
                return broadcast(sample["conditioning"]["positive"]), None

            def extract_turbo_latents(self, sample, shape):
                owner.assertEqual(shape, "shape")
                owner.assertEqual(sample["inputs"], {})
                return (sample["meta"]["video_latent_path"], sample["meta"]["audio_latent_path"])

        capability = Capability()
        clean, condition = stream.load(capability, 6)
        index = stream.index_at(6)
        self.assertEqual(capability.caption, f"real caption {index}")
        self.assertEqual(clean[0], str(self.root / f"video-{index}.pt"))
        self.assertEqual(condition["task"], "t2av")
        with patch.object(capability, "encode_conditions", return_value=({"task": "t2av"}, {"unexpected": True})):
            with self.assertRaisesRegex(ValueError, "negative"):
                stream.load(capability, 6)


if __name__ == "__main__":
    unittest.main()
