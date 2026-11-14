"""CPU contracts for importing server-side H3 trajectories without copying them."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("_teacher_import", Path(__file__).with_name("import_lightx2v_ref_teacher_latents.py"))
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
HAS_TORCH = importlib.util.find_spec("torch") is not None


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ExportFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw_path = self.root / "raw.jsonl"
        self.index_path = self.root / "index.node0.jsonl"
        self.output = self.root / "published"
        self.directory = self.root / "server" / "sample-0"
        self.directory.mkdir(parents=True)
        self.images = [str(self.root / f"image-{number}.png") for number in range(2)]
        self.raw = dict(sample_id="sample-0", enhanced_prompt="A speaker beside <Picture 2>", reference_images=self.images)
        self.index = dict(
            source_jsonl=str(self.raw_path),
            source_index=0,
            source_id="sample-0",
            prompt=self.raw["enhanced_prompt"],
            prompt_source="enhanced_prompt",
            reference_images=self.images,
            ref_image_count=2,
            target_height=64,
            target_width=96,
            num_frames=124,
            fps=24,
            seed=42,
            cache_fingerprint="fingerprint-0",
            export_dir=str(self.directory),
        )
        self.manifest = dict(
            schema=helper.SCHEMA,
            task="ref2av",
            normalized=True,
            dtype="float32",
            num_steps=29,
            num_states=30,
            state_paths=[f"state_{number:03d}.pt" for number in range(30)],
            condition_path="condition.pt",
            video_latents_path="video_latents.pt",
            audio_latents_path="audio_latents.pt",
            video_shape=[24, 37, 4, 6],
            audio_shape=[2, 32, 207],
            num_frames=124,
            height=64,
            width=96,
            fps=24,
            seed=42,
            prompt=self.raw["enhanced_prompt"],
            image_paths=self.images,
            video_paths=[],
            audio_paths=[],
            scheduler_type="reference_blend",
            video_sigmas=[1 - number / 29 for number in range(30)],
            audio_sigmas=[1 - number / 29 for number in range(30)],
            video_flow_shift=12,
            audio_flow_shift=3,
            model_path="/models/official-decfg",
            transformer_partition="transformer_ref",
            transformer_path="/models/official-decfg/transformer_ref",
            model_variant="ref2av",
            source_id="sample-0",
            cache_fingerprint="fingerprint-0",
            prompt_token_count=3,
            reference_video_rows=12,
            reference_audio_rows=0,
            ref_image_count=2,
            reference_video_count=0,
            reference_audio_count=0,
            references=[dict(kind="image", normalized=True, path=path, num_latent_frames=1, latent_height=4, latent_width=6, num_audio_latents=0) for path in self.images],
            model_identity={"sha256": "model-content-identity"},
            decfg_merged_identity=[{"sha256": "decfg-receipt"}],
            guidance_scale=1.0,
        )
        for name in [*self.manifest["state_paths"], "condition.pt", "video_latents.pt", "audio_latents.pt"]:
            (self.directory / name).write_bytes(b"opaque; never deserialize in metadata-only import")
        self.save()

    def save(self):
        write_jsonl(self.raw_path, [self.raw])
        write_jsonl(self.index_path, [self.index])
        write_json(self.directory / "manifest.json", self.manifest)

    def run_import(self, **kwargs):
        return helper.import_exports(self.raw_path, [self.index_path], self.output, media_root=self.root, **kwargs)

    def rows(self, name="metadata"):
        return [row for _, row in helper.jsonl(self.output / f"{name}.jsonl")]


class ImportTests(ExportFixture):
    def test_manifest_only_import_uses_existing_gan_loader_and_receipt_contract(self):
        self.run_import(expected_count=1)
        row = self.rows()[0]
        self.assertEqual(row["caption"], self.raw["enhanced_prompt"])
        self.assertEqual(row["packed_sequence_tokens_124"], 3 + 12 + 37 * 2 * 3 + 414)
        self.assertEqual(row["reference_compute_cost"], 15)
        self.assertEqual(row["video_latent_path"], str(self.directory / "video_latents.pt"))
        self.assertNotIn("state_", row["video_latent_path"])
        self.assertEqual(row["teacher_trajectory_states"], 30)
        self.assertNotIn("video_latent_path", self.rows("condition_metadata")[0])
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["condition_metadata.complete.json", "condition_metadata.jsonl", "metadata.complete.json", "metadata.jsonl"])
        gan = helper._leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_test_import_gan")
        manifest, records, fingerprint = gan.load_gan_real_records(self.output, task="ref2av", geometry=None)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(fingerprint), 64)
        receipt_api = helper._leaf("scripts/minimax_h3_cm_pdmd_preflight.py", "_test_import_receipt")
        receipt_api._receipt(manifest, 1)
        receipt_api._receipt(self.output / "condition_metadata.jsonl", 1)
        receipt = json.loads((self.output / "metadata.complete.json").read_text())
        self.assertFalse(receipt["trajectory_tensors_loaded"])
        self.assertEqual(receipt["raw_source_rows"], 1)

    def test_cli_import_requires_no_torch(self):
        script = "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__'); assert 'torch' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(helper.__file__)), "--raw-jsonl", str(self.raw_path), "--index", str(self.index_path), "--output-dir", str(self.output), "--expected-count", "1"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["completed_count"], 1)

    def test_rejects_broken_finalized_export_and_never_publishes(self):
        mutations = [
            ("schema", "h3_teacher_trajectory_v2"),
            ("normalized", False),
            ("num_states", 29),
            ("scheduler_type", "euler"),
            ("state_paths", self.manifest["state_paths"][:-1]),
            ("video_shape", [24, 36, 4, 6]),
            ("audio_shape", [2, 32, 206]),
            ("video_sigmas", [0.0] * 30),
            ("audio_sigmas", [float("nan")] * 30),
            ("video_sigmas", list(reversed(self.manifest["video_sigmas"]))),
            ("audio_sigmas", [1 - number / 30 for number in range(30)]),
            ("source_id", "wrong"),
            ("cache_fingerprint", "wrong"),
            ("prompt", "wrong"),
            ("seed", 41),
            ("image_paths", list(reversed(self.images))),
            ("ref_image_count", 3),
            ("reference_audio_count", 1),
            ("reference_audio_rows", 1),
            ("prompt_token_count", 0),
            ("reference_video_rows", True),
            ("condition_path", "../condition.pt"),
            ("model_variant", ""),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                broken = {**self.manifest, key: value}
                write_json(self.directory / "manifest.json", broken)
                with self.assertRaises(ValueError):
                    self.run_import()
                self.assertFalse(self.output.exists())

    def test_missing_or_empty_state_blocks_publication(self):
        path = self.directory / "state_016.pt"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            self.run_import()
        path.touch()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            self.run_import()

    def test_missing_final_manifest_is_not_a_completed_export(self):
        (self.directory / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "manifest.json"):
            self.run_import()

    def test_duplicate_node_output_or_source_identity_rejected(self):
        second = self.root / "index.node1.jsonl"
        write_jsonl(second, [self.index])
        with self.assertRaisesRegex(ValueError, "Duplicate source_id"):
            helper.import_exports(self.raw_path, [self.index_path, second], self.output)
        self.assertFalse(self.output.exists())

    def test_raw_caption_and_reference_order_must_match_exactly(self):
        for updates in ({"enhanced_prompt": "changed caption"}, {"reference_images": self.images[::-1]}, {"sample_id": "different"}):
            write_jsonl(self.raw_path, [{**self.raw, **updates}])
            with self.assertRaises(ValueError):
                self.run_import()

    def test_preserves_reference_numbering_and_uses_original_parser_aliases(self):
        self.raw.pop("reference_images")
        self.raw["references"] = [dict(kind="reference_image", path=self.images[0], order=1, valid=False), dict(kind="image_url", path=self.images[1], order=2)]
        self.raw["valid_reference_indices"] = [1]
        self.raw["reference_videos"] = ["unused fallback entry"]
        self.raw["reference_audios"] = ["unused fallback entry"]
        self.save()
        self.run_import()
        self.assertEqual([reference["path"] for reference in self.rows()[0]["references"]], self.images)
        self.raw["references"].reverse()
        self.save()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.run_import()

    def test_source_index_ignores_blank_lines_like_original_cache_builder(self):
        self.save()
        self.raw_path.write_text("\n\n" + json.dumps(self.raw) + "\n")
        self.run_import()
        self.assertEqual(self.rows()[0]["source_index"], 0)

    def test_expected_count_gate_preserves_previous_valid_publication(self):
        self.run_import(expected_count=1)
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, "Expected 5000"):
            self.run_import(expected_count=5000)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_partial_import_receipt_reports_actual_subset_size(self):
        write_jsonl(self.raw_path, [self.raw, {**self.raw, "sample_id": "unselected"}])
        receipt = self.run_import()
        self.assertEqual((receipt["completed_count"], receipt["input_total_rows"], receipt["raw_source_rows"], receipt["raw_unselected_rows"]), (1, 1, 2, 1))
        self.assertIsNone(receipt["expected_count"])

    def test_models_or_schedule_settings_cannot_be_mixed(self):
        other_directory = self.root / "server" / "sample-1"
        other_directory.mkdir()
        for path in self.directory.iterdir():
            (other_directory / path.name).write_bytes(path.read_bytes())
        other_raw = {**self.raw, "sample_id": "sample-1"}
        other_index = {**self.index, "source_index": 1, "source_id": "sample-1", "cache_fingerprint": "fingerprint-1", "export_dir": str(other_directory)}
        other_manifest = {**self.manifest, "source_id": "sample-1", "cache_fingerprint": "fingerprint-1"}
        write_jsonl(self.raw_path, [self.raw, other_raw])
        write_jsonl(self.index_path, [self.index, other_index])
        for changes in ({"model_identity": {"sha256": "different"}}, {"decfg_merged_identity": []}, {"guidance_scale": 2.0}, {"video_sigmas": [value**2 for value in self.manifest["video_sigmas"]]}):
            write_json(other_directory / "manifest.json", {**other_manifest, **changes})
            with self.assertRaisesRegex(ValueError, "Inconsistent teacher"):
                self.run_import()
        write_json(other_directory / "manifest.json", other_manifest)
        self.run_import(expected_count=2)
        self.assertEqual(len(self.rows()), 2)


@unittest.skipUnless(HAS_TORCH, "Optional CPU tensor checks require torch")
class TensorImportTests(ExportFixture):
    def setUp(self):
        super().setUp()
        import torch

        self.torch = torch
        self.condition = dict(
            cache_schema_version=1,
            cache_stage="complete",
            source_id="sample-0",
            cache_fingerprint="fingerprint-0",
            prompt=self.raw["enhanced_prompt"],
            conditioning=dict(
                positive=dict(
                    task="ref2av",
                    target_num_frames=124,
                    target_height=64,
                    target_width=96,
                    prompt_embeds=torch.ones(1, 3, 5120),
                    text_token_tags=torch.tensor([0, 1, 1]),
                    references=[
                        dict(kind="image", path=path, normalized=True, num_latent_frames=1, latent_height=4, latent_width=6, video_latents=torch.ones(6, 96), audio_latents=None, num_audio_latents=0)
                        for path in self.images
                    ],
                )
            ),
        )
        torch.save(self.condition, self.directory / "condition.pt")
        torch.save(dict(normalized=True, latents=torch.ones(*self.manifest["video_shape"])), self.directory / "video_latents.pt")
        torch.save(torch.ones(*self.manifest["audio_shape"]), self.directory / "audio_latents.pt")

    def test_check_tensors_and_existing_deferred_gan_dataset(self):
        self.run_import(check_tensors=True)
        sys.path.insert(0, str(helper.TRAIN_ROOT))
        self.addCleanup(sys.path.remove, str(helper.TRAIN_ROOT))
        gan = helper._leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_test_import_deferred_gan")
        manifest, rows, _ = gan.load_gan_real_records(self.output, task="ref2av", geometry=None)
        dataset = gan._deferred_dataset(manifest, rows, task="ref2av")
        sample = dataset[0]
        self.assertEqual(sample["conditioning"]["positive"]["prompt_embeds"].ndim, 2)
        self.assertEqual(sample["conditioning"]["positive"]["text_token_tags"].shape, (3,))
        self.assertEqual(sample["conditioning"]["prompt"], self.raw["enhanced_prompt"])
        self.assertEqual(sample["meta"]["video_latent_path"], str(self.directory / "video_latents.pt"))
        # State files deliberately contain opaque bytes. This validates that
        # neither optional import validation nor GAN loading reads trajectories.
        stream = gan.DMD2GanRealStream(self.output, task="ref2av", geometry=None, seed=42)
        self.assertEqual(stream.recipe()["geometries"], [[124, 64, 96]])

    def test_check_tensors_rejects_wrong_condition_identity_caption_order_and_finiteness(self):
        for key, value in (("source_id", "wrong"), ("cache_fingerprint", "wrong"), ("prompt", "wrong")):
            self.torch.save({**self.condition, key: value}, self.directory / "condition.pt")
            with self.assertRaises(ValueError):
                self.run_import(check_tensors=True)
        condition = copy.deepcopy(self.condition)
        condition["conditioning"]["positive"]["references"].reverse()
        self.torch.save(condition, self.directory / "condition.pt")
        with self.assertRaisesRegex(ValueError, "path/order"):
            self.run_import(check_tensors=True)
        self.torch.save(self.condition, self.directory / "condition.pt")
        audio = self.torch.ones(*self.manifest["audio_shape"])
        audio[0, 0, 0] = float("nan")
        self.torch.save(audio, self.directory / "audio_latents.pt")
        with self.assertRaisesRegex(ValueError, "Nonfinite final audio"):
            self.run_import(check_tensors=True)


if __name__ == "__main__":
    unittest.main()
