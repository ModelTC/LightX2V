"""Metadata-only end-to-end checks for the two-node Ref6k data publication."""

import copy
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("ref6k_prepare", Path(__file__).with_name("minimax_h3_ref6k_prepare.py"))
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


class Ref6kTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.exports = self.root / "exports"
        self.raw = self.root / "raw.jsonl"
        self.output = self.root / "mixed"
        self.selfgen = self.root / "selfgen/metadata.jsonl"
        raw, indexes = [], [[], []]
        self.manifests = []
        for count in range(2, 7):
            for node in (0, 1):
                source_id = f"teacher-{count}-{node}"
                directory = self.exports / source_id
                directory.mkdir(parents=True)
                images = [str(self.root / f"image-{number}.png") for number in range(count)]
                source = dict(sample_id=source_id, enhanced_prompt=f"own prompt {source_id}", reference_images=images)
                index = dict(
                    source_jsonl=str(self.raw),
                    source_index=len(raw),
                    source_id=source_id,
                    prompt=source["enhanced_prompt"],
                    prompt_source="enhanced_prompt",
                    reference_images=images,
                    ref_image_count=count,
                    target_height=768,
                    target_width=1344,
                    num_frames=124,
                    fps=24,
                    seed=42,
                    cache_fingerprint=source_id,
                    export_dir=str(directory),
                    node_index=node,
                    selection_sha256="shared-selection",
                    request_fingerprint=f"request-{source_id}",
                )
                manifest = dict(
                    schema="h3_teacher_final_v1",
                    export_mode="final",
                    task="ref2av",
                    normalized=True,
                    dtype="float32",
                    num_steps=29,
                    num_states=1,
                    state_paths=[],
                    final_state_index=29,
                    final_video_sigma=0.0,
                    final_audio_sigma=0.0,
                    condition_path="condition.pt",
                    video_latents_path="video_latents.pt",
                    audio_latents_path="audio_latents.pt",
                    video_shape=[24, 37, 48, 84],
                    audio_shape=[2, 32, 207],
                    num_frames=124,
                    height=768,
                    width=1344,
                    fps=24,
                    seed=42,
                    prompt=source["enhanced_prompt"],
                    image_paths=images,
                    video_paths=[],
                    audio_paths=[],
                    scheduler_type="reference_blend",
                    video_sigmas=[1 - i / 29 for i in range(30)],
                    audio_sigmas=[1 - i / 29 for i in range(30)],
                    video_flow_shift=12,
                    audio_flow_shift=3,
                    model_path="/models/original",
                    transformer_partition="transformer_ref",
                    transformer_path="/models/original/transformer_ref",
                    model_variant="ref2av",
                    source_id=source_id,
                    cache_fingerprint=source_id,
                    prompt_token_count=3,
                    reference_video_rows=count * 6,
                    reference_audio_rows=0,
                    ref_image_count=count,
                    reference_video_count=0,
                    reference_audio_count=0,
                    guidance_scale=1.0,
                    enable_cfg=False,
                    lora_configs=[],
                    reference_image_resize_mode="match",
                    references=[dict(kind="image", normalized=True, path=image, num_latent_frames=1, latent_height=4, latent_width=6, num_audio_latents=0) for image in images],
                    model_identity={"sha256": "original"},
                    decfg_merged_identity=[],
                )
                path = directory / "manifest.json"
                write_json(path, manifest)
                write_json(directory / "completion.json", dict(manifest_sha256=prepare.file_digest(path), request_fingerprint=index["request_fingerprint"], cache_fingerprint=source_id))
                for name in ("condition.pt", "video_latents.pt", "audio_latents.pt"):
                    (directory / name).write_bytes(b"opaque fixture; metadata preparation never deserializes tensors")
                self.manifests.append(path)
                raw.append(source)
                indexes[node].append(index)
        write_rows(self.raw, raw)
        for node in (0, 1):
            index = self.exports / f"index.node{node}.jsonl"
            failed = self.exports / f"failures.node{node}.jsonl"
            write_rows(index, indexes[node])
            write_rows(failed, [])
            write_json(
                self.exports / f"complete.node{node}.json",
                dict(expected=5, completed=5, failed=0, selection_sha256="shared-selection", index_sha256=prepare.file_digest(index), failures_sha256=prepare.file_digest(failed)),
            )
        self.selfgen_rows = []
        for number in range(6):
            count = 2 if number == 4 else 1
            height, width = (512, 512) if number == 5 else ((768, 1344) if number % 2 else (1344, 768))
            row = dict(
                task="ref2av",
                source_id=f"selfgen-{number}",
                caption=f"selfgen prompt {number}",
                cache_fingerprint=f"condition-{number}",
                latent_fingerprint=f"latents-{number}",
                target_num_frames=124,
                target_height=height,
                target_width=width,
                fps=24,
                reference_image_count=count,
                reference_video_count=0,
                reference_audio_count=0,
                references=[dict(kind="image", path=f"ref-{number}-{i}.png") for i in range(count)],
                vae_normalized=True,
                video_latent_shape=[24, 37, height // 16, width // 16],
                audio_latent_shape=[2, 32, 207],
                packed_sequence_tokens_124=40000,
                condition_path=f"condition-{number}.pt",
                video_latent_path=f"video-{number}.pt",
                audio_latent_path=f"audio-{number}.pt",
            )
            self.selfgen.parent.mkdir(exist_ok=True)
            for key in ("condition_path", "video_latent_path", "audio_latent_path"):
                (self.selfgen.parent / row[key]).write_bytes(b"opaque fixture")
            self.selfgen_rows.append(row)
        self.publish_selfgen()

    def publish_selfgen(self):
        write_rows(self.selfgen, self.selfgen_rows)
        write_json(
            self.selfgen.with_suffix(".complete.json"),
            dict(
                completed_count=len(self.selfgen_rows),
                input_total_rows=len(self.selfgen_rows),
                failed_count=0,
                preprocess_fingerprint="selfgen-complete",
                manifest_sha256=prepare.file_digest(self.selfgen),
            ),
        )

    def run_prepare(self, **kwargs):
        return prepare.prepare(self.exports, self.raw, self.root, self.selfgen, self.output, per_count=2, **kwargs)

    def test_publish_balanced_pairs_and_reuse_existing_tensor_paths(self):
        receipt = self.run_prepare()
        self.assertEqual(receipt["reference_image_counts"], {str(i): 2 for i in range(1, 7)})
        self.assertEqual(receipt["source_counts"], {"omni_teacher_base": 10, "selfgen_single_image": 2})
        self.assertEqual(receipt["selfgen_eligible_single_image_rows"], 4)
        values, _ = prepare.checked_receipt(self.output / "metadata.jsonl")
        self.assertEqual(Counter(row["reference_image_count"] for row in values), {i: 2 for i in range(1, 7)})
        self.assertTrue(all(Path(row[key]).is_absolute() for row in values for key in ("condition_path", "video_latent_path", "audio_latent_path")))
        self.assertEqual(values[0]["prompt"], "own prompt teacher-2-0")
        self.assertEqual(values[0]["cache_fingerprint"], "teacher-2-0")
        self.assertFalse(list(self.output.rglob("*.pt")))
        gan = prepare.teacher_import._leaf("lightx2v_train/trainers/dmd/dmd2_gan_data.py", "_ref6k_test_gan")
        _, eligible, _ = gan.load_gan_real_records(self.output, task="ref2av", geometry=None)
        self.assertEqual(len(eligible), 12)
        conditions, _ = prepare.checked_receipt(self.output / "condition_metadata.jsonl")
        self.assertTrue(all("video_latent_path" not in row and "condition_path" in row for row in conditions))

    def test_deterministic_resume_and_changed_seed_requires_new_output(self):
        first = self.run_prepare()
        manifest = self.output / "metadata.jsonl"
        timestamp = manifest.stat().st_mtime_ns
        self.assertEqual(first, self.run_prepare())
        self.assertEqual(manifest.stat().st_mtime_ns, timestamp)
        with self.assertRaisesRegex(ValueError, "inputs/selection changed"):
            self.run_prepare(seed=43)

    def test_incomplete_node_is_rejected_before_publication(self):
        marker = self.exports / "complete.node1.json"
        receipt = prepare.read_json(marker)
        receipt["failed"] = 1
        write_json(marker, receipt)
        with self.assertRaisesRegex(ValueError, "zero failures"):
            self.run_prepare()
        self.assertFalse((self.output / "metadata.complete.json").exists())

    def test_modified_sample_manifest_breaks_completion_hash(self):
        path = self.manifests[0]
        manifest = prepare.read_json(path)
        manifest["prompt"] = "changed"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "sample completion differs"):
            self.run_prepare()

    def test_decfg_exports_are_rejected(self):
        for path in self.manifests:
            manifest = prepare.read_json(path)
            manifest["decfg_merged_identity"] = [{"sha256": "adapted"}]
            write_json(path, manifest)
            completion = prepare.read_json(path.parent / "completion.json")
            completion["manifest_sha256"] = prepare.file_digest(path)
            write_json(path.parent / "completion.json", completion)
        with self.assertRaisesRegex(ValueError, "without DeCFG"):
            self.run_prepare()
        self.assertFalse((self.output / "metadata.complete.json").exists())

    def test_missing_selfgen_target_and_insufficient_single_image_rows(self):
        original = copy.deepcopy(self.selfgen_rows)
        self.selfgen_rows = [original[0], original[4], original[5]]
        self.publish_selfgen()
        with self.assertRaisesRegex(ValueError, "only 1 eligible"):
            self.run_prepare()
        self.selfgen_rows = original
        self.publish_selfgen()
        (self.selfgen.parent / self.selfgen_rows[0]["audio_latent_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing file"):
            self.run_prepare()

    def test_missing_cost_is_rejected(self):
        for row in self.selfgen_rows:
            row.pop("packed_sequence_tokens_124")
        self.publish_selfgen()
        with self.assertRaisesRegex(ValueError, "Missing Ref token cost"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
