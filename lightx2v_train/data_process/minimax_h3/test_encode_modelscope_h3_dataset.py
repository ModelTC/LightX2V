"""Cache provenance, resume, failures, audio semantics and actual CM loader."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_test_modelscope_encoder", HERE / "encode_modelscope_h3_dataset.py")
ENCODER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENCODER
SPEC.loader.exec_module(ENCODER)
REF = ENCODER._module("build_minimax_h3_ref2av_condition_caches.py", "_test_modelscope_reference")
AV = ENCODER._module("encode_minimax_h3_teacher_av_latents.py", "_test_modelscope_av")


class ModelscopeEncoderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source.mp4"
        self.source.write_bytes(b"fake media; heavy calls replaced in lifecycle tests")
        self.output = self.root / "cache"
        self.metadata = self.root / "prepared.jsonl"
        self.row = {
            "source_id": "dataset:t2av:000001",
            "source_index": 19,
            "task": "t2av",
            "caption": "完整 prompt unchanged",
            "video_path": str(self.source),
            "target_height": 768,
            "target_width": 1344,
            "target_num_frames": 124,
            "num_frames": 124,
            "fps": 24,
        }
        self.write_rows([self.row])

    def write_rows(self, rows):
        self.metadata.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    def args(self, *extra):
        return ENCODER.parse_args(["--metadata", str(self.metadata), "--output-dir", str(self.output), "--model-path", str(self.root / "model"), "--device", "cpu", "--num-shards", "1", *extra])

    def run_toy(self, *, fail_text=False, nonfinite=False, expected_loads=True, extra=()):
        import diffusers

        from lightx2v_train.model_zoo.minimax_h3 import condition_encoder

        calls = {"text": 0, "vae": 0, "tile": 0}

        class Text:
            def __init__(self, *args, **kwargs):
                calls["text"] += 1
                if not expected_loads:
                    raise AssertionError("Complete cache reloaded Qwen")

            def encode(self, caption):
                if fail_text:
                    raise ValueError("deliberate bad text sample")
                return {"prompt_embeds": torch.ones(1, 2, 5120, dtype=torch.bfloat16), "text_token_tags": torch.ones(2, dtype=torch.long)}

        class VAE:
            def requires_grad_(self, value):
                return self

            def eval(self):
                return self

            def to(self, device):
                return self

            def enable_tiling(self, **kwargs):
                calls["tile"] += 1

        def vae(*args, **kwargs):
            calls["vae"] += 1
            if not expected_loads:
                raise AssertionError("Complete cache reloaded VAEs")
            return VAE()

        def video(vae, pixels, *args):
            return torch.zeros(24, 37, pixels[0] // 16, pixels[1] // 16)

        audio = torch.zeros(2, 32, 207)
        if nonfinite:
            audio[0, 0, 0] = float("nan")
        with (
            patch.object(ENCODER, "_module", side_effect=lambda filename, name: AV if filename.startswith("encode_minimax") else REF),
            patch.object(ENCODER, "model_identity", return_value={"model": "toy"}),
            patch.object(condition_encoder, "MiniMaxH3ConditionEncoder", Text),
            patch.object(diffusers.AutoencoderKLMiniMaxH3, "from_pretrained", side_effect=vae),
            patch.object(diffusers.AutoencoderKLMiniMaxH3Audio, "from_pretrained", side_effect=vae),
            patch.object(AV, "decode_audio", return_value=torch.zeros(2, 165600)),
            patch.object(AV, "decode_video", side_effect=lambda path, height, width, *args: (height, width)),
            patch.object(AV, "encode_video_latent", side_effect=video),
            patch.object(AV, "encode_audio_latent", return_value=audio),
        ):
            rc = ENCODER.run(self.args(*extra))
        return rc, calls

    def test_shards_follow_manifest_order_even_when_source_indices_have_gaps(self):
        rows = [dict(self.row, source_id=f"dataset:{index}", source_index=19 + 13 * index) for index in range(11)]
        self.write_rows(rows)
        shards = [ENCODER.read_rows(self.metadata, rank, 4) for rank in range(4)]
        for rank, shard in enumerate(shards):
            self.assertEqual([row["source_id"] for row in shard], [row["source_id"] for row in rows[rank::4]])
        self.assertEqual(sum(map(len, shards)), len(rows))
        self.assertEqual(len({row["source_id"] for shard in shards for row in shard}), len(rows))

    def test_duplicate_identity_and_bad_native_geometry_rejected(self):
        for rows in ([self.row, self.row], [dict(self.row, target_width=1234)]):
            self.write_rows(rows)
            with self.assertRaises(ValueError):
                ENCODER.read_rows(self.metadata, 0, 1)

    def test_roundtrip_resumes_without_models_and_loads_actual_cm_dataset(self):
        from lightx2v_train.data.minimax_h3_turbo_cm_dataset import MiniMaxH3TurboCMDataset

        rc, calls = self.run_toy()
        self.assertEqual((rc, calls), (0, {"text": 1, "vae": 2, "tile": 1}))
        manifest = self.output / "metadata.shard-00000-of-00001.jsonl"
        row = json.loads(manifest.read_text())
        self.assertEqual(row["caption"], self.row["caption"])
        dataset = MiniMaxH3TurboCMDataset(manifest, defer_latent_loading=False)
        sample = dataset[0]
        self.assertEqual(sample["inputs"]["latents"].shape, (24, 37, 48, 84))
        self.assertEqual(sample["inputs"]["audio_latents"].shape, (2, 32, 207))
        self.assertEqual(sample["inputs"]["latents"].dtype, torch.bfloat16)
        self.assertTrue((sample["conditioning"]["positive"]["text_token_tags"] == 1).all())
        self.assertEqual(sample["meta"]["source_id"], self.row["source_id"])
        self.assertEqual(self.run_toy(expected_loads=False), (0, {"text": 0, "vae": 0, "tile": 0}))
        receipt = json.loads((self.output / "receipt.shard-00000-of-00001.json").read_text())
        self.assertEqual(receipt["prepared_sha256"], ENCODER.sha256_file(self.metadata))
        self.assertEqual((receipt["expected_count"], receipt["completed_count"], receipt["failed_count"]), (1, 1, 0))

    def test_native_non_768p_shape_is_preserved(self):
        self.row.update(target_height=1184, target_width=864)
        self.write_rows([self.row])
        self.assertEqual(self.run_toy()[0], 0)
        row = json.loads((self.output / "metadata.shard-00000-of-00001.jsonl").read_text())
        self.assertEqual(row["video_latent_shape"], [24, 37, 74, 54])

    def test_changed_source_or_tiling_contract_cannot_reuse_cache(self):
        self.assertEqual(self.run_toy()[0], 0)
        self.source.write_bytes(self.source.read_bytes() + b"changed")
        with self.assertRaisesRegex(RuntimeError, "source contract changed"):
            self.run_toy(expected_loads=False)
        with self.assertRaisesRegex(RuntimeError, "namespace changed"):
            self.run_toy(expected_loads=False, extra=("--no-vae-tiling",))

    def test_nonfinite_target_becomes_accounted_failure_not_training_manifest(self):
        self.assertEqual(self.run_toy(nonfinite=True)[0], 1)
        self.assertEqual((self.output / "metadata.shard-00000-of-00001.jsonl").read_text(), "")
        failed = json.loads((self.output / "failed.shard-00000-of-00001.jsonl").read_text())
        self.assertEqual(failed["source_id"], self.row["source_id"])
        receipt = json.loads((self.output / "receipt.shard-00000-of-00001.json").read_text())
        self.assertEqual((receipt["expected_count"], receipt["completed_count"], receipt["failed_count"]), (1, 0, 1))
        self.assertEqual(self.run_toy(nonfinite=True, extra=("--allow-failures",))[0], 0)

    def test_all_text_failures_do_not_reference_uninitialized_tensors(self):
        rc, calls = self.run_toy(fail_text=True)
        self.assertEqual(rc, 1)
        self.assertEqual(calls["vae"], 0)

    def test_ref_image_keeps_identity_prompt_and_native_cost_metadata(self):
        image = self.root / "reference.jpg"
        image.write_bytes(b"schema fixture; media content not decoded here")
        row = dict(self.row, task="ref2av", source_id="dataset:ref2av:000001", references=[{"kind": "image", "path": str(image)}])
        job = ENCODER.build_jobs([row], self.output, {"model": "test"}, REF, "bf16")[0]
        sample = job["reference_sample"]
        self.assertEqual((sample.source_id, sample.prompt), (row["source_id"], row["caption"]))
        payload = {
            "cache_schema_version": 1,
            "cache_stage": "complete",
            "cache_fingerprint": sample.fingerprint,
            "source_id": row["source_id"],
            "task": "ref2av",
            "conditioning": {
                "positive": {
                    "task": "ref2av",
                    "target_height": 768,
                    "target_width": 1344,
                    "target_num_frames": 124,
                    "prompt_embeds": torch.zeros(1, 2, 5120, dtype=torch.bfloat16),
                    "text_token_tags": torch.tensor([0, 1]),
                    "references": [{"kind": "image", "normalized": True, "video_latents": torch.zeros(4, 96, dtype=torch.bfloat16), "num_latent_frames": 1, "latent_height": 4, "latent_width": 4}],
                }
            },
        }
        AV.atomic_torch_save(payload, job["condition_path"])
        ENCODER.validate_condition(job, torch.bfloat16, REF)
        record = ENCODER.manifest_row(job, payload, REF)
        self.assertEqual((record["reference_image_count"], record["reference_video_count"], record["reference_audio_count"]), (1, 0, 0))
        self.assertEqual(record["source_id"], row["source_id"])
        self.assertEqual(record["packed_sequence_tokens_124"], 2 + 4 + 37 * 24 * 42 + 414)

    def test_installed_reference_classes_normalize_and_encode_image_contract(self):
        from PIL import Image
        from diffusers.modular_pipelines.minimax_h3 import encoders as native

        image = self.root / "real_reference.jpg"
        Image.new("RGB", (32, 32), "red").save(image)
        row = dict(self.row, task="ref2av", source_id="dataset:ref2av:real-image", references=[{"kind": "image", "path": str(image)}])
        job = ENCODER.build_jobs([row], self.output, {"model": "test"}, REF, "bf16")[0]
        runtime = ENCODER.adapt_reference_runtime(REF.import_official_runtime())
        prepared = REF.prepare_official_references(job["reference_sample"], runtime)
        self.assertEqual(prepared[0].kind, "image")
        self.assertEqual(prepared[0].image.size, (32, 32))
        self.assertFalse(prepared[0].has_audio)
        if hasattr(native.MiniMaxH3Ref2VATextEncoderStep, "encode_prompt"):
            self.skipTest("This integration assertion covers the newer PipelineState block API")

        class Processor:
            merge_size = 1

            def __call__(self, **kwargs):
                return {"pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.tensor([[1, 2, 2]])}

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1]}

            def convert_tokens_to_ids(self, token):
                return 2

        components = SimpleNamespace(
            text_encoder=SimpleNamespace(dtype=torch.bfloat16), tokenizer=Tokenizer(), processor=SimpleNamespace(image_processor=Processor()), _execution_device=torch.device("cpu")
        )
        with patch.object(native, "get_qwen3vl_prompt_embeds", side_effect=lambda encoder, processor, ids, vision, **kwargs: torch.zeros(1, len(ids), 5120, dtype=torch.bfloat16)):
            payload = REF.encode_text_stage(job["reference_sample"], runtime, components, torch.bfloat16)
        self.assertTrue((payload["conditioning"]["positive"]["text_token_tags"] == 0).any())

        class VAE:
            config = SimpleNamespace(latents_mean=[0.0] * 24, latents_std=[1.0] * 24)

            def encode(self, pixels, **kwargs):
                value = torch.zeros(1, 24, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
                return (SimpleNamespace(sample=lambda **kwargs: value),)

        components = SimpleNamespace(
            vae=VAE(),
            audio_vae=SimpleNamespace(config=SimpleNamespace(latents_mean=[0.0] * 32, latents_std=[1.0] * 32, sampling_rate=32000)),
            _execution_device=torch.device("cpu"),
            patch_size=(1, 2, 2),
            audio_latent_channels=32,
        )
        payload = REF.encode_reference_stage(job["reference_sample"], payload, runtime, components, torch.bfloat16)
        self.assertEqual(payload["cache_stage"], "complete")
        visual = payload["conditioning"]["positive"]["references"][0]
        self.assertEqual(visual["video_latents"].shape, (1, 96))
        self.assertEqual((visual["num_latent_frames"], visual["latent_height"], visual["latent_width"]), (1, 2, 2))

    def test_video_only_reference_strips_audio_without_changing_original(self):
        import av

        source = self.root / "with_audio.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=32x32:r=24:d=0.2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=32000:duration=0.2",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
        )
        stripped = ENCODER._silent_reference(source, self.output)
        with av.open(str(source)) as opened:
            self.assertEqual(len(opened.streams.audio), 1)
        with av.open(str(stripped)) as opened:
            self.assertEqual((len(opened.streams.video), len(opened.streams.audio)), (1, 0))
        self.assertEqual(ENCODER._silent_reference(source, self.output), stripped)

    def test_video_only_reference_pyav_fallback_preserves_frames_and_original_audio(self):
        import av

        source = self.root / "fallback_with_audio.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=32x32:r=24:d=0.5",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=32000:duration=0.5",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
                str(source),
            ],
            check=True,
        )
        with av.open(str(source)) as opened:
            expected_frames = sum(1 for _ in opened.decode(video=0))
        with patch.object(ENCODER.shutil, "which", return_value=None), patch.object(ENCODER.subprocess, "run", side_effect=AssertionError("PyAV fallback must not launch an OS binary")):
            stripped = ENCODER._silent_reference(source, self.output)
        with av.open(str(source)) as opened:
            self.assertEqual(len(opened.streams.audio), 1)
        with av.open(str(stripped)) as opened:
            self.assertEqual((len(opened.streams.video), len(opened.streams.audio)), (1, 0))
            frames = sum(1 for _ in opened.decode(video=0))
        self.assertEqual(frames, expected_frames)


if __name__ == "__main__":
    unittest.main()
