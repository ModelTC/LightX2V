"""Encoder RGB input contracts, without requiring cached posterior latents."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from lightx2v_train.data.video_dataset import VideoDataset
from lightx2v_train.model_zoo.minimax_h3.vae_data_process import MiniMaxH3VAEDistillationProcessor
from lightx2v_train.utils.registry import build_sample_processor


class EncoderDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_encoder_processor_is_available_in_a_fresh_process(self):
        program = """
from lightx2v_train.data import build_sample_processor
processor = build_sample_processor({
    'model': {'name': 'minimax_h3_pruned_encoder'},
    'data': {'processor': {'name': 'minimax_h3_pruned_encoder'}},
})
assert processor.load_cached_latents is False
assert processor.requires_audio is False
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_latent_does_not_remove_encoder_rgb_samples(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            metadata = root / "metadata.jsonl"
            records = [
                {"video_path": str(video), "latent_path": "missing.pt", "target_height": 32, "target_width": 32},
                {"video_path": str(video), "target_height": 32, "target_width": 32},
            ]
            metadata.write_text("".join(json.dumps(record) + "\n" for record in records))
            encoder = MiniMaxH3VAEDistillationProcessor(load_cached_latents=False)
            dataset = VideoDataset([metadata], sample_processor=encoder, skip_missing=False)
            self.assertEqual(len(dataset), 2)
            self.assertTrue(all("latent_path" not in sample["meta"] for sample in dataset.samples))

            decoder = MiniMaxH3VAEDistillationProcessor()
            with self.assertRaisesRegex(FileNotFoundError, "Latent path points to a missing file"):
                VideoDataset([metadata], sample_processor=decoder, skip_missing=False)
            dataset = VideoDataset([metadata], sample_processor=decoder, skip_missing=True)
            self.assertEqual(len(dataset), 1)

    def test_encoder_uses_original_rgb_and_never_loads_latent_cache(self):
        processor = build_sample_processor({
            "model": {"name": "minimax_h3_pruned_encoder"},
            "data": {"processor": {"name": "minimax_h3_pruned_encoder"}},
        })
        for source_frames in (120, 124):
            with self.subTest(source_frames=source_frames):
                video = torch.linspace(-1, 1, 3 * source_frames * 32 * 32).reshape(3, source_frames, 32, 32)
                original = video.clone()
                sample = {
                    "inputs": {"video": video},
                    "conditioning": {"prompt": ""},
                    "meta": {"source_frame_rate": 24, "latent_path": "/missing/encoder-cache.pt"},
                }
                with patch("torch.load", side_effect=AssertionError("Encoder processor must not load cached latents")):
                    result = processor(sample)
                self.assertEqual(tuple(result["inputs"]["video"].shape), (3, 124, 32, 32))
                self.assertNotIn("latents", result["inputs"])
                self.assertEqual(result["meta"]["source_num_frames"], source_frames)
                self.assertEqual(result["meta"]["num_frames"], 124)
                torch.testing.assert_close(result["inputs"]["video"][:, :source_frames], (original + 1) * 0.5)
                if source_frames < 124:
                    torch.testing.assert_close(
                        result["inputs"]["video"][:, source_frames:],
                        ((original[:, -1:] + 1) * 0.5).expand(-1, 124 - source_frames, -1, -1),
                    )


if __name__ == "__main__":
    unittest.main()
