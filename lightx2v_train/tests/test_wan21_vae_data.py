"""Wan frame alignment and RGB-only reuse of existing video manifests."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from lightx2v_train.data.utils import VideoFrameSampler, load_video_tensor
from lightx2v_train.data.video_dataset import VideoDataset
from lightx2v_train.model_zoo.wan.vae_data_process import Wan21VAEDistillationProcessor
from lightx2v_train.utils.registry import build_sample_processor


class Wan21VAEDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_both_components_build_rgb_only_processors(self):
        for component in ("encoder", "decoder"):
            name = f"wan21_pruned_{component}"
            processor = build_sample_processor({"model": {"name": name}, "data": {"processor": {"name": name}}})
            self.assertIsInstance(processor, Wan21VAEDistillationProcessor)
            self.assertFalse(processor.requires_audio)
            self.assertFalse(processor.load_cached_latents)

    def test_padding_preserves_source_frames_and_value_space(self):
        processor = Wan21VAEDistillationProcessor()
        for source_frames in (1, 32, 33, 64, 65):
            with self.subTest(source_frames=source_frames):
                original = torch.linspace(-1, 1, 3 * source_frames * 24 * 40).reshape(3, source_frames, 24, 40)
                sample = {
                    "inputs": {"video": original.clone()},
                    "meta": {"source_frame_rate": 30, "latent_path": "/missing/h3-latent.pt"},
                }
                with patch("torch.load", side_effect=AssertionError("Wan training must not read H3 latents")):
                    output = processor(sample)
                video = output["inputs"]["video"]
                self.assertEqual(video.shape[1] % 4, 1)
                self.assertEqual(video.shape[-2:], (24, 40))
                self.assertEqual(output["meta"]["source_num_frames"], source_frames)
                self.assertEqual(output["meta"]["source_frame_rate"], 30)
                torch.testing.assert_close(video[:, :source_frames], (original + 1) * 0.5)
                if video.shape[1] > source_frames:
                    torch.testing.assert_close(video[:, source_frames:], video[:, source_frames - 1:source_frames].expand(-1, video.shape[1] - source_frames, -1, -1))
                self.assertNotIn("latents", output["inputs"])
                self.assertNotIn("latent_path", output["meta"])

    def test_invalid_spatial_geometry_is_rejected(self):
        processor = Wan21VAEDistillationProcessor()
        for shape in ((3, 33, 23, 32), (3, 0, 32, 32), (1, 33, 32, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                processor({"inputs": {"video": torch.zeros(shape)}, "meta": {}})

    def test_training_rejects_short_source_before_padding(self):
        processor = Wan21VAEDistillationProcessor(min_source_frames=65)
        for frames in (1, 33, 61, 64):
            with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "real source frames"):
                processor({"inputs": {"video": torch.zeros(3, frames, 32, 32)}, "meta": {}})
        result = processor({"inputs": {"video": torch.zeros(3, 65, 32, 32)}, "meta": {}})
        self.assertEqual(result["meta"]["source_num_frames"], 65)
        self.assertEqual(result["inputs"]["video"].shape[1], 65)

    def test_processor_frame_contract_covers_largest_loss_crop(self):
        config = {
            "model": {"name": "wan21_pruned_encoder"},
            "data": {"processor": {"name": "wan21_pruned_encoder", "min_source_frames": 33}},
            "training": {"vae_distillation": {"stages": [{"crop_num_frames": 33}, {"crop_num_frames": 65}]}},
        }
        with self.assertRaisesRegex(ValueError, "largest training crop"):
            build_sample_processor(config)
        config["data"]["processor"]["min_source_frames"] = 65
        self.assertEqual(build_sample_processor(config).min_source_frames, 65)

    def test_missing_h3_latent_does_not_discard_original_video(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "video.mp4"
            video_path.touch()
            manifest = root / "metadata.jsonl"
            manifest.write_text(json.dumps({"video_path": str(video_path), "latent_path": "missing.pt"}) + "\n")
            dataset = VideoDataset([manifest], sample_processor=Wan21VAEDistillationProcessor(), skip_missing=False)
            self.assertEqual(len(dataset), 1)
            self.assertNotIn("latent_path", dataset.samples[0]["meta"])

    def test_video_resize_keeps_aspect_ratio_then_center_crops(self):
        frame = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None], (32, 64, 3)).copy()
        reader = Mock()
        reader.get_meta_data.return_value = {"fps": 24}
        reader.count_frames.return_value = 1
        reader.get_data.return_value = frame
        with patch("lightx2v_train.data.utils.imageio.get_reader", return_value=reader):
            actual = load_video_tensor("unused.mp4", 16, 16, VideoFrameSampler(num_frames=1))
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        expected = np.array(Image.fromarray(frame).resize((32, 16), resampling).crop((8, 0, 24, 16)), copy=True)
        expected = torch.from_numpy(expected).permute(2, 0, 1).unsqueeze(1).float() / 127.5 - 1.0
        torch.testing.assert_close(actual, expected)
        reader.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
