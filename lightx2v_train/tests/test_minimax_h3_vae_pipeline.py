import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from lightx2v_train.data.utils import VideoFrameSampler
from lightx2v_train.data.video_dataset import build_video_dataset
from lightx2v_train.infer.vae import MiniMaxH3VAEReconstructionInferencer, _chunked_video_metric, _pil_frames
from lightx2v_train.model_capabilities import VAEDistillationStepContext
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
)
from lightx2v_train.model_zoo.minimax_h3.vae_data_process import MiniMaxH3VAEDistillationProcessor
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    align_num_frames,
    spatial_tile_layout,
    temporal_decode_windows,
    video_latent_num_frames,
    video_num_frames_from_latent,
)
from lightx2v_train.runtime.distributed import _resolve_parallel_sizes
from lightx2v_train.utils.video import video_psnr, video_ssim


class _Reader:
    def __init__(self, frames, fps=24.0, duration=None):
        self.frames = frames
        self.fps = fps
        self.duration = frames / fps if duration is None else duration
        self.count_calls = 0

    def count_frames(self):
        self.count_calls += 1
        return self.frames

    def get_meta_data(self):
        return {"fps": self.fps, "duration": self.duration}


class MiniMaxH3GeometryTest(unittest.TestCase):
    def test_4_to_15_second_shapes(self):
        expected = {
            4: (107, 32),
            5: (124, 37),
            6: (158, 47),
            7: (175, 52),
            8: (192, 57),
            9: (226, 67),
            10: (243, 72),
            11: (277, 82),
            12: (294, 87),
            13: (328, 97),
            14: (345, 102),
            15: (362, 107),
        }
        for seconds, (frames, latent_frames) in expected.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(align_num_frames(seconds * 24), frames)
                self.assertEqual(video_latent_num_frames(frames), latent_frames)
                self.assertEqual(video_num_frames_from_latent(latent_frames), frames)
                windows = temporal_decode_windows(latent_frames)
                self.assertGreater(len(windows), 0)
                self.assertEqual(len(windows) * 17 + 5, frames)

    def test_spatial_layout_matches_h3_tiles(self):
        layout = spatial_tile_layout(48, 84)
        self.assertEqual(layout.num_tiles, 28)
        self.assertTrue(all(item.stop - item.start == 16 for item in layout.height_slices))
        self.assertTrue(all(item.stop - item.start == 16 for item in layout.width_slices))


class MiniMaxH3PreprocessingTest(unittest.TestCase):
    @patch("lightx2v_train.data.video_dataset._build_dataloader")
    @patch("lightx2v_train.data.video_dataset.VideoDataset")
    def test_video_builder_accepts_registry_prompt(self, video_dataset, build_dataloader):
        processor = object()
        build_video_dataset(
            {"data_path": "metadata.jsonl"},
            sample_processor=processor,
            unconditional_prompt="negative prompt",
        )

        self.assertEqual(video_dataset.call_args.kwargs["sample_processor"], processor)
        self.assertEqual(video_dataset.call_args.kwargs["unconditional_prompt"], "negative prompt")
        build_dataloader.assert_called_once()

    def test_preserve_frame_sampler_leaves_h3_alignment_to_processor(self):
        sampler = VideoFrameSampler(
            num_frames=362,
            time_division_factor=1,
            time_division_remainder=0,
            frame_rate=24,
            fix_frame_rate=False,
        )
        self.assertEqual(sampler.sample_count(_Reader(96)), 96)
        self.assertEqual(sampler.sample_count(_Reader(124)), 124)
        self.assertEqual(sampler.sample_count(_Reader(158, duration=6.58)), 158)
        self.assertEqual(sampler.sample_count(_Reader(362)), 362)

    def test_frame_sampler_counts_source_frames_once(self):
        reader = _Reader(124)
        sampler = VideoFrameSampler(num_frames=124, frame_rate=24, fix_frame_rate=False)

        selection = sampler.sample(reader)

        self.assertEqual(reader.count_calls, 1)
        self.assertEqual(selection.frame_ids, tuple(range(124)))

    def test_processor_pads_four_seconds_to_h3_geometry(self):
        sample = {
            "inputs": {"video": torch.zeros(3, 96, 32, 64)},
            "conditioning": {"prompt": ""},
            "meta": {"source_frame_rate": 24.0},
        }
        result = MiniMaxH3VAEDistillationProcessor()(sample)
        self.assertEqual(result["inputs"]["video"].shape, (3, 107, 32, 64))
        self.assertEqual(result["meta"]["source_num_frames"], 96)
        self.assertEqual(result["meta"]["num_frames"], 107)
        self.assertTrue(torch.all(result["inputs"]["video"] == 0.5))

    def test_processor_loads_normalized_latent_cache(self):
        with TemporaryDirectory() as directory:
            latent_path = Path(directory) / "latent.pt"
            latent = torch.randn(24, 32, 2, 4, dtype=torch.bfloat16)
            torch.save(latent, latent_path)
            sample = {
                "inputs": {"video": torch.zeros(3, 96, 32, 64)},
                "conditioning": {"prompt": ""},
                "meta": {"latent_path": str(latent_path), "source_frame_rate": 24.0},
            }
            result = MiniMaxH3VAEDistillationProcessor()(sample)
            self.assertTrue(torch.equal(result["inputs"]["latents"], latent))

    def test_processor_loads_provenance_wrapped_latent_cache(self):
        with TemporaryDirectory() as directory:
            latent_path = Path(directory) / "latent.pt"
            latent = torch.randn(24, 32, 2, 4, dtype=torch.bfloat16)
            torch.save({"latents": latent, "normalized": True}, latent_path)
            sample = {
                "inputs": {"video": torch.zeros(3, 96, 32, 64)},
                "conditioning": {"prompt": ""},
                "meta": {"latent_path": str(latent_path), "source_frame_rate": 24.0},
            }
            result = MiniMaxH3VAEDistillationProcessor()(sample)
            self.assertTrue(torch.equal(result["inputs"]["latents"], latent))

    def test_processor_rejects_non_h3_frame_rate(self):
        sample = {
            "inputs": {"video": torch.zeros(3, 96, 32, 64)},
            "conditioning": {"prompt": ""},
            "meta": {"source_frame_rate": 23.976},
        }
        with self.assertRaisesRegex(ValueError, "requires 24 fps"):
            MiniMaxH3VAEDistillationProcessor()(sample)


class MiniMaxH3VAEDistillationWindowTest(unittest.TestCase):
    class _Student:
        tile_sample_min_height = 32
        tile_sample_min_width = 32
        tile_sample_min_overlap_height = 16
        tile_sample_min_overlap_width = 16

    class _Model:
        def __init__(self):
            self.student = MiniMaxH3VAEDistillationWindowTest._Student()

        def denoiser_module(self):
            return self.student

    def test_training_spatial_tiling_is_independent_from_inference(self):
        latents = torch.zeros(1, 24, 7, 3, 4)
        context = VAEDistillationStepContext(running_dtype=torch.float32)

        full_model = self._Model()
        full_capability = MiniMaxH3VAEDistillationCapability(full_model, {"use_spatial_tiling": False})
        _, full_stage = full_capability._stage_for(0)
        full_layout, full_windows, full_group = full_capability._sample_training_group(latents, full_stage, context)
        full_latents = full_capability._latent_tiles(latents, full_layout, full_windows, full_group)
        self.assertEqual(full_latents.shape[-2:], (3, 4))

        tiled_model = self._Model()
        tiled_capability = MiniMaxH3VAEDistillationCapability(tiled_model, {})
        _, tiled_stage = tiled_capability._stage_for(0)
        tiled_layout, tiled_windows, tiled_group = tiled_capability._sample_training_group(latents, tiled_stage, context)
        tiled_latents = tiled_capability._latent_tiles(latents, tiled_layout, tiled_windows, tiled_group)
        self.assertEqual(tiled_latents.shape[-2:], (2, 2))


class MiniMaxH3ReconstructionTest(unittest.TestCase):
    def test_chunked_metrics_and_streamed_frames(self):
        target = torch.rand(3, 5, 16, 16)
        prediction = (target + 0.05 * torch.randn_like(target)).clamp(0.0, 1.0)
        for metric in (video_psnr, video_ssim):
            with self.subTest(metric=metric.__name__):
                expected = metric(prediction, target)
                actual = _chunked_video_metric(metric, prediction, target, frame_batch_size=2)
                self.assertAlmostEqual(actual, expected, places=5)
        self.assertEqual(len(list(_pil_frames(prediction))), 5)

    def test_preview_decodes_cached_latents_and_saves_comparison(self):
        class Model:
            device = torch.device("cpu")

            def __init__(self, reconstruction):
                self.reconstruction = reconstruction
                self.decoded_latents = None

            def set_denoiser_eval(self):
                pass

            def decode_latents(self, latents):
                self.decoded_latents = latents
                return self.reconstruction

            def reconstruct(self, _video):
                raise AssertionError("Cached preview must not run the teacher encoder.")

        video = torch.rand(1, 3, 5, 16, 16)
        latents = torch.rand(1, 24, 2, 1, 1)
        sample = {
            "inputs": {"video": video, "latents": latents},
            "meta": {"source_num_frames": torch.tensor(5), "video_path": "/tmp/example.mp4"},
        }
        with TemporaryDirectory() as directory:
            inferencer = MiniMaxH3VAEReconstructionInferencer(
                {
                    "inference": {
                        "output_dir": directory,
                        "fps": 24,
                        "metric_frame_batch_size": 2,
                        "save_comparison": True,
                        "save_png_frame_count": 3,
                    }
                }
            )
            model = Model(video)
            inferencer.set_model(model)
            inferencer.set_data([sample])
            with patch("lightx2v_train.infer.vae.save_mp4") as save_mp4_mock:
                metrics = inferencer.infer()
            png_count = len(list(Path(directory).glob("*_frames/*.png")))

        self.assertIs(model.decoded_latents, latents)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(save_mp4_mock.call_count, 2)
        self.assertEqual(png_count, 3)


class DistributedConfigTest(unittest.TestCase):
    def test_ddp_uses_full_world_as_data_parallel_size(self):
        config = {
            "distributed": {
                "sequence_parallel": {"enabled": False, "size": 1},
                "dp": {"enabled": True},
                "fsdp2": {"enabled": False},
            }
        }
        self.assertEqual(_resolve_parallel_sizes(config, 8), (1, 8))


if __name__ == "__main__":
    unittest.main()
