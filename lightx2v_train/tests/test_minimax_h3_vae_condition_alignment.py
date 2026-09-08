import unittest
from unittest.mock import patch

import torch

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_vae_distillation_capability import (
    MiniMaxH3VAEDistillationCapability,
    _SampleGroup,
)
from lightx2v_train.model_zoo.native.minimax_h3.vae_protocol import MiniMaxH3VideoVAE
from lightx2v_train.model_zoo.native.minimax_h3.vae_geometry import (
    CLIP_LENGTH,
    FRAME_OVERLAP,
    SPATIAL_COMPRESSION_RATIO,
    spatial_tile_layout,
    temporal_decode_windows,
    video_latent_num_frames,
)
from lightx2v_train.trainers.vae.adversarial import VAEAdversarialObjective


class _Student:
    blend = staticmethod(MiniMaxH3VideoVAE.blend)


class _Model:
    def denoiser_module(self):
        return _Student()


class ConditionAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.capability = MiniMaxH3VAEDistillationCapability(self.model, {})

    @staticmethod
    def _condition_sequence(num_latent_frames):
        tokens = torch.arange(num_latent_frames, dtype=torch.float32)
        # Independently express H3's 17-frame chunk: one frame, then four frames per token.
        repeats = torch.where(tokens.long().remainder(5) == 0, 1, 4)
        return tokens.repeat_interleave(repeats)

    def test_raw_window_padding_is_not_uniform_seven_to_twenty_two_resizing(self):
        latents = torch.arange(7, dtype=torch.float32).view(1, 1, 7, 1, 1)
        group = _SampleGroup("single", 0, 1, 0, 1, 0, 1)

        condition = self.capability._frame_aligned_latent_condition(latents, ((0, 7),), group)

        expected = torch.tensor([0] + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] + [6] * 4)
        torch.testing.assert_close(condition.flatten(), expected.float())

    def test_all_window_routes_match_global_frame_condition_after_valid_crop(self):
        for num_frames in (107, 124, 362):
            latent_frames = video_latent_num_frames(num_frames)
            latents = torch.arange(latent_frames, dtype=torch.float32).view(1, 1, latent_frames, 1, 1)
            windows = temporal_decode_windows(latent_frames)
            full_condition = self._condition_sequence(latent_frames)
            layout = spatial_tile_layout(1, 1)
            video = torch.zeros(1, 1, num_frames, 16, 16)
            for count in (1, 2, 4):
                for start in range(len(windows) - count + 1):
                    with self.subTest(frames=num_frames, count=count, start=start):
                        group = _SampleGroup("single", start, count, 0, 1, 0, 1)
                        stitched = torch.zeros(1, 1, count * CLIP_LENGTH + FRAME_OVERLAP, 16, 16)

                        prediction, target, condition, _ = self.capability._valid_stitched_crop(
                            stitched, video, latents, num_frames, layout, windows, group
                        )

                        front = FRAME_OVERLAP if start else 0
                        back = FRAME_OVERLAP if start + count < len(windows) else 0
                        expected_start = start * CLIP_LENGTH + front
                        expected_end = (start + count) * CLIP_LENGTH + FRAME_OVERLAP - back
                        torch.testing.assert_close(condition.flatten(), full_condition[expected_start:expected_end])
                        self.assertEqual(condition.shape[2], prediction.shape[2])
                        self.assertEqual(target.shape, prediction.shape)

    def test_spatial_cropping_remains_aligned_with_frame_condition(self):
        latents = torch.arange(37 * 4 * 6, dtype=torch.float32).reshape(1, 1, 37, 4, 6)
        windows = temporal_decode_windows(37)
        layout = spatial_tile_layout(4, 6, tile_height=48, tile_width=64, overlap_height=16, overlap_width=16)
        group = _SampleGroup("single", 2, 1, 1, 1, 1, 1)
        video = torch.zeros(1, 1, 124, 64, 96)
        stitched = torch.zeros(1, 1, 22, 48, 64)

        prediction, _, condition, _ = self.capability._valid_stitched_crop(
            stitched, video, latents, 124, layout, windows, group
        )

        expected = latents[:, :, :, 3:4, 4:6]
        expected = self.capability._frame_aligned_latent_condition(expected, windows, group)[:, :, 5:17]
        torch.testing.assert_close(condition, expected)
        self.assertEqual(condition.shape[2], prediction.shape[2])
        self.assertEqual(condition.shape[-2] * SPATIAL_COMPRESSION_RATIO, prediction.shape[-2])
        self.assertEqual(condition.shape[-1] * SPATIAL_COMPRESSION_RATIO, prediction.shape[-1])

    def test_gan_crop_preserves_per_frame_condition_at_nonzero_offset(self):
        objective = object.__new__(VAEAdversarialObjective)
        objective.spatial_compression_ratio = SPATIAL_COMPRESSION_RATIO
        time = torch.arange(63, dtype=torch.float32)
        video = time.view(1, 1, 63, 1, 1).expand(1, 1, 63, 32, 48)
        condition = time.view(1, 1, 63, 1, 1).expand(1, 1, 63, 2, 3)

        with patch.object(objective, "_random_start", side_effect=(0, 1, 17)):
            fake, real, cropped_condition, shape = objective._aligned_crop(video, video, condition, 32, 29)

        self.assertEqual(shape, (29, 32, 32))
        torch.testing.assert_close(fake[0, 0, :, 0, 0], time[17:46])
        torch.testing.assert_close(real, fake)
        torch.testing.assert_close(cropped_condition[0, 0, :, 0, 0], time[17:46])


if __name__ == "__main__":
    unittest.main()
