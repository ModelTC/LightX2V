import unittest

import torch
import torch.nn.functional as F

from lightx2v_train.trainers.dmd.math import (
    dmd_loss,
    dmd_loss_pair,
    dmd_loss_with_stats,
    project_dmd_direction,
)


class DmdMathTest(unittest.TestCase):
    def test_disabled_projection_preserves_float32_baseline(self):
        generator = torch.Generator().manual_seed(42)
        latents = torch.randn(2, 3, 4, generator=generator)
        fake = torch.randn(2, 3, 4, generator=generator)
        teacher = torch.randn(2, 3, 4, generator=generator)
        for norm_clip_min in (None, 2.0):
            with self.subTest(norm_clip_min=norm_clip_min):
                normalizer = (latents - teacher).abs().mean(dim=(1, 2), keepdim=True)
                if norm_clip_min is not None:
                    normalizer = normalizer.clamp(min=norm_clip_min)
                direction = torch.nan_to_num((fake - teacher) / normalizer)
                baseline = 0.5 * F.mse_loss(latents, (latents - direction).detach())
                actual = dmd_loss(latents, fake, teacher, norm_clip_min, projected=False)
                torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    def test_projection_is_orthogonal_per_sample(self):
        direction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-2.0, 3.0], [0.0, 1.0]]])
        residual = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]])
        projected = project_dmd_direction(direction, residual)
        torch.testing.assert_close((projected * residual).sum(dim=(1, 2)), torch.zeros(2), rtol=0, atol=1e-6)
        self.assertLessEqual(projected.square().sum().item(), direction.square().sum().item())
        torch.testing.assert_close(projected[1], torch.tensor([[0.0, 3.0], [0.0, 1.0]]))

    def test_projection_is_independent_across_samples(self):
        direction = torch.tensor([[1.0, 1.0], [3.0, 2.0]])
        residual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        projected = project_dmd_direction(direction, residual)
        torch.testing.assert_close(projected, torch.tensor([[0.0, 1.0], [3.0, 0.0]]))
        for index in range(direction.shape[0]):
            single = project_dmd_direction(direction[index : index + 1], residual[index : index + 1])
            torch.testing.assert_close(single, projected[index : index + 1], rtol=0, atol=0)

    def test_zero_residual_leaves_direction_and_loss_unchanged(self):
        latents = torch.tensor([[1.0, 2.0], [3.0, -2.0]])
        teacher = torch.zeros_like(latents)
        direction = latents - teacher
        torch.testing.assert_close(project_dmd_direction(direction, torch.zeros_like(latents)), direction, rtol=0, atol=0)
        baseline = dmd_loss(latents, latents, teacher)
        projected = dmd_loss(latents, latents, teacher, projected=True)
        torch.testing.assert_close(projected, baseline, rtol=0, atol=0)

    def test_projected_loss_uses_critic_student_residual(self):
        latents = torch.tensor([[0.0, 1.0]], requires_grad=True)
        fake = torch.tensor([[1.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, -1.0]], requires_grad=True)
        loss, normalizer, direction_rms = dmd_loss_with_stats(latents, fake, teacher, normalize=False, projected=True)
        torch.testing.assert_close(loss, torch.tensor(1.0))
        torch.testing.assert_close(normalizer, torch.tensor(1.0))
        torch.testing.assert_close(direction_rms, torch.tensor(2.0).sqrt())
        self.assertFalse(normalizer.requires_grad)
        self.assertFalse(direction_rms.requires_grad)
        loss.backward()
        torch.testing.assert_close(latents.grad, torch.tensor([[0.0, 1.0]]))
        self.assertIsNone(fake.grad)
        self.assertIsNone(teacher.grad)

    def test_sum_reduction_preserves_legacy_scale_and_gradient(self):
        latents = torch.tensor([[2.0, 4.0], [1.0, 3.0]], requires_grad=True)
        fake = torch.tensor([[4.0, 0.0], [2.0, 6.0]])
        teacher = torch.zeros_like(fake)
        epsilon = 1e-6
        loss, normalizer, direction_rms = dmd_loss_with_stats(latents, fake, teacher, normalization_epsilon=epsilon, reduction="sum")
        scale = latents.detach().abs().mean(dim=1, keepdim=True)
        direction = (fake - teacher) / (scale + epsilon)
        torch.testing.assert_close(loss, direction.square().sum(dim=1).mean())
        torch.testing.assert_close(normalizer, scale.mean())
        torch.testing.assert_close(direction_rms, direction.square().mean().sqrt())
        loss.backward()
        torch.testing.assert_close(latents.grad, 2.0 * direction / latents.shape[0])

    def test_normalizer_is_computed_per_sample(self):
        latents = torch.tensor([[1.0, 1.0], [100.0, 100.0]], requires_grad=True)
        fake = torch.ones_like(latents)
        teacher = torch.zeros_like(latents)
        dmd_loss(latents, fake, teacher).backward()
        torch.testing.assert_close(latents.grad, torch.tensor([[1.0, 1.0], [0.01, 0.01]]) / latents.numel())

    def test_modalities_are_projected_and_normalized_independently(self):
        video = torch.tensor([[0.0, 1.0]], requires_grad=True)
        audio = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        fake = (torch.tensor([[1.0, 1.0]]), torch.tensor([[2.0, 3.0, 0.0]]))
        teacher = (torch.tensor([[0.0, -1.0]]), torch.tensor([[-1.0, 0.0, -1.0]]))
        options = {"normalization_epsilon": 1e-6, "projected": True}
        video_loss = dmd_loss(video, fake[0], teacher[0], **options)
        audio_loss = dmd_loss(audio, fake[1], teacher[1], **options)
        pair_loss = dmd_loss_pair((video, audio), fake, teacher, video_weight=2.0, audio_weight=3.0, **options)
        torch.testing.assert_close(pair_loss, 2.0 * video_loss + 3.0 * audio_loss, rtol=0, atol=0)
        pair_gradients = torch.autograd.grad(pair_loss, (video, audio))
        torch.testing.assert_close(pair_gradients[0], torch.autograd.grad(2.0 * video_loss, video)[0])
        torch.testing.assert_close(pair_gradients[1], torch.autograd.grad(3.0 * audio_loss, audio)[0])

    def test_bfloat16_inputs_compute_float32_statistics_and_projection(self):
        latents = torch.tensor([[0.25, 0.5, -1.0]], dtype=torch.bfloat16, requires_grad=True)
        fake = torch.tensor([[1.25, 2.0, -2.0]], dtype=torch.bfloat16)
        teacher = torch.tensor([[0.0, -1.0, 0.25]], dtype=torch.bfloat16)
        for projected in (False, True):
            with self.subTest(projected=projected):
                result = dmd_loss_with_stats(latents, fake, teacher, normalization_epsilon=1e-6, projected=projected)
                reference = dmd_loss_with_stats(latents.float(), fake.float(), teacher.float(), normalization_epsilon=1e-6, projected=projected)
                for actual, expected in zip(result, reference):
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_epsilon_keeps_zero_normalizer_finite(self):
        latents = torch.zeros(1, 3, requires_grad=True)
        fake = torch.ones_like(latents)
        teacher = torch.zeros_like(latents)
        loss, normalizer, direction_rms = dmd_loss_with_stats(latents, fake, teacher, normalization_epsilon=1e-3)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(direction_rms))
        self.assertEqual(normalizer.item(), 0.0)
        self.assertTrue(torch.isfinite(latents.grad).all())

    def test_invalid_options_are_rejected(self):
        value = torch.ones(1, 2)
        with self.assertRaisesRegex(ValueError, "epsilon"):
            dmd_loss_with_stats(value, value, value, normalization_epsilon=-1e-6)
        with self.assertRaisesRegex(ValueError, "reduction"):
            dmd_loss_with_stats(value, value, value, reduction="median")


if __name__ == "__main__":
    unittest.main()
