"""CPU regression tests for the opt-in released PDMD objective."""

import unittest

import torch
import torch.nn.functional as F

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
    MiniMaxH3DistributionMatchingOptions,
)
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.trainers.dmd.math import (
    dmd_loss_with_stats,
    official_pdmd_critic_loss,
    official_pdmd_loss_with_stats,
)


def released_reference(student, teacher, critic, *, projected=True, floor=1e-5, epsilon=1e-8, clamp=5.0):
    """Independent transcription of objectives.py in official commit 6b6e106.

    Keep its endpoint_update/surrogate_loss arithmetic order, not the legacy
    fake-minus-teacher subtraction, and do not import a downloaded checkout.
    """
    with torch.no_grad():
        p_real = student.float() - teacher.float()
        p_fake = student.float() - critic.float()
        direction = p_real - p_fake
        dims = tuple(range(1, student.ndim))
        if projected:
            direction = direction - (direction * p_fake).sum(dims, keepdim=True) / (p_fake.square().sum(dims, keepdim=True) + epsilon) * p_fake
        normalizer = p_real.abs().mean(dims, keepdim=True).clamp_min(floor)
        update = direction / normalizer
        nonfinite = (~torch.isfinite(update)).sum()
        update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
    target = (student.float() - update.float()).detach()
    loss = F.mse_loss(student.float(), target)
    if clamp:
        loss = loss.clamp(0.0, clamp)
    return loss, normalizer.mean(), update.square().mean().sqrt(), nonfinite


class OfficialPDMDLossTest(unittest.TestCase):
    def test_exact_released_reference_loss_statistics_and_gradients(self):
        generator = torch.Generator().manual_seed(812)
        for dtype in (torch.float32, torch.bfloat16):
            for projected in (False, True):
                with self.subTest(dtype=dtype, projected=projected):
                    student = torch.randn(3, 2, 5, generator=generator).to(dtype).requires_grad_()
                    teacher = torch.randn(3, 2, 5, generator=generator).to(dtype).requires_grad_()
                    critic = torch.randn(3, 2, 5, generator=generator).to(dtype).requires_grad_()
                    actual = official_pdmd_loss_with_stats(student, critic, teacher, projected=projected)
                    expected = released_reference(student, teacher, critic, projected=projected)
                    for value, reference in zip(actual, expected):
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    torch.testing.assert_close(torch.autograd.grad(actual[0], student)[0], torch.autograd.grad(expected[0], student)[0], rtol=0, atol=0)
                    self.assertEqual(actual[0].dtype, torch.float32)
                    self.assertTrue(all(not value.requires_grad for value in actual[1:]))

    def test_known_projection_and_unhalved_gradient_only_reach_student(self):
        student = torch.tensor([[0.0, 1.0]], requires_grad=True)
        critic = torch.tensor([[1.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, -1.0]], requires_grad=True)
        loss, normalizer, rms, count = official_pdmd_loss_with_stats(student, critic, teacher)
        torch.testing.assert_close(loss, torch.tensor(2.0))
        torch.testing.assert_close(normalizer, torch.tensor(1.0))
        torch.testing.assert_close(rms, torch.tensor(2.0).sqrt())
        self.assertEqual(count.item(), 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([[0.0, 2.0]]))
        self.assertIsNone(critic.grad)
        self.assertIsNone(teacher.grad)

    def test_near_zero_residual_uses_additive_projection_epsilon(self):
        student = torch.zeros(1, 1, requires_grad=True)
        critic = torch.full_like(student, 1e-5)
        teacher = -torch.ones_like(student)
        actual = official_pdmd_loss_with_stats(student, critic, teacher)
        expected = released_reference(student, teacher, critic)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertGreater(actual[0].item(), 0.9)
        legacy, _, _ = dmd_loss_with_stats(student, critic, teacher, projected=True)
        self.assertLess(legacy.item(), 1e-10)

    def test_zero_residual_retains_direction_and_normalizer_is_a_floor(self):
        student = torch.zeros(1, 2, requires_grad=True)
        teacher = torch.full_like(student, -1e-7)
        loss, normalizer, rms, count = official_pdmd_loss_with_stats(student, student.detach(), teacher)
        torch.testing.assert_close(normalizer, torch.tensor(1e-5), rtol=0, atol=0)
        torch.testing.assert_close(rms, torch.tensor(0.01))
        torch.testing.assert_close(loss, torch.tensor(0.0001))
        self.assertEqual(count.item(), 0)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.full_like(student, 0.01))

    def test_nan_and_both_infinities_zero_update_without_poisoning_gradients(self):
        student = torch.zeros(1, 4, requires_grad=True)
        critic = torch.tensor([[float("nan"), float("inf"), -float("inf"), 2.0]])
        teacher = torch.ones_like(student)
        loss, _, rms, count = official_pdmd_loss_with_stats(student, critic, teacher, projected=False)
        torch.testing.assert_close(loss, torch.tensor(0.25))
        torch.testing.assert_close(rms, torch.tensor(0.5))
        self.assertEqual(count.item(), 3)
        loss.backward()
        torch.testing.assert_close(student.grad, torch.tensor([[0.0, 0.0, 0.0, 0.5]]))

    def test_projection_and_normalization_are_per_sample(self):
        student = torch.tensor([[0.0, 1.0], [0.0, 100.0]], requires_grad=True)
        critic = torch.tensor([[1.0, 1.0], [1.0, 100.0]])
        teacher = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
        batched = official_pdmd_loss_with_stats(student, critic, teacher)
        singles = [official_pdmd_loss_with_stats(student[i : i + 1], critic[i : i + 1], teacher[i : i + 1]) for i in range(2)]
        torch.testing.assert_close(batched[0], torch.stack([value[0] for value in singles]).mean())
        torch.testing.assert_close(batched[1], torch.tensor(25.5))
        gradients = torch.autograd.grad(batched[0], student)[0]
        torch.testing.assert_close(gradients, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

    def test_student_loss_cap_disables_gradient_and_zero_disables_cap(self):
        student = torch.zeros(1, 2, requires_grad=True)
        critic, teacher = torch.full_like(student, 10.0), torch.ones_like(student)
        capped = official_pdmd_loss_with_stats(student, critic, teacher, projected=False)[0]
        uncapped = official_pdmd_loss_with_stats(student, critic, teacher, projected=False, clamp_max=0)[0]
        torch.testing.assert_close(capped, torch.tensor(5.0))
        torch.testing.assert_close(uncapped, torch.tensor(81.0))
        torch.testing.assert_close(torch.autograd.grad(capped, student)[0], torch.zeros_like(student))
        torch.testing.assert_close(torch.autograd.grad(uncapped, student)[0], torch.full_like(student, 9.0))

    def test_critic_cap_target_detachment_and_cleanward_sign_equivalence(self):
        clean = torch.tensor([[3.0, 2.0]], requires_grad=True)
        noise = torch.tensor([[0.0, 1.0]], requires_grad=True)
        noise_ward = torch.tensor([[-2.0, 0.0]], requires_grad=True)
        target = noise - clean
        actual = official_pdmd_critic_loss(-noise_ward, -target)
        torch.testing.assert_close(actual, F.mse_loss(noise_ward, target.detach()))
        actual.backward()
        torch.testing.assert_close(noise_ward.grad, torch.ones_like(noise_ward))
        self.assertIsNone(clean.grad)
        self.assertIsNone(noise.grad)
        prediction = torch.full((1, 2), 10.0, requires_grad=True)
        capped = official_pdmd_critic_loss(prediction, torch.zeros_like(prediction))
        torch.testing.assert_close(capped, torch.tensor(5.0))
        capped.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_invalid_options_and_shapes_fail(self):
        value = torch.ones(1, 2)
        for kwargs in ({"normalizer_floor": 0}, {"normalizer_floor": float("nan")}, {"projection_epsilon": -1}, {"projection_epsilon": float("inf")}, {"clamp_max": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                official_pdmd_loss_with_stats(value, value, value, **kwargs)
        with self.assertRaisesRegex(ValueError, "shapes"):
            official_pdmd_loss_with_stats(value, value[:, :1], value)
        with self.assertRaisesRegex(ValueError, "shapes"):
            official_pdmd_loss_with_stats(value[0], value[0], value[0])


class OfficialPDMDH3CapabilityTest(unittest.TestCase):
    def capability(self, **options):
        self.model = model_wrapper()
        return MiniMaxH3DistributionMatchingCapability(self.model, options)

    @staticmethod
    def pair(video, audio):
        return MiniMaxH3JointLatents(video, audio, None)

    def test_opt_in_defaults_and_legacy_loss_and_metadata_are_unchanged(self):
        legacy = self.capability(projected_dmd=True)
        self.assertFalse(legacy.options.official_pdmd)
        self.assertNotIn("minimax_h3_official_pdmd_loss", legacy.extra_checkpoint_metadata())
        student = self.pair(torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[0.0, 2.0]]]))
        fake = self.pair(torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[1.0, 2.0]]]))
        teacher = self.pair(torch.tensor([[[0.0, -1.0]]]), torch.zeros(1, 1, 2))
        expected = dmd_loss_with_stats(student.video, fake.video, teacher.video, projected=True)[0] + dmd_loss_with_stats(student.audio, fake.audio, teacher.audio, projected=True)[0]
        torch.testing.assert_close(legacy.dmd_loss(student, fake, teacher), expected, rtol=0, atol=0)
        official = self.capability(official_pdmd=True)
        self.assertTrue(official.options.projected_dmd)
        metadata = official.extra_checkpoint_metadata()["minimax_h3_official_pdmd_loss"]
        self.assertEqual(metadata["normalizer_floor"], 1e-5)
        self.assertEqual(metadata["projection_epsilon"], 1e-8)
        self.assertEqual(metadata["loss_clamp"], 5.0)
        self.assertEqual(metadata["mse_scale"], 1.0)
        self.assertTrue(metadata["clamp_before_modality_weight"])

    def test_modality_clamps_precede_single_point_eight_weight(self):
        capability = self.capability(official_pdmd=True, projected_dmd=False, video_loss_weight=0.8, audio_loss_weight=0.8, audio_dmd_loss_weight=0.8)
        student = self.pair(torch.zeros(1, 1, 2, requires_grad=True), torch.zeros(1, 1, 2, requires_grad=True))
        fake = self.pair(torch.full((1, 1, 2), 10.0), torch.full((1, 1, 2), 2.0))
        teacher = self.pair(torch.ones(1, 1, 2), torch.ones(1, 1, 2))
        loss = capability.dmd_loss(student, fake, teacher)
        torch.testing.assert_close(loss, torch.tensor(4.8))
        torch.testing.assert_close(capability.dmd_metrics()["dmd_video"], torch.tensor(4.0))
        torch.testing.assert_close(capability.dmd_metrics()["dmd_audio"], torch.tensor(0.8))
        loss.backward()
        torch.testing.assert_close(student.video.grad, torch.zeros_like(student.video))
        torch.testing.assert_close(student.audio.grad, torch.full_like(student.audio, 0.8))
        prediction = self.pair(torch.full((1, 1, 2), 10.0, requires_grad=True), torch.ones(1, 1, 2, requires_grad=True))
        target = self.pair(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
        regression = capability.regression_loss(prediction, target)
        torch.testing.assert_close(regression, torch.tensor(4.8))
        regression.backward()
        torch.testing.assert_close(prediction.video.grad, torch.zeros_like(prediction.video))
        torch.testing.assert_close(prediction.audio.grad, torch.full_like(prediction.audio, 0.8))

    def test_packed_stereo_is_one_sample_and_modalities_stay_independent(self):
        capability = self.capability(official_pdmd=True, video_loss_weight=0.8, audio_dmd_loss_weight=0.8)
        student = self.pair(torch.tensor([[[0.0, 1.0]]], requires_grad=True), torch.tensor([[[0.0, 1.0], [0.0, 100.0]]], requires_grad=True))
        critic = self.pair(torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[1.0, 1.0], [0.0, 100.0]]]))
        teacher = self.pair(torch.tensor([[[0.0, -1.0]]]), torch.zeros(1, 2, 2))
        expected = 0.8 * released_reference(student.video, teacher.video, critic.video)[0] + 0.8 * released_reference(student.audio, teacher.audio, critic.audio)[0]
        actual = capability.dmd_loss(student, critic, teacher)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(capability.dmd_metrics()["dmd_audio_normalizer"], torch.tensor(25.25))

    def test_physical_sigma_pair_bypasses_shifts_only_when_explicitly_enabled(self):
        capability = self.capability(official_pdmd=True, video_flow_shift=12, audio_flow_shift=3)
        physical = torch.tensor([0.4, 0.9])
        video_sigma, audio_sigma = capability._modality_sigmas(physical)
        torch.testing.assert_close(video_sigma, physical[:1], rtol=0, atol=0)
        torch.testing.assert_close(audio_sigma, physical[1:], rtol=0, atol=0)
        scalar = torch.tensor([0.5])
        shifted = capability._modality_sigmas(scalar)
        torch.testing.assert_close(shifted[0], 12 * scalar / (1 + 11 * scalar))
        torch.testing.assert_close(shifted[1], 3 * scalar / (1 + 2 * scalar))
        sample = self.pair(torch.ones(1, 2, 3), torch.ones(1, 4, 2))
        velocity = self.pair(torch.full_like(sample.video, 2), torch.full_like(sample.audio, 3))
        clean = capability.x0_from_velocity(sample, velocity, physical)
        torch.testing.assert_close(clean.video, sample.video + 0.4 * velocity.video)
        torch.testing.assert_close(clean.audio, sample.audio + 0.9 * velocity.audio)
        legacy = self.capability()
        with self.assertRaisesRegex(ValueError, "shared base sigma"):
            legacy._modality_sigmas(physical)

    def test_official_options_reject_legacy_or_incompatible_loss_settings(self):
        for extra in (
            {"legacy_numerics": True},
            {"dmd_normalization": False},
            {"dmd_reduction": "sum"},
            {"dmd_normalization_epsilon": 1e-5},
            {"official_pdmd_normalizer_floor": 0},
            {"official_pdmd_projection_epsilon": -1},
            {"official_pdmd_loss_clamp": float("nan")},
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "Official PDMD"):
                MiniMaxH3DistributionMatchingOptions.from_mapping({"official_pdmd": True, **extra})


if __name__ == "__main__":
    unittest.main()
