"""CPU regressions for the legacy recipe and raw H3 x0 arithmetic.

Expected expressions below are independent transcriptions of the original
H3 trainer's condition noise and physical sigma formulas. No model weights,
CUDA kernels, or distributed process groups are needed.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
)
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.schedulers.dmd_scheduler import DMDFlowMatchingScheduler
from lightx2v_train.trainers.dmd.score_sampling import (
    H3ShiftedUniformScoreSigmaSampler,
    ScoreSigmaContext,
    build_score_sigma_sampler,
)

CAPABILITY_MODULE = "lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability"


def capability(legacy=True):
    return MiniMaxH3DistributionMatchingCapability(
        model_wrapper(),
        {
            "legacy_numerics": legacy,
            "video_flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    )


def old_shift(base, shift):
    return shift * base / (1.0 + (shift - 1.0) * base)


def old_score_pair(base, minimum=0.02, maximum=1.0):
    base = torch.ceil(base * 1000) / 1000
    video = old_shift(base, 12.0).clamp(minimum, maximum)
    audio = 3.0 * video / (12.0 + (3.0 - 12.0) * video)
    return torch.stack((video, audio))


def score_context():
    return ScoreSigmaContext(None, None, 1000, torch.device("cpu"), None)


class LegacyH3NumericsTest(unittest.TestCase):
    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_dtype=True)

    def test_rollout_scalar_and_physical_score_sigmas(self):
        legacy = capability()
        modern = capability(legacy=False)
        for base in (torch.tensor(0.371), torch.tensor([0.371])):
            actual = legacy._modality_sigmas(base)
            for value, shift in zip(actual, (12.0, 3.0)):
                self.assertEqual(value.ndim, 0)
                self.assert_exact(value, old_shift(base.reshape(()), shift))
        physical = old_score_pair(torch.tensor(0.371))
        video, audio = legacy._modality_sigmas(physical)
        self.assert_exact(video, physical[0])
        self.assert_exact(audio, physical[1])
        self.assertEqual(modern._modality_sigmas(torch.tensor(0.371))[0].shape, (1,))
        with self.assertRaisesRegex(ValueError, "one shared base sigma"):
            modern._modality_sigmas(physical)
        with self.assertRaisesRegex(ValueError, "one shared base sigma"):
            legacy._modality_sigmas(torch.ones(3))

    def test_raw_x0_matches_scalar_or_batched_values_and_gradients(self):
        generator = torch.Generator().manual_seed(73)
        base_samples = [torch.randn(1, 8, 16, generator=generator) for _ in range(2)]
        base_velocities = [torch.randn(1, 8, 16, generator=generator).bfloat16() for _ in range(2)]
        base_sigma = torch.tensor(0.371)
        sigmas = [old_shift(base_sigma, shift) for shift in (12.0, 3.0)]
        for dtype in (torch.float32, torch.bfloat16):
            for legacy in (False, True):
                with self.subTest(dtype=dtype, legacy=legacy):
                    samples = [value.to(dtype).clone().requires_grad_() for value in base_samples]
                    velocities = [value.clone().requires_grad_() for value in base_velocities]
                    expected_samples = [value.detach().clone().requires_grad_() for value in samples]
                    expected_velocities = [value.detach().clone().requires_grad_() for value in velocities]
                    expected = [sample + (sigma if legacy else sigma.reshape(1, 1, 1)) * velocity for sample, velocity, sigma in zip(expected_samples, expected_velocities, sigmas)]
                    actual = capability(legacy).x0_from_velocity(
                        MiniMaxH3JointLatents(*samples, None),
                        MiniMaxH3JointLatents(*velocities, None),
                        base_sigma,
                    )
                    for result, reference in zip((actual.video, actual.audio), expected):
                        self.assert_exact(result, reference)
                    sum(value.float().square().mean() for value in (actual.video, actual.audio)).backward()
                    sum(value.float().square().mean() for value in expected).backward()
                    for value, reference in zip(samples + velocities, expected_samples + expected_velocities):
                        self.assert_exact(value.grad, reference.grad)
                    # FP32 xt does not undo the BF16 rounding of sigma*v when
                    # sigma is scalar. The legacy branch must retain it.
                    old_x0 = samples[0].detach() + sigmas[0] * velocities[0].detach()
                    if legacy:
                        self.assert_exact(actual.video.detach(), old_x0)
                        fp32_x0 = samples[0].detach().float() + sigmas[0] * velocities[0].detach().float()
                        self.assertFalse(torch.equal(actual.video.detach().float(), fp32_x0))
                    else:
                        self.assertEqual(actual.video.dtype, torch.float32)
                        self.assertFalse(torch.equal(actual.video.detach(), old_x0))

    def test_raw_x0_broadcasts_batch_sigmas_without_forcing_output_dtype(self):
        sample = torch.tensor([[[0.25, -0.5]], [[1.25, -2.5]]], dtype=torch.bfloat16, requires_grad=True)
        velocity = torch.tensor([[[0.703125, 1.1015625]], [[-0.90234375, 0.30078125]]], dtype=torch.bfloat16, requires_grad=True)
        sigma = torch.tensor([0.371, 0.619], requires_grad=True)
        expected_sample = sample.detach().clone().requires_grad_()
        expected_velocity = velocity.detach().clone().requires_grad_()
        expected_sigma = sigma.detach().clone().requires_grad_()
        expected = expected_sample + expected_sigma[:, None, None] * expected_velocity
        actual = capability()._cleanward_x0(sample, velocity, sigma)
        self.assertEqual(actual.dtype, torch.float32)
        self.assert_exact(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for value, reference in zip((sample, velocity, sigma), (expected_sample, expected_velocity, expected_sigma)):
            self.assert_exact(value.grad, reference.grad)

    def test_legacy_physical_sigmas_retain_raw_scalar_x0(self):
        generator = torch.Generator().manual_seed(801)
        samples = [torch.randn(1, 8, 16, generator=generator, requires_grad=True) for _ in range(2)]
        velocities = [torch.randn(1, 8, 16, generator=generator).bfloat16().requires_grad_() for _ in range(2)]
        sigmas = torch.tensor([0.731, 0.281])
        actual = capability().x0_from_velocity(MiniMaxH3JointLatents(*samples, None), MiniMaxH3JointLatents(*velocities, None), sigmas)
        for result, sample, velocity, sigma in zip((actual.video, actual.audio), samples, velocities, sigmas):
            self.assert_exact(result, sample + sigma * velocity)
            self.assertFalse(torch.equal(result, sample + sigma * velocity.float()))

    def test_euler_step_keeps_original_fp32_arithmetic(self):
        generator = torch.Generator().manual_seed(17)
        sample_values = torch.randn(1, 8, 16, generator=generator)
        velocity_values = torch.randn(1, 8, 16, generator=generator).bfloat16()
        sigma, sigma_next = old_shift(torch.tensor(0.5), 12.0), old_shift(torch.tensor(0.375), 12.0)
        for legacy in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(legacy=legacy, dtype=dtype):
                    sample = sample_values.to(dtype).clone().requires_grad_()
                    velocity = velocity_values.clone().requires_grad_()
                    reference_sample = sample.detach().clone().requires_grad_()
                    reference_velocity = velocity.detach().clone().requires_grad_()
                    expected = (reference_sample.float() + (sigma - sigma_next) * reference_velocity.float()).to(dtype)
                    actual = capability(legacy)._cleanward_step(sample, velocity, sigma, sigma_next)
                    self.assert_exact(actual, expected)
                    actual.float().square().sum().backward()
                    expected.float().square().sum().backward()
                    self.assert_exact(sample.grad, reference_sample.grad)
                    self.assert_exact(velocity.grad, reference_velocity.grad)

    def test_reference_noise_uses_exact_old_fp32_coefficient(self):
        generator = torch.Generator().manual_seed(921)
        clean = torch.randn(24, 96, generator=generator).bfloat16()
        noise = torch.randn(24, 96, generator=generator)
        audio = torch.randn(4, 32, generator=generator).bfloat16()
        condition = {
            "condition_video_latents": clean,
            "condition_audio_latents": audio,
            "references": (
                SimpleNamespace(kind="image", num_latent_frames=1, latent_height=4, latent_width=6),
                SimpleNamespace(kind="video", num_latent_frames=3, latent_height=4, latent_width=6),
                SimpleNamespace(kind="audio"),
            ),
        }
        clean_before = clean.clone()
        scalar = torch.tensor(0.999, dtype=torch.float32)
        expected = scalar * clean.float() + (1.0 - scalar) * noise
        python_float_version = 0.999 * clean.float() + (1.0 - 0.999) * noise
        self.assertFalse(torch.equal(expected, python_float_version))
        broadcasts = []

        def broadcast(value):
            broadcasts.append(value)
            return value

        with patch(f"{CAPABILITY_MODULE}.keyframe_condition_noise", return_value=noise) as noise_call:
            actual = capability()._prepare_condition_for_rollout(condition, broadcast)
        noise_call.assert_called_once_with(((1, 4, 6), (3, 4, 6)), (1, 2, 2), 24, device=torch.device("cpu"), dtype=torch.float32)
        self.assert_exact(actual["noised_condition_video_latents"], expected)
        self.assertEqual(len(broadcasts), 1)
        self.assertIs(actual["condition_audio_latents"], audio)
        self.assertNotIn("noised_condition_video_latents", condition)
        self.assert_exact(clean, clean_before)

    def test_score_sigma_seed_sequence_matches_old_trainer(self):
        sampler = build_score_sigma_sampler(
            {"type": "h3_shifted_uniform", "legacy_numerics": True, "video_flow_shift": 12.0, "audio_flow_shift": 3.0},
            use_rollout_min=False,
            use_rollout_max=False,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(124)
            expected = [old_score_pair(torch.rand((), dtype=torch.float32)) for _ in range(64)]
            expected_next_random = torch.rand(())
            torch.manual_seed(124)
            actual = [sampler.sample(score_context()) for _ in range(64)]
            actual_next_random = torch.rand(())
        self.assert_exact(torch.stack(actual), torch.stack(expected))
        self.assert_exact(actual_next_random, expected_next_random)

    def test_score_sigma_clamp_boundaries_keep_physical_pair(self):
        sampler = H3ShiftedUniformScoreSigmaSampler(video_flow_shift=12.0, max_sigma=0.98, legacy_numerics=True)
        legacy = capability()
        for random_value in (0.0, 0.0001, 0.0011, 0.5, 0.999999):
            with self.subTest(random_value=random_value):
                base = torch.tensor(random_value, dtype=torch.float32)
                expected = old_score_pair(base, maximum=0.98)
                with patch("lightx2v_train.trainers.dmd.score_sampling.torch.rand", return_value=base) as rand:
                    actual = sampler.sample(score_context())
                rand.assert_called_once_with((), device=torch.device("cpu"), dtype=torch.float32)
                self.assert_exact(actual, expected)
                video, audio = legacy._modality_sigmas(actual)
                self.assert_exact(video, expected[0])
                self.assert_exact(audio, expected[1])
                self.assertGreaterEqual(video.item(), torch.tensor(0.02).item())
                self.assertLessEqual(video.item(), torch.tensor(0.98).item())

    def test_all_discrete_score_grid_points_equal_old_physical_sigmas(self):
        grid = torch.arange(1001, dtype=torch.float32) / 1000
        for maximum in (0.98, 1.0):
            with self.subTest(maximum=maximum):
                sampler = H3ShiftedUniformScoreSigmaSampler(video_flow_shift=12.0, max_sigma=maximum, legacy_numerics=True)
                with patch("lightx2v_train.trainers.dmd.score_sampling.torch.rand", side_effect=list(grid)):
                    actual = torch.stack([sampler.sample(score_context()) for _ in grid])
                expected = torch.stack([old_score_pair(base, maximum=maximum) for base in grid])
                self.assert_exact(actual, expected)

    def test_checkpoint_records_numerics_and_preserves_old_default_resume(self):
        # Reuse a lightweight checkpoint owner, without loading model weights.
        from lightx2v_train.trainers.dmd.tests import test_dmd_checkpoint_metadata as checkpoint_helpers

        helpers = checkpoint_helpers.DmdCheckpointMetadataTest()
        for enabled in (False, True):
            with self.subTest(legacy_numerics=enabled):
                current = capability(enabled)
                self.assertIs(current.extra_checkpoint_metadata()["minimax_h3_legacy_numerics"], enabled)
                self.assertIs(current.legacy_extra_checkpoint_metadata()["minimax_h3_legacy_numerics"], False)
                owner, manager, state = helpers.fixture(legacy_numerics=enabled)
                helpers.validate(manager, state)
                state.pop("minimax_h3_legacy_numerics")
                if not enabled:
                    helpers.validate(manager, state)
                else:
                    with self.assertRaisesRegex(RuntimeError, "minimax_h3_legacy_numerics.*allow_distribution_matching_transition"):
                        helpers.validate(manager, state)
                    owner.config["resume"] = {"allow_distribution_matching_transition": True}
                    helpers.validate(manager, state)

    def test_eight_step_rollout_schedule_equals_old_cpu_schedule(self):
        config = {"model": {"running_dtype": "bf16"}, "scheduler": {"time_shift_settings": {"do_time_shift": False}}}
        with patch("lightx2v_train.schedulers.flow_matching.get_device", return_value=torch.device("cpu")):
            scheduler = DMDFlowMatchingScheduler(config)
        scheduler.set_timesteps(8, device="cpu")
        old_base = torch.linspace(1.0, 0.0, 9, dtype=torch.float32)
        self.assert_exact(scheduler.sigmas, old_base)
        old_video, old_audio = old_shift(old_base, 12.0), old_shift(old_base, 3.0)
        legacy = capability()
        for step in range(9):
            actual_video, actual_audio = legacy._modality_sigmas(scheduler.sigma_at(step))
            self.assert_exact(actual_video, old_video[step])
            self.assert_exact(actual_audio, old_audio[step])


if __name__ == "__main__":
    unittest.main()
