"""Analytic checks for Turbo math, velocity conventions and joint forward JVP."""

import math
import unittest

import torch

from lightx2v_train.trainers.turbo.core import (
    critic_loss,
    dcm_loss,
    dcm_schedule,
    dmd_loss,
    postprocess_trig_jvp,
    prediction_from_cleanward_velocity,
    preprocess_trig_jvp,
    rf_to_trig_time,
    sampled_trig_time,
    scm_direction,
    scm_loss,
    scm_tangent,
    shift_rf_time,
    trig_corrupt,
    trig_euler_step,
    trig_to_rf_state,
    trig_to_rf_time,
)


class TurboCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_rf_trig_round_trip_and_endpoints(self):
        time = torch.tensor([0.0, 0.02, 0.2, 0.7, 0.99, 1.0], dtype=torch.float64)
        theta = rf_to_trig_time(time)
        torch.testing.assert_close(trig_to_rf_time(theta), time)
        self.assertEqual(float(theta[0]), 0)
        self.assertEqual(float(theta[-1]), math.pi / 2)
        torch.testing.assert_close(sampled_trig_time(torch.tensor([0.0])), torch.tensor([math.atan(math.exp(-0.8))]))

    def test_cleanward_velocity_conversion_including_zero_noise(self):
        clean, noise = torch.randn(2, 3, 5, dtype=torch.float64), torch.randn(2, 3, 5, dtype=torch.float64)
        for theta in (torch.tensor([0.0, math.pi / 2]), torch.tensor([0.3, 1.1])):
            theta = theta.double()
            cosine, sine = theta.cos()[:, None, None], theta.sin()[:, None, None]
            state = trig_corrupt(clean, noise, theta)
            prediction = prediction_from_cleanward_velocity(state, theta, clean - noise)
            torch.testing.assert_close(prediction.x0, clean)
            torch.testing.assert_close(prediction.F, -sine * clean + cosine * noise)
            sigma = trig_to_rf_time(theta)[:, None, None]
            torch.testing.assert_close(trig_to_rf_state(state, theta), (1 - sigma) * clean + sigma * noise)

    def test_trig_coefficients_preserve_source_fp64(self):
        clean, noise = torch.randn(2, 3), torch.randn(2, 3)
        theta = torch.tensor([0.3, 1.1], dtype=torch.float64)
        state = trig_corrupt(clean, noise, theta)
        self.assertEqual(state.dtype, torch.float64)
        prediction = prediction_from_cleanward_velocity(state, theta, clean - noise)
        self.assertEqual(prediction.x0.dtype, torch.float64)
        self.assertEqual(prediction.F.dtype, torch.float64)
        torch.testing.assert_close(prediction.x0, clean.double())

    def test_shared_shifted_path_has_modality_specific_trig_tangents(self):
        base = torch.tensor([0.1, 0.3, 0.7, 0.97], dtype=torch.float64)
        for shift in (1.0, 3.0, 6.0, 12.0):
            theta, theta_tangent = torch.func.jvp(lambda u: rf_to_trig_time(shift_rf_time(u, shift)), (base,), (base * (1 - base),))
            torch.testing.assert_close(theta_tangent, theta.cos() * theta.sin())

    def test_jvp_conversion_preserves_cross_modal_derivatives(self):
        video, audio = torch.randn(1, 2, 3, dtype=torch.float64), torch.randn(1, 4, 2, dtype=torch.float64)
        times = torch.tensor([0.8, 0.5], dtype=torch.float64)
        video_tangent, audio_tangent = video * 0.2, audio * -0.3
        time_tangent = times.cos() * times.sin()

        def native(video_rf, audio_rf, sigmas):
            return (
                video_rf.square() + 0.7 * audio_rf.mean() + sigmas[0] * sigmas[1],
                audio_rf.sin() + 0.2 * video_rf.mean() + sigmas[0].square() - sigmas[1],
            )

        def whole(video_trig, audio_trig, theta):
            velocities = native(trig_to_rf_state(video_trig, theta[0]), trig_to_rf_state(audio_trig, theta[1]), trig_to_rf_time(theta))
            return (
                prediction_from_cleanward_velocity(video_trig, theta[0], velocities[0]).F,
                prediction_from_cleanward_velocity(audio_trig, theta[1], velocities[1]).F,
            )

        reference, reference_tangent = torch.func.jvp(whole, (video, audio, times), (video_tangent, audio_tangent, time_tangent))
        v_rf, v_sigma, v_rf_t, v_sigma_t = preprocess_trig_jvp(video, times[0], video_tangent, time_tangent[0])
        a_rf, a_sigma, a_rf_t, a_sigma_t = preprocess_trig_jvp(audio, times[1], audio_tangent, time_tangent[1])
        velocities, velocity_tangents = torch.func.jvp(native, (v_rf, a_rf, torch.stack((v_sigma, a_sigma))), (v_rf_t, a_rf_t, torch.stack((v_sigma_t, a_sigma_t))))
        for i, (state, tangent) in enumerate(((video, video_tangent), (audio, audio_tangent))):
            output, output_tangent = postprocess_trig_jvp(state, times[i], velocities[i], tangent, time_tangent[i], velocity_tangents[i])
            torch.testing.assert_close(output.F, reference[i])
            torch.testing.assert_close(output_tangent.F, reference_tangent[i])

        # Cross-modal terms must contribute to the video JVP.
        _, isolated = torch.func.jvp(whole, (video, audio, times), (video_tangent, torch.zeros_like(audio_tangent), torch.tensor([time_tangent[0], 0.0])))
        self.assertGreater(float((reference_tangent[0] - isolated[0]).abs().max()), 1e-4)

    def test_dcm_interval_and_euler_follow_upstream(self):
        uniform = torch.tensor([0.25, 0.75], dtype=torch.float64)
        schedule = dcm_schedule(uniform, total_steps=48, skip_steps=2, shift=6.0)
        self.assertEqual(len(schedule), 3)
        for k, theta in enumerate(schedule):
            rf = 1 - (uniform * (1 - 2 / 48) + k / 48)
            torch.testing.assert_close(theta, torch.atan2(6 * rf, 1 - rf))
        self.assertTrue(bool((schedule[0] > schedule[-1]).all()))
        state, field = torch.randn(2, 3), torch.randn(2, 3)
        torch.testing.assert_close(trig_euler_step(state, field, schedule[0], schedule[1]), state - (schedule[0] - schedule[1])[:, None] * field)
        prediction = torch.randn(2, 3, requires_grad=True)
        target = torch.randn(2, 3, requires_grad=True)
        loss = dcm_loss(prediction, target, scale=2)
        torch.testing.assert_close(loss, 2 * (prediction - target).square().sum(dim=1).mean())
        loss.backward()
        self.assertIsNone(target.grad)
        torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - target.detach()))

    def test_scm_perfect_consistency_has_zero_direction(self):
        clean, noise = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64)
        theta = torch.tensor([0.4, 0.8], dtype=torch.float64)
        state = trig_corrupt(clean, noise, theta)

        def field(value, time):
            return (time.cos()[:, None] * value - clean) / time.sin()[:, None]

        teacher = field(state, theta)
        state_tangent, time_tangent = scm_tangent(state, teacher, theta)
        primal, tangent = torch.func.jvp(field, (state, theta), (state_tangent, time_tangent))
        direction = scm_direction(state, primal, teacher, tangent, theta)
        torch.testing.assert_close(direction, torch.zeros_like(direction), atol=1e-12, rtol=0)

    def test_scm_formula_normalization_and_gradient(self):
        state, student, teacher, jvp = (torch.randn(2, 3, dtype=torch.float64) for _ in range(4))
        student.requires_grad_(True)
        theta, ratio = torch.tensor([0.2, 1.1], dtype=torch.float64), 0.3
        cosine, sine = theta.cos()[:, None], theta.sin()[:, None]
        expected = -cosine * torch.sqrt(1 - ratio**2 * sine.square()) * (student.detach() - teacher) - (ratio * cosine * sine * state + jvp)
        expected /= expected.norm(dim=1, keepdim=True) + 0.1
        direction = scm_direction(state, student, teacher, jvp, theta, warmup_ratio=ratio)
        torch.testing.assert_close(direction, expected)
        self.assertFalse(direction.requires_grad)
        loss = scm_loss(student, direction, scale=7)
        loss.backward()
        torch.testing.assert_close(student.grad, -7 * expected)

    def test_dmd_is_ordinary_source_direction_and_detaches_scorers(self):
        clean = torch.tensor([[2.0, 1.0], [1.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[1.0, -1.0], [1.0, 0.0]], requires_grad=True)
        fake = torch.tensor([[4.0, 2.0], [1.0, 0.0]], requires_grad=True)
        loss = dmd_loss(clean, fake, teacher)
        weight = (clean.detach().double() - teacher.detach()).abs().mean(dim=1, keepdim=True).clamp_min(1e-5)
        expected = (fake.detach() - teacher.detach()) / weight
        torch.testing.assert_close(loss, expected.square().sum(dim=1).mean())
        loss.backward()
        torch.testing.assert_close(clean.grad.double(), expected)
        self.assertIsNone(fake.grad)
        self.assertIsNone(teacher.grad)
        # Fake direction parallel to the teacher residual remains nonzero.
        self.assertGreater(float(clean.grad[0].abs().sum()), 0)

    def test_critic_uses_sine_squared_weight_and_detaches_generator(self):
        clean = torch.randn(2, 3, requires_grad=True)
        fake = torch.randn(2, 3, requires_grad=True)
        theta = torch.tensor([0.5, 0.9])
        expected = ((clean - fake).square() / theta.sin()[:, None].square()).sum(dim=1).mean()
        loss = critic_loss(clean, fake, theta)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertTrue(bool(torch.isfinite(fake.grad).all()))


if __name__ == "__main__":
    unittest.main()
