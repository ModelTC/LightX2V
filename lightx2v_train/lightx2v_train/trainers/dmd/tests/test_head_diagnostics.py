import unittest

import torch

from lightx2v_train.trainers.dmd.head_diagnostics import (
    CANDIDATE_LAMBDAS,
    residual_risk_statistics,
    student_direction_statistics,
)


class HeadDiagnosticsTest(unittest.TestCase):
    def test_paired_risk_identity_and_candidate_optimum(self):
        generated = torch.tensor([[1.0, -1.0], [2.0, 3.0], [0.0, 0.0]])
        residual = torch.tensor([[1.0, 2.0], [1.0, -1.0], [1.0, 2.0]])
        correction = torch.tensor([[2.0, 4.0], [-1.0, 1.0], [0.5, 1.0]])
        fake = generated + residual
        applied_lambda = torch.tensor([0.25, 0.0, 0.75])
        stats = residual_risk_statistics(fake, generated, correction, applied_lambda)
        torch.testing.assert_close(stats["fake_mse"], residual.square().mean(dim=1))
        torch.testing.assert_close(stats["residual_head_dot"], (residual * correction).mean(dim=1))
        torch.testing.assert_close(stats["head_energy"], correction.square().mean(dim=1))
        torch.testing.assert_close(
            stats["full_corrected_mse"],
            stats["fake_mse"] - 2 * stats["residual_head_dot"] + stats["head_energy"],
        )
        torch.testing.assert_close(
            stats["selected_lambda_mse"],
            stats["fake_mse"] - 2 * applied_lambda * stats["residual_head_dot"] + applied_lambda.square() * stats["head_energy"],
        )
        for suffix, candidate in CANDIDATE_LAMBDAS:
            torch.testing.assert_close(
                stats[f"candidate_mse_{suffix}"],
                (residual - candidate * correction).square().mean(dim=1),
            )
        torch.testing.assert_close(stats["optimal_lambda"], torch.tensor([0.5, 0.0, 1.0]))
        torch.testing.assert_close(stats["optimal_lambda_mse"], torch.tensor([0.0, 1.0, 0.625]))
        torch.testing.assert_close(stats["optimal_lambda_valid"], torch.ones(3))
        self.assertTrue(all((stats["optimal_lambda_mse"] <= stats[f"candidate_mse_{suffix}"]).all() for suffix, _ in CANDIDATE_LAMBDAS))

    def test_correct_subtraction_sign_and_frozen_applied_lambda(self):
        fake = torch.tensor([[2.0, 4.0]])
        target = torch.zeros_like(fake)
        correction = fake.clone()
        stats = residual_risk_statistics(fake, target, correction, 0.25)
        torch.testing.assert_close(stats["full_corrected_mse"], torch.zeros(1))
        torch.testing.assert_close(stats["selected_lambda_mse"], torch.tensor([5.625]))
        torch.testing.assert_close(stats["optimal_lambda"], torch.ones(1))
        # The counterfactual optimum must never replace the supplied gate value.
        torch.testing.assert_close(stats["lambda_actual"], torch.tensor([0.25]))

    def test_zero_lambda_is_identity_and_zero_head_has_no_optimum(self):
        fake = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        zero = torch.zeros_like(fake)
        correction = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
        stats = residual_risk_statistics(fake, zero, correction, 0)
        torch.testing.assert_close(stats["selected_lambda_mse"], stats["fake_mse"], rtol=0, atol=0)
        directions = student_direction_statistics(fake, zero, correction, 0)
        torch.testing.assert_close(directions["direction_corrected_rms"], directions["direction_raw_rms"], rtol=0, atol=0)
        torch.testing.assert_close(directions["direction_cosine"], torch.ones(2))
        torch.testing.assert_close(directions["direction_angle_degrees"], torch.zeros(2))
        torch.testing.assert_close(directions["correction_direction_norm_ratio"], torch.zeros(2))
        stats = residual_risk_statistics(fake, zero, zero, 1)
        torch.testing.assert_close(stats["optimal_lambda"], torch.zeros(2))
        torch.testing.assert_close(stats["optimal_lambda_valid"], torch.zeros(2))
        torch.testing.assert_close(stats["optimal_lambda_mse"], stats["fake_mse"])

    def test_direction_right_angle_and_anti_alignment(self):
        fake = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        teacher = torch.zeros_like(fake)
        correction = torch.tensor([[1.0, -1.0], [2.0, 0.0]])
        stats = student_direction_statistics(fake, teacher, correction, 1)
        torch.testing.assert_close(stats["direction_cosine"], torch.tensor([0.0, -1.0]))
        torch.testing.assert_close(stats["direction_angle_degrees"], torch.tensor([90.0, 180.0]))
        torch.testing.assert_close(stats["correction_direction_norm_ratio"], torch.tensor([2.0**0.5, 2.0]))
        torch.testing.assert_close(stats["direction_cosine_valid"], torch.ones(2))
        torch.testing.assert_close(stats["correction_direction_norm_ratio_valid"], torch.ones(2))

    def test_zero_directions_are_finite_and_validity_is_explicit(self):
        fake = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        teacher = torch.zeros_like(fake)
        correction = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        stats = student_direction_statistics(fake, teacher, correction, 1)
        self.assertTrue(all(torch.isfinite(value).all() for value in stats.values()))
        torch.testing.assert_close(stats["direction_cosine_valid"], torch.zeros(3))
        torch.testing.assert_close(stats["direction_angle_degrees"], torch.zeros(3))
        torch.testing.assert_close(stats["correction_direction_norm_ratio_valid"], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(stats["correction_direction_norm_ratio"], torch.tensor([0.0, 1.0, 0.0]))

    def test_fp32_detached_metrics_preserve_inputs_and_rng(self):
        fake = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True)
        generated = torch.tensor([[0.25, 0.5], [1.0, 1.5]], dtype=torch.bfloat16, requires_grad=True)
        correction = torch.tensor([[0.5, -1.0], [1.5, 2.0]], dtype=torch.bfloat16, requires_grad=True)
        lam = torch.tensor([0.25, 0.75], requires_grad=True)
        values = (fake, generated, correction, lam)
        originals = [value.detach().clone() for value in values]
        rng_before = torch.random.get_rng_state()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            risk = residual_risk_statistics(fake, generated, correction, lam)
            directions = student_direction_statistics(fake, generated, correction, lam)
        for stats in (risk, directions):
            for value in stats.values():
                self.assertEqual(value.shape, (2,))
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(value.device, fake.device)
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
        for value, original in zip(values, originals):
            torch.testing.assert_close(value.detach(), original, rtol=0, atol=0)
            self.assertIsNone(value.grad)
        torch.testing.assert_close(rng_before, torch.random.get_rng_state(), rtol=0, atol=0)
        # Diagnostics have not damaged the independent student computation.
        generated.float().square().mean().backward()
        self.assertIsNotNone(generated.grad)
        self.assertIsNone(fake.grad)
        self.assertIsNone(correction.grad)
        self.assertIsNone(lam.grad)

    def test_expanded_lambda_and_latent_shapes(self):
        fake = torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 2, 2)
        zero = torch.zeros_like(fake)
        head = fake / 2
        flat_lambda = torch.tensor([0.25, 0.75])
        expanded_lambda = flat_lambda.reshape(2, 1, 1, 1, 1)
        for function in (residual_risk_statistics, student_direction_statistics):
            flat = function(fake, zero, head, flat_lambda)
            expanded = function(fake, zero, head, expanded_lambda)
            for name in flat:
                torch.testing.assert_close(flat[name], expanded[name], rtol=0, atol=0)

    def test_shape_errors_do_not_silently_broadcast_examples(self):
        fake = torch.ones(2, 3)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            residual_risk_statistics(fake, torch.zeros(1, 3), fake, 0)
        with self.assertRaisesRegex(ValueError, "one value per example"):
            student_direction_statistics(fake, fake, fake, torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
