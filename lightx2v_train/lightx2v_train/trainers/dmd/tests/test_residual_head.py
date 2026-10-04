import unittest
from unittest import mock

import torch

from lightx2v_train.trainers.dmd.residual_head import NoiseBinGate, ResidualHeadConfig, TokenResidualHead


class ResidualHeadConfigTest(unittest.TestCase):
    def test_defaults_and_metadata_round_trip(self):
        config = ResidualHeadConfig.from_mapping(None)
        self.assertFalse(config.enabled)
        self.assertEqual(config.fit_steps, 5)
        self.assertEqual(config.fit_grad_accum_steps, 1)
        self.assertEqual(config.gate_mode, "full")
        self.assertEqual(ResidualHeadConfig.from_mapping(config.checkpoint_metadata()), config)

    def test_invalid_configuration_is_rejected(self):
        for mapping in (
            {"enabled": "false"},
            {"hidden_dim": 0},
            {"fit_steps": 1.5},
            {"fit_grad_accum_steps": 0},
            {"fit_grad_accum_steps": 1.5},
            {"fit_grad_accum_steps": True},
            {"noise_bins": True},
            {"min_checks": 0},
            {"learning_rate": 0},
            {"ema_decay": 1},
            {"gate_ramp": 0},
            {"gate_ramp": 2},
            {"gate_mode": "adaptive"},
            {"gate_mode": None},
            {"max_grad_norm": -1},
            {"weight_decay": -1},
            {"min_relative_improvement": 1},
            {"learning_rate": float("nan")},
            {"head_steps": 3},
        ):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                ResidualHeadConfig.from_mapping(mapping)
        with self.assertRaises(ValueError):
            ResidualHeadConfig.from_mapping("enabled")


class TokenResidualHeadTest(unittest.TestCase):
    def test_zero_initialization_preserves_fake_prediction(self):
        head = TokenResidualHead(8, 2, (1, 2, 2), 4)
        features = torch.randn(2, 10, 8, dtype=torch.bfloat16)
        fake = torch.randn(2, 2, 2, 4, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            correction = head(features, torch.tensor([0.1, 0.9]), fake.shape)
        self.assertEqual(correction.dtype, torch.float32)
        torch.testing.assert_close(correction, torch.zeros_like(fake), rtol=0, atol=0)
        torch.testing.assert_close(fake - correction, fake, rtol=0, atol=0)
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in head.parameters()))

    def test_unpatchify_matches_wan_layout_and_ignores_padding(self):
        head = TokenResidualHead(4, 2, (2, 2, 3), 3)
        grid = (2, 2, 2)
        shape = (2, 2, 4, 4, 6)
        tokens = torch.arange(2 * 10 * 24, dtype=torch.float32).reshape(2, 10, 24).requires_grad_()
        actual = head.unpatchify(tokens, shape)
        expected = torch.stack([torch.einsum("fhwpqrc->cfphqwr", item[:8].reshape(*grid, 2, 2, 3, 2)).reshape(shape[1:]) for item in tokens])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        torch.testing.assert_close(tokens.grad[:, :8], torch.ones_like(tokens.grad[:, :8]), rtol=0, atol=0)
        torch.testing.assert_close(tokens.grad[:, 8:], torch.zeros_like(tokens.grad[:, 8:]), rtol=0, atol=0)

    def test_features_and_sigma_are_detached_but_head_is_trainable(self):
        head = TokenResidualHead(4, 1, (1, 2, 2), 3)
        features = torch.randn(2, 4, 4, requires_grad=True)
        sigma = torch.tensor([0.2, 0.8], requires_grad=True)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        correction = head(features, sigma, (2, 1, 1, 4, 4))
        (correction - 1).square().mean().backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(sigma.grad)
        self.assertGreater(head.output_projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertGreater(head(features, sigma, (2, 1, 1, 4, 4)).abs().sum().item(), 0)

    def test_sigma_is_an_explicit_condition(self):
        head = TokenResidualHead(2, 1, (1, 1, 1), 2)
        with torch.no_grad():
            head.feature_projection.weight.zero_()
            head.feature_projection.bias.zero_()
            head.sigma_projection.weight.fill_(1)
            head.output_projection.weight.fill_(1)
        features = torch.zeros(2, 1, 2)
        correction = head(features, torch.tensor([0.0, 0.5]), (2, 1, 1, 1, 1))
        self.assertEqual(correction[0].item(), 0)
        self.assertGreater(correction[1].item(), 0)

    def test_invalid_shapes_and_sigmas_are_rejected(self):
        head = TokenResidualHead(4, 2, (1, 2, 2), 3)
        features = torch.zeros(2, 4, 4)
        for sigma, shape in (
            (torch.zeros(3), (2, 2, 1, 4, 4)),
            (torch.tensor([-0.1, 0.2]), (2, 2, 1, 4, 4)),
            (torch.tensor([0.1, float("nan")]), (2, 2, 1, 4, 4)),
            (0.2, (2, 2, 1, 3, 4)),
            (0.2, (2, 1, 1, 4, 4)),
            (0.2, (2, 2, 2, 4, 4)),
        ):
            with self.subTest(sigma=sigma, shape=shape), self.assertRaises(ValueError):
                head(features, sigma, shape)
        with self.assertRaises(ValueError):
            head(torch.zeros(2, 4, 3), 0.2, (2, 2, 1, 4, 4))


class NoiseBinGateTest(unittest.TestCase):
    def make_gate(self, **options):
        mapping = {"enabled": True, "noise_bins": 2, "min_checks": 3, "ema_decay": 0.0}
        mapping.update(options)
        return NoiseBinGate(ResidualHeadConfig.from_mapping(mapping), "cpu")

    def test_sparse_evidence_falls_back_and_reliable_evidence_ramps(self):
        gate = self.make_gate()
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        for round_index in range(6):
            gate.update([(0.1, torch.tensor([4.0, 4.0]), torch.tensor([2.0, 2.0]))])
            expected = max(0, min(1, (round_index - 1) * 0.25))
            self.assertEqual(gate.lambda_for(0.1).item(), expected)
            self.assertEqual(gate.lambda_for(0.8).item(), 0)
            self.assertEqual(gate.counts[0].item(), round_index + 1)
        self.assertEqual(gate.counts[1].item(), 0)

    def test_regression_or_tied_error_resets_gate_to_baseline(self):
        gate = self.make_gate(min_checks=1)
        gate.update([(0.1, 4.0, 2.0)])
        self.assertEqual(gate.lambda_for(0.1).item(), 0.25)
        gate.update([(0.1, 2.0, 4.0)])
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        gate.update([(0.1, 2.0, 1.0)])
        self.assertEqual(gate.lambda_for(0.1).item(), 0.25)
        gate.update([(0.1, 2.0, 2.0)])
        self.assertEqual(gate.lambda_for(0.1).item(), 0)

    def test_ema_initialization_and_relative_improvement_threshold(self):
        gate = self.make_gate(min_checks=1, ema_decay=0.5, min_relative_improvement=0.1)
        gate.update([(0.1, 10.0, 9.5)])
        self.assertEqual(gate.ema_fake_mse[0].item(), 10)
        self.assertEqual(gate.ema_delta[0].item(), 0.5)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        gate.update([(0.1, 10.0, 8.0)])
        self.assertEqual(gate.ema_delta[0].item(), 1.25)
        self.assertEqual(gate.lambda_for(0.1).item(), 0.25)
        gate.update([(0.1, 10.0, 12.0)])
        self.assertEqual(gate.ema_delta[0].item(), -0.375)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)

    def test_bins_are_independent_and_endpoints_have_valid_indices(self):
        gate = self.make_gate(min_checks=1)
        gate.update([(torch.tensor([0.0, 0.499, 0.5, 1.0]), torch.tensor([4.0, 2.0, 2.0, 4.0]), torch.tensor([2.0, 1.0, 3.0, 6.0]))])
        torch.testing.assert_close(gate.lambda_for(torch.tensor([0.0, 0.49, 0.5, 1.0])), torch.tensor([0.25, 0.25, 0.0, 0.0]))
        torch.testing.assert_close(gate.ema_fake_mse, torch.tensor([3.0, 3.0]))
        torch.testing.assert_close(gate.ema_delta, torch.tensor([1.5, -1.5]))
        torch.testing.assert_close(gate.counts, torch.tensor([1, 1]))

    def test_distributed_sum_produces_equal_gates_and_world_size_invariant_counts(self):
        gates = [self.make_gate(min_checks=1), self.make_gate(min_checks=1)]
        # Pooled fake/corrected sums and sample counts from two DP ranks.
        pooled = torch.tensor([[8.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
        with (
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.is_initialized", return_value=True),
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.all_reduce", side_effect=lambda value, op: value.copy_(pooled)) as all_reduce,
        ):
            gates[0].update([(0.1, 2.0, 1.0)])
            gates[1].update([(0.1, 6.0, 2.0)])
        self.assertEqual(all_reduce.call_count, 2)
        self.assertEqual(gates[0].ema_fake_mse[0].item(), 4)
        self.assertEqual(gates[0].ema_delta[0].item(), 2.5)
        self.assertEqual(gates[0].counts[0].item(), 1)
        self.assertEqual(gates[0].metrics(), gates[1].metrics())

    def test_empty_round_does_not_create_evidence_or_ramp(self):
        gate = self.make_gate(min_checks=1)
        gate.update([(0.1, 2.0, 1.0)])
        before = {key: value.clone() for key, value in gate.state_dict().items()}
        gate.update([])
        for key, value in before.items():
            torch.testing.assert_close(gate.state_dict()[key], value, rtol=0, atol=0)

    def test_disabled_gate_always_falls_back(self):
        gate = self.make_gate(enabled=False, min_checks=1)
        gate.update([(0.1, 2.0, 1.0)])
        self.assertEqual(gate.lambda_for(0.1).item(), 0)

    def test_state_round_trip_keeps_ema_counts_and_lambdas(self):
        gate = self.make_gate(min_checks=1)
        gate.update([(0.1, 4.0, 2.0), (0.9, 3.0, 2.0)])
        restored = self.make_gate(min_checks=1)
        restored.load_state_dict(gate.state_dict())
        self.assertEqual(restored.metrics(), gate.metrics())
        restored.update([(0.1, 4.0, 2.0)])
        self.assertEqual(restored.lambda_for(0.1).item(), 0.5)
        self.assertEqual(restored.lambda_for(0.9).item(), 0.25)

    def test_observations_are_detached_and_invalid_inputs_rejected(self):
        gate = self.make_gate(min_checks=1)
        fake_mse = torch.tensor([2.0, 2.0], requires_grad=True)
        corrected_mse = torch.tensor([1.0, 1.0], requires_grad=True)
        gate.update([(0.1, fake_mse, corrected_mse)])
        self.assertFalse(gate.ema_delta.requires_grad)
        self.assertFalse(gate.lambda_for(torch.tensor(0.1, requires_grad=True)).requires_grad)
        self.assertIsNone(fake_mse.grad)
        self.assertIsNone(corrected_mse.grad)
        for check in (
            (float("nan"), 1.0, 1.0),
            (1.1, 1.0, 1.0),
            (0.1, -1.0, 1.0),
            (0.1, 1.0, float("inf")),
            (torch.zeros(2), torch.ones(3), torch.ones(2)),
        ):
            with self.subTest(check=check), self.assertRaises(ValueError):
                gate.update([check])


class CalibratedNoiseBinGateTest(unittest.TestCase):
    def make_gate(self, **options):
        mapping = {"enabled": True, "noise_bins": 2, "min_checks": 1, "ema_decay": 0.0, "gate_mode": "calibrated"}
        mapping.update(options)
        return NoiseBinGate(ResidualHeadConfig.from_mapping(mapping), "cpu")

    def test_small_scale_passes_when_full_correction_is_harmful(self):
        gate = self.make_gate()
        # base=.04, dot=.1, energy=1: full correction MSE=.84,
        # but calibration-optimal lambda=.1 gives MSE=.03.
        snapshot = gate.calibrate([(0.9, 0.1, 1.0)])
        self.assertEqual(gate.lambda_for(0.9).item(), 0)
        gate.update_calibrated([(0.9, 0.04, 0.1, 1.0)], snapshot)
        self.assertAlmostEqual(gate.lambda_for(0.9).item(), 0.1)
        self.assertAlmostEqual(gate.ema_delta[1].item(), 0.01)
        baseline = NoiseBinGate(ResidualHeadConfig(enabled=True, noise_bins=2, min_checks=1))
        baseline.update([(0.9, 0.04, 0.84)])
        self.assertEqual(baseline.lambda_for(0.9).item(), 0)

    def test_calibration_does_not_open_gate_or_touch_validation(self):
        gate = self.make_gate()
        before = [value.clone() for value in (gate.counts, gate.ema_fake_mse, gate.ema_delta, gate.validation_dot, gate.validation_energy)]
        gate.calibrate([(0.1, 0.3, 1.0)])
        for actual, expected in zip((gate.counts, gate.ema_fake_mse, gate.ema_delta, gate.validation_dot, gate.validation_energy), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        self.assertEqual(gate.calibration_counts[0].item(), 1)

    def test_validation_accepts_or_rejects_selected_scale_without_refitting(self):
        gate = self.make_gate()
        snapshot = gate.calibrate([(0.1, 0.8, 1.0)])
        before = [value.clone() for value in (gate.calibration_dot, gate.calibration_energy, gate.calibration_counts, gate.candidate_lambdas)]
        # Validation would prefer .1, but the chosen .8 is harmful: reject it.
        gate.update_calibrated([(0.1, 0.1, 0.1, 1.0)], snapshot)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        for actual, expected in zip((gate.calibration_dot, gate.calibration_energy, gate.calibration_counts, gate.candidate_lambdas), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Validation would prefer 1, but may only enable the chosen .8.
        gate.update_calibrated([(0.1, 1.0, 1.0, 1.0)], snapshot)
        self.assertAlmostEqual(gate.lambda_for(0.1).item(), 0.8)

    def test_each_stream_requires_minimum_independent_rounds(self):
        gate = self.make_gate(min_checks=3)
        snapshot = gate.calibrate([(0.1, 0.1, 1.0)])
        for _ in range(3):
            gate.update_calibrated([(0.1, 0.04, 0.1, 1.0)], snapshot)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        for _ in range(2):
            snapshot = gate.calibrate([(0.1, 0.1, 1.0)])
        # Reaching calibration min_checks alone must not open the gate.
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        gate.update_calibrated([(0.1, 0.04, 0.1, 1.0)], snapshot)
        self.assertAlmostEqual(gate.lambda_for(0.1).item(), 0.1)
        self.assertEqual(gate.counts[0].item(), 4)
        self.assertEqual(gate.calibration_counts[0].item(), 3)

    def test_changed_scale_recomputes_risk_instead_of_using_old_scaled_mse(self):
        gate = self.make_gate(ema_decay=0.5)
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        self.assertAlmostEqual(gate.ema_delta[0].item(), 0.04)
        # EMA calibration dot becomes .8, but validation evidence remains .2.
        snapshot = gate.calibrate([(0.1, 1.4, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        self.assertAlmostEqual(gate.ema_delta[0].item(), -0.32, places=6)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)

    def test_unobserved_bins_never_apply_a_new_scale(self):
        gate = self.make_gate()
        snapshot = gate.calibrate([(0.1, 0.2, 1.0), (0.9, 0.4, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        self.assertAlmostEqual(gate.lambda_for(0.1).item(), 0.2)
        self.assertEqual(gate.lambda_for(0.9).item(), 0)
        snapshot = gate.calibrate([(0.1, 0.8, 1.0)])
        gate.update_calibrated([(0.9, 1.0, 0.4, 1.0)], snapshot)
        # The absent bin retains its previously validated .2, never the new .8.
        self.assertAlmostEqual(gate.lambda_for(0.1).item(), 0.2)
        self.assertAlmostEqual(gate.lambda_for(0.9).item(), 0.4)
        self.assertAlmostEqual(gate.ema_delta[0].item(), 0.04)

    def test_calibration_snapshot_is_detached_and_cannot_alias_or_be_stale(self):
        gate = self.make_gate()
        dot = torch.tensor([0.2], requires_grad=True)
        snapshot = gate.calibrate([(0.1, dot, 1.0)])
        self.assertFalse(snapshot.requires_grad)
        self.assertNotEqual(snapshot.data_ptr(), gate.candidate_lambdas.data_ptr())
        old_snapshot = snapshot.clone()
        snapshot[0] = 0.8
        self.assertAlmostEqual(gate.candidate_lambdas[0].item(), 0.2)
        with self.assertRaisesRegex(ValueError, "unchanged latest"):
            gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        gate.calibrate([(0.1, 0.5, 1.0)])
        with self.assertRaisesRegex(ValueError, "unchanged latest"):
            gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], old_snapshot)
        self.assertEqual(gate.counts.sum().item(), 0)
        self.assertIsNone(dot.grad)

    def test_distributed_stats_are_pooled_with_world_size_invariant_counts(self):
        gates = [self.make_gate(), self.make_gate()]
        pooled_calibration = torch.tensor([[0.2, 0.4], [4.0, 2.0], [2.0, 1.0]])
        pooled_validation = torch.tensor([[0.2, 0.25], [0.2, 0.4], [4.0, 2.0], [2.0, 1.0]])
        with (
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.is_initialized", return_value=True),
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.all_reduce", side_effect=lambda value, op: value.copy_(pooled_calibration)) as reduce_calibration,
        ):
            snapshots = [gates[0].calibrate([(0.1, 0.2, 3.0)]), gates[1].calibrate([(0.9, 0.4, 2.0)])]
        with (
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.is_initialized", return_value=True),
            mock.patch("lightx2v_train.trainers.dmd.residual_head.dist.all_reduce", side_effect=lambda value, op: value.copy_(pooled_validation)) as reduce_validation,
        ):
            for gate, snapshot in zip(gates, snapshots):
                gate.update_calibrated([], snapshot)
        self.assertEqual(reduce_calibration.call_count, 2)
        self.assertEqual(reduce_validation.call_count, 2)
        torch.testing.assert_close(snapshots[0], torch.tensor([0.05, 0.2]))
        torch.testing.assert_close(gates[0].lambdas, snapshots[0])
        torch.testing.assert_close(gates[0].counts, torch.tensor([1, 1]))
        torch.testing.assert_close(gates[0].calibration_counts, torch.tensor([1, 1]))
        self.assertEqual(gates[0].metrics(), gates[1].metrics())

    def test_independent_ema_means_and_relative_threshold(self):
        gate = self.make_gate(ema_decay=0.5, min_relative_improvement=0.1)
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)
        snapshot = gate.calibrate([(0.1, 0.4, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.8, 1.0)], snapshot)
        self.assertAlmostEqual(gate.calibration_dot[0].item(), 0.3)
        self.assertAlmostEqual(gate.validation_dot[0].item(), 0.5)
        self.assertAlmostEqual(gate.ema_delta[0].item(), 0.21)
        self.assertAlmostEqual(gate.lambda_for(0.1).item(), 0.3)

    def test_zero_energy_negative_dot_and_clipping_are_safe(self):
        gate = self.make_gate()
        snapshot = gate.calibrate([(0.1, 0.0, 0.0), (0.9, -1.0, 1.0)])
        torch.testing.assert_close(snapshot, torch.zeros(2))
        gate.update_calibrated([(torch.tensor([0.1, 0.9]), 1.0, 0.0, 0.0)], snapshot)
        torch.testing.assert_close(gate.lambdas, torch.zeros(2))
        snapshot = gate.calibrate([(0.9, 2.0, 1.0)])
        self.assertEqual(snapshot[1].item(), 1)

    def test_state_round_trip_preserves_both_independent_streams(self):
        gate = self.make_gate(ema_decay=0.5)
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        restored = self.make_gate(ema_decay=0.5)
        restored.load_state_dict(gate.state_dict())
        self.assertEqual(restored.metrics(), gate.metrics())
        for item in (gate, restored):
            snapshot = item.calibrate([(0.1, 0.4, 1.0)])
            item.update_calibrated([(0.1, 1.0, 0.8, 1.0)], snapshot)
        self.assertEqual(restored.metrics(), gate.metrics())
        full_gate = NoiseBinGate(ResidualHeadConfig())
        self.assertEqual(set(full_gate.state_dict()), {"ema_delta", "ema_fake_mse", "counts", "lambdas"})

    def test_empty_rounds_preserve_state(self):
        gate = self.make_gate()
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        before = {key: value.clone() for key, value in gate.state_dict().items()}
        snapshot = gate.calibrate([])
        gate.update_calibrated([], snapshot)
        for key, expected in before.items():
            torch.testing.assert_close(gate.state_dict()[key], expected, rtol=0, atol=0)

    def test_disabled_gate_always_zero(self):
        gate = self.make_gate(enabled=False)
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        gate.update_calibrated([(0.1, 1.0, 0.2, 1.0)], snapshot)
        self.assertEqual(gate.lambda_for(0.1).item(), 0)

    def test_invalid_statistics_and_mode_mixing_are_rejected(self):
        gate = self.make_gate()
        snapshot = gate.calibrate([(0.1, 0.2, 1.0)])
        for check in (
            (float("nan"), 0.1, 1.0),
            (1.1, 0.1, 1.0),
            (0.1, float("inf"), 1.0),
            (0.1, 0.1, -1.0),
            (torch.zeros(2), torch.ones(3), torch.ones(2)),
        ):
            with self.subTest(check=check), self.assertRaises(ValueError):
                gate.calibrate([check])
        for check in (
            (0.1, -1.0, 0.1, 1.0),
            (0.1, 1.0, float("nan"), 1.0),
            (0.1, 1.0, 0.1, -1.0),
            (0.1, 1.0, 0.1),
        ):
            with self.subTest(check=check), self.assertRaises(ValueError):
                gate.update_calibrated([check], snapshot)
        for invalid in (torch.ones(3), torch.tensor([float("nan"), 0.0]), torch.tensor([1.1, 0.0])):
            with self.assertRaises(ValueError):
                gate.update_calibrated([], invalid)
        with self.assertRaises(ValueError):
            gate.update([(0.1, 1.0, 0.5)])
        full = NoiseBinGate(ResidualHeadConfig())
        with self.assertRaises(ValueError):
            full.calibrate([])
        with self.assertRaises(ValueError):
            full.update_calibrated([], snapshot)


if __name__ == "__main__":
    unittest.main()
