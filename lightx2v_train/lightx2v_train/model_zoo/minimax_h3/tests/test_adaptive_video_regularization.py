import unittest

import torch

from lightx2v_train.model_zoo.minimax_h3.adaptive_video_regularization import (
    AdaptiveVideoRegularizer,
)


class AdaptiveVideoRegularizerTest(unittest.TestCase):
    def test_disabled_is_backward_compatible(self):
        regularizer = AdaptiveVideoRegularizer({}, 4)
        self.assertFalse(regularizer.enabled)
        self.assertFalse(regularizer.regression_enabled)
        self.assertFalse(regularizer.audio_regression_enabled)
        self.assertFalse(regularizer.temporal_enabled)

    def test_audio_regression_is_independent_and_defaults_off(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {"enabled": False},
                "audio_regression": {"enabled": True, "weight": 0.5},
                "temporal": {"enabled": False},
            },
            4,
        )
        self.assertFalse(regularizer.regression_enabled)
        self.assertTrue(regularizer.audio_regression_enabled)
        self.assertEqual(regularizer.audio_regression.weight, 0.5)

        with self.assertRaisesRegex(ValueError, "audio_regression.weight"):
            AdaptiveVideoRegularizer(
                {
                    "enabled": True,
                    "regression": {"enabled": False},
                    "audio_regression": {"enabled": True, "weight": -0.1},
                    "temporal": {"enabled": False},
                },
                4,
            )

    def test_adaptive_regression_uses_previous_ema_and_roundtrips_state(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {
                    "enabled": True,
                    "weight": 2.0,
                    "ema_decay": 0.95,
                    "sigmoid_scale": 3.0,
                },
                "temporal": {"enabled": False},
            },
            4,
        )
        raw = torch.tensor(2.0, requires_grad=True)
        weighted, adaptive_weight = regularizer.regression_loss(raw, 2, 2.0)
        self.assertAlmostEqual(adaptive_weight.item(), 0.5)
        self.assertAlmostEqual(weighted.item(), 2.0)
        self.assertIsNone(regularizer.regression_ema[2])
        regularizer.commit_regression_ema()
        self.assertAlmostEqual(regularizer.regression_ema[2], 2.0)

        restored = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {"enabled": True},
                "temporal": {"enabled": False},
            },
            4,
        )
        restored.load_state_dict(regularizer.state_dict())
        self.assertEqual(restored.regression_ema, regularizer.regression_ema)
        self.assertEqual(restored.regression_updates, regularizer.regression_updates)

    def test_adaptive_weight_uses_local_loss_but_ema_uses_global_mean(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {
                    "enabled": True,
                    "weight": 2.0,
                    "ema_decay": 0.95,
                    "sigmoid_scale": 3.0,
                },
                "temporal": {"enabled": False},
            },
            4,
        )
        regularizer.regression_loss(torch.tensor(2.0), 1, 2.0)
        regularizer.commit_regression_ema()
        _, adaptive_weight = regularizer.regression_loss(
            torch.tensor(4.0),
            1,
            # Simulate other data-parallel samples balancing the global mean.
            2.0,
        )
        expected = 1.0 - torch.sigmoid(torch.tensor(3.0 * (4.0 - 2.0)))
        self.assertAlmostEqual(adaptive_weight.item(), expected.item())
        regularizer.commit_regression_ema()
        self.assertAlmostEqual(regularizer.regression_ema[1], 2.0)

    def test_gradient_accumulation_commits_one_mean_ema_update(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {"enabled": True, "ema_decay": 0.95},
                "temporal": {"enabled": False},
            },
            4,
        )
        _, first_weight = regularizer.regression_loss(torch.tensor(2.0), 3, 2.0)
        _, second_weight = regularizer.regression_loss(torch.tensor(4.0), 3, 4.0)
        self.assertAlmostEqual(first_weight.item(), 0.5)
        self.assertAlmostEqual(second_weight.item(), 0.5)
        regularizer.commit_regression_ema()
        self.assertAlmostEqual(regularizer.regression_ema[3], 3.0)
        self.assertEqual(regularizer.regression_updates[3], 1)

    def test_checkpoint_rejects_uncommitted_microbatch_state(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {"enabled": True},
                "temporal": {"enabled": False},
            },
            4,
        )
        regularizer.regression_loss(torch.tensor(1.0), 0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "middle of a student optimizer step"):
            regularizer.state_dict()
        regularizer.commit_regression_ema()
        self.assertEqual(regularizer.state_dict()["regression_updates"][0], 1)

    def test_temporal_gate(self):
        regularizer = AdaptiveVideoRegularizer(
            {
                "enabled": True,
                "regression": {"enabled": False},
                "temporal": {
                    "enabled": True,
                    "weight": 0.05,
                    "epsilon": 1.0e-6,
                    "loss_threshold": 0.6,
                    "compute_dtype": "fp32",
                },
            },
            4,
        )
        low_motion = torch.tensor([[[0.0], [0.01], [0.0], [0.01]]], requires_grad=True)
        weighted, raw, variance = regularizer.temporal_loss(low_motion, 4)
        self.assertGreater(raw.item(), 0.6)
        self.assertGreater(weighted.item(), 0.0)
        self.assertGreater(variance.item(), 0.0)

        high_motion = torch.tensor([[[0.0], [4.0], [-4.0], [2.0]]], requires_grad=True)
        weighted, raw, _ = regularizer.temporal_loss(high_motion, 4)
        self.assertLess(raw.item(), 0.6)
        self.assertEqual(weighted.item(), 0.0)
