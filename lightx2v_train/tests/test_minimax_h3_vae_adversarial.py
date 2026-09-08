import unittest

import torch
import torch.nn.functional as F

from lightx2v_train.model_zoo.native.minimax_h3.vae_adversarial import (
    LatentConditionedFramePatchDiscriminator,
    LatentConditionedVideoPatchDiscriminator,
    discriminator_lsgan_loss,
    make_ensemble_seraena_correction_target,
    make_seraena_correction_target,
)
from lightx2v_train.trainers.vae.adversarial import VAEAdversarialObjective


class MiniMaxH3VAEAdversarialTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.discriminator = LatentConditionedVideoPatchDiscriminator(
            base_channels=4,
            channel_multipliers=(1, 2),
            temporal_strides=(1, 2),
            group_norm_groups=2,
            spectral_normalization=False,
        )
        self.real = torch.rand(2, 3, 5, 32, 40)
        self.fake = torch.rand(2, 3, 5, 32, 40)
        self.latent = torch.rand(2, 24, 2, 2, 3)

    def test_patchgan_returns_logits_from_each_layer(self):
        logits = self.discriminator(self.real, self.latent)

        self.assertEqual(len(logits), 2)
        self.assertEqual(tuple(logits[0].shape), (2, 1, 5, 16, 20))
        self.assertEqual(tuple(logits[1].shape), (2, 1, 3, 8, 10))
        self.assertTrue(all(torch.isfinite(value).all() for value in logits))

    def test_frame_patchgan_returns_per_frame_multiscale_logits(self):
        discriminator = LatentConditionedFramePatchDiscriminator(
            base_channels=4,
            channel_multipliers=(1, 2, 4),
            spatial_strides=(2, 2, 1),
            group_norm_groups=2,
            spectral_normalization=False,
        )
        logits = discriminator(self.real, self.latent)

        self.assertEqual(len(logits), 3)
        self.assertEqual(tuple(logits[0].shape), (2, 1, 5, 16, 20))
        self.assertEqual(tuple(logits[1].shape), (2, 1, 5, 8, 10))
        self.assertEqual(tuple(logits[2].shape), (2, 1, 5, 8, 10))
        self.assertTrue(all(torch.isfinite(value).all() for value in logits))

    def test_lsgan_discriminator_loss_backpropagates(self):
        real_logits = self.discriminator(self.real, self.latent)
        fake_logits = self.discriminator(self.fake, self.latent)
        result = discriminator_lsgan_loss(real_logits, fake_logits)

        result.total.backward()

        gradients = [parameter.grad for parameter in self.discriminator.parameters() if parameter.requires_grad]
        self.assertTrue(torch.isfinite(result.total))
        self.assertTrue(torch.isfinite(result.real))
        self.assertTrue(torch.isfinite(result.fake))
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients))

    def test_correction_target_is_detached_and_trains_generator_output(self):
        fake = self.fake.clone().requires_grad_(True)
        correction = make_seraena_correction_target(
            self.discriminator.eval(),
            self.real,
            fake,
            self.latent,
            correction_scale=0.02,
            normalized_clamp=3.0,
        )

        self.assertEqual(correction.target.shape, fake.shape)
        self.assertEqual(correction.correction.shape, fake.shape)
        self.assertEqual(correction.feature_distance.shape, (2,))
        self.assertEqual(correction.raw_correction_rms.shape, (2,))
        self.assertEqual(len(correction.branch_feature_distances), 1)
        self.assertFalse(correction.target.requires_grad)
        self.assertFalse(correction.correction.requires_grad)
        self.assertTrue(torch.isfinite(correction.target).all())
        self.assertTrue(torch.isfinite(correction.correction).all())
        self.assertTrue(torch.isfinite(correction.feature_distance).all())
        self.assertTrue(torch.isfinite(correction.raw_correction_rms).all())

        generator_loss = F.mse_loss(fake, correction.target)
        generator_loss.backward()

        self.assertIsNotNone(fake.grad)
        self.assertTrue(torch.isfinite(fake.grad).all())
        self.assertGreater(fake.grad.abs().sum().item(), 0.0)

    def test_ensemble_correction_combines_video_and_frame_discriminators(self):
        frame_discriminator = LatentConditionedFramePatchDiscriminator(
            base_channels=4,
            channel_multipliers=(1, 2),
            spatial_strides=(2, 2),
            group_norm_groups=2,
            spectral_normalization=False,
        )
        fake = self.fake.clone().requires_grad_(True)
        correction = make_ensemble_seraena_correction_target(
            ((self.discriminator.eval(), 1.0), (frame_discriminator.eval(), 0.75)),
            self.real,
            fake,
            self.latent,
            correction_scale=0.02,
            normalized_clamp=3.0,
        )

        self.assertEqual(len(correction.branch_feature_distances), 2)
        self.assertTrue(all(value.shape == (2,) for value in correction.branch_feature_distances))
        self.assertTrue(all(torch.isfinite(value).all() for value in correction.branch_feature_distances))

        F.mse_loss(fake, correction.target).backward()
        self.assertIsNotNone(fake.grad)
        self.assertGreater(fake.grad.abs().sum().item(), 0.0)

    def test_objective_reports_separate_discriminator_metrics(self):
        config = {
            "crop_specs": [{"crop_size": 32, "max_frames": 5, "weight": 1.0}],
            "correction_scale": 0.02,
            "discriminator_warmup_iters": 0,
            "generator_ramp_iters": 1,
            "discriminator": {
                "base_channels": 4,
                "channel_multipliers": (1, 2),
                "temporal_strides": (1, 2),
                "group_norm_groups": 2,
                "spectral_normalization": False,
            },
            "frame_discriminator": {
                "enabled": True,
                "weight": 0.75,
                "base_channels": 4,
                "channel_multipliers": (1, 2),
                "spatial_strides": (2, 2),
                "group_norm_groups": 2,
                "spectral_normalization": False,
            },
            "optimizer": {"learning_rate": 1e-4},
        }
        objective = VAEAdversarialObjective(
            config,
            device=torch.device("cpu"),
            latent_channels=24,
            gradient_accumulation_iters=1,
        )
        real = torch.rand(1, 3, 5, 32, 48)
        fake = torch.rand(1, 3, 5, 32, 48, requires_grad=True)
        latent = torch.rand(1, 24, 2, 2, 3)

        generator_loss, metrics = objective(fake, real, latent, stage_step=0)
        generator_loss.backward()

        self.assertGreater(metrics["discriminator_3d"].item(), 0.0)
        self.assertGreater(metrics["discriminator_2d"].item(), 0.0)
        self.assertGreater(metrics["adversarial_feature_distance_3d"].item(), 0.0)
        self.assertGreater(metrics["adversarial_feature_distance_2d"].item(), 0.0)
        self.assertIn("frame_discriminator", objective.state_dict())
        self.assertIsNotNone(fake.grad)
        objective.step()

    def test_frame_discriminator_is_disabled_by_default(self):
        objective = VAEAdversarialObjective(
            {
                "discriminator": {
                    "base_channels": 4,
                    "channel_multipliers": (1, 2),
                    "temporal_strides": (1, 2),
                    "group_norm_groups": 2,
                    "spectral_normalization": False,
                }
            },
            device=torch.device("cpu"),
            latent_channels=24,
            gradient_accumulation_iters=1,
        )

        self.assertIsNone(objective.frame_discriminator)
        self.assertNotIn("frame_discriminator", objective.state_dict())


if __name__ == "__main__":
    unittest.main()
