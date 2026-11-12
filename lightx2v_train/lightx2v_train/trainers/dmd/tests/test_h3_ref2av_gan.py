"""Ref2AV GAN conditioning, native target features and role gradients on CPU."""

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from lightx2v_train.model_zoo.native.minimax_h3.modeling import install_minimax_h3_dmad_features
from lightx2v_train.trainers.dmad.tests import test_dmad as native_fixtures
from lightx2v_train.trainers.dmd.dmd2_gan import Dmd2GanObjective, validate_gan_config
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead, frozen_dmd2_gan_critic


class Ref2AVGanConfigTests(unittest.TestCase):
    def config(self, *, model="minimax_h3_ref2av", metadata=True, shapes=()):
        return {
            "model": {"name": model, "capabilities": {"distribution_matching": {"geometry_from_metadata": metadata}}},
            "training": {
                "fake": {"train_type": "full"},
                "dmd": {"official_pdmd": True, "generation_shapes": list(shapes), "gan": {"enabled": True, "real_cache": "positive/metadata.jsonl"}},
            },
        }

    def test_ref2av_accepts_metadata_or_fixed_target_geometry(self):
        validate_gan_config(self.config())
        config = self.config()
        config["training"]["dmd"]["generation_shapes"] = None
        validate_gan_config(config)
        validate_gan_config(self.config(metadata=False, shapes=[{"value": [124, 768, 1344]}]))

    def test_geometry_and_backbone_constraints_remain_explicit(self):
        for config in (
            self.config(metadata=False),
            self.config(model="minimax_h3_t2av"),
            self.config(model="minimax_h3_image"),
            self.config(shapes=[{"value": [124, 768, 1344]}, {"value": [124, 1344, 768]}]),
        ):
            with self.subTest(model=config["model"]["name"], shapes=config["training"]["dmd"]["generation_shapes"]):
                with self.assertRaisesRegex(ValueError, "generation geometry"):
                    validate_gan_config(config)
        config = self.config()
        config["training"]["fake"]["train_type"] = "lora"
        with self.assertRaisesRegex(ValueError, "full fake-score"):
            validate_gan_config(config)


class Ref2AVGanNativeTests(unittest.TestCase):
    def fixture(self, *, checkpointing=False, image_only=False):
        transformer, capability, clean, condition = native_fixtures.DmadH3FeaturesTest().fixture(checkpointing=checkpointing)
        capability.options = replace(capability.options, official_pdmd=True)
        if image_only:
            condition["references"] = condition["references"][:1]
            del condition["condition_audio_latents"]
        trainer = SimpleNamespace(
            config={"seed": 42, "model": {"name": "minimax_h3_ref2av"}},
            dmd_config={"gan": {"feature_block": 1}},
            student=capability,
            fake=capability,
            fake_trainable_params=list(transformer.parameters()),
            dmd2_gan_head=H3GanHead(16, hidden_dim=8, temporal_bins=4),
        )
        trainer.dmd2_gan_trainable_params = list(trainer.dmd2_gan_head.parameters())
        real_condition = copy.deepcopy(condition)
        real_condition["noised_condition_video_latents"] = -3 * real_condition["noised_condition_video_latents"]
        real_condition["prompt_embeds"] = 2 * real_condition["prompt_embeds"]
        real = SimpleNamespace(load=Mock(return_value=(capability.detach(clean), real_condition)))
        objective = Dmd2GanObjective(trainer, real)
        return transformer, capability, clean, condition, trainer, real, objective

    def test_native_feature_dispatch_preserves_predictions_and_excludes_reference_tokens(self):
        transformer, capability, clean, condition, *_ = self.fixture(image_only=True)
        sigma = torch.tensor([0.4, 0.3])
        captured = []
        handle = transformer.transformer_blocks[1].register_forward_hook(lambda _module, _args, output: captured.append(output))
        try:
            before = capability.predict_velocity(clean, sigma, condition)
        finally:
            handle.remove()
        install_minimax_h3_dmad_features(transformer)
        features = capability.predict_dmad_features(clean, sigma, condition, feature_block=1)
        after = capability.predict_velocity(clean, sigma, condition)
        torch.testing.assert_close(after.video, before.video, rtol=0, atol=0)
        torch.testing.assert_close(after.audio, before.audio, rtol=0, atol=0)
        layout = capability._layout(condition, clean.shape)
        self.assertGreater(layout.num_condition_video_rows, 0)
        indices = layout.video_indices[layout.num_condition_video_rows :]
        torch.testing.assert_close(features["video"], captured[0].index_select(1, indices), rtol=0, atol=0)
        self.assertEqual(features["video"].shape[:2], clean.video.shape[:2])
        changed = dict(condition, noised_condition_video_latents=condition["noised_condition_video_latents"] + 5)
        changed_features = capability.predict_dmad_features(clean, sigma, changed, feature_block=1)
        self.assertFalse(torch.equal(changed_features["video"], features["video"]))

    def test_generator_reaches_video_and_audio_through_frozen_native_ref_critic(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                transformer, _, clean, condition, trainer, _, objective = self.fixture(checkpointing=checkpointing)
                install_minimax_h3_dmad_features(transformer)
                pooled = []
                handle = trainer.dmd2_gan_head.register_forward_pre_hook(lambda _module, inputs: pooled.append(inputs[0]))
                try:
                    with frozen_dmd2_gan_critic(trainer):
                        loss, _ = objective.generator(clean, condition, 5, 0)
                        loss.backward()
                finally:
                    handle.remove()
                self.assertEqual(pooled[0].shape[:2], clean.video.shape[:2])
                for latent in (clean.video, clean.audio):
                    self.assertIsNotNone(latent.grad)
                    self.assertTrue(torch.isfinite(latent.grad).all())
                    self.assertGreater(float(latent.grad.abs().sum()), 0)
                self.assertTrue(all(parameter.grad is None for parameter in [*trainer.fake_trainable_params, *trainer.dmd2_gan_trainable_params]))
                self.assertTrue(all(parameter.requires_grad for parameter in trainer.fake_trainable_params))

    def test_discriminator_uses_matching_live_geometry_and_each_samples_own_references(self):
        transformer, capability, clean, condition, trainer, real, objective = self.fixture(image_only=True)
        install_minimax_h3_dmad_features(transformer)
        calls = []
        feature_forward = capability.predict_dmad_features

        def features(latents, sigmas, own_condition, feature_block):
            calls.append(own_condition)
            return feature_forward(latents, sigmas, own_condition, feature_block=feature_block)

        capability.predict_dmad_features = features
        trajectory = SimpleNamespace(states=[clean], condition=condition)
        _, metrics = objective.backward_discriminator(trajectory, 17, 4, 0, 1.0)
        shape = clean.shape
        scale = capability.model.vae_spatial_scale_factor
        real.load.assert_called_once_with(capability, 17, geometry=(shape.num_frames, shape.latent_height * scale, shape.latent_width * scale))
        self.assertIs(calls[0], condition)
        self.assertIs(calls[1], real.load.return_value[1])
        self.assertIsNot(calls[0], calls[1])
        self.assertTrue(all(value["references"] for value in calls))
        self.assertIsNone(clean.video.grad)
        self.assertIsNone(clean.audio.grad)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in trainer.fake_trainable_params if parameter.grad is not None), 0)
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in trainer.dmd2_gan_trainable_params if parameter.grad is not None), 0)
        self.assertIn("gan_d_real", metrics)


if __name__ == "__main__":
    unittest.main()
