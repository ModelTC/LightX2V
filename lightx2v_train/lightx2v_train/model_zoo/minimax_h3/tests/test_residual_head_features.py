"""Target-only H3 HEAD features, including the native Diffusers forward."""

import unittest
from unittest.mock import patch

import torch

from lightx2v_train.model_capabilities import DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests import test_ref2av_capability as fixtures
from lightx2v_train.model_zoo.native.minimax_h3.modeling import _transformer_class


class PackedProjectionFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_out = torch.nn.Linear(4, 96)
        self.audio_proj_out = torch.nn.Linear(4, 32)
        self.fail = False
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        sequence = kwargs["token_tags"].numel()
        packed = torch.arange(sequence, dtype=torch.float32).reshape(1, -1, 1).expand(1, -1, 4)
        video = self.proj_out(packed).index_select(1, kwargs["video_indices"])
        if self.fail:
            raise RuntimeError("injected forward failure")
        audio = self.audio_proj_out(packed).index_select(1, kwargs["audio_indices"])
        return video, audio


class H3ResidualHeadFeatureTests(unittest.TestCase):
    def setup_inputs(self, model=None):
        self.helper = fixtures.Ref2AVCapabilityTest()
        self.model = model or fixtures.model_wrapper()
        self.capability = self.model.ensure_capabilities().require(DistributionMatchingCapability)
        sample, self.condition = self.helper.prepare(self.capability, fixtures.reference_condition())
        self.shape = self.capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda x: x)
        self.latents = self.capability.initial_latents(self.shape, torch.float32, lambda x: x)
        self.sigma = torch.tensor([0.3])

    def test_exact_target_modality_indices_exclude_all_references_and_text(self):
        self.setup_inputs()
        self.model.transformer = PackedProjectionFixture()
        ordinary = self.capability.predict_velocity(self.latents, self.sigma, self.condition)
        velocity, features = self.model.predict_velocity_with_features(self.latents, self.sigma, self.condition)
        self.assertEqual(self.model.transformer.calls, 2)
        layout = self.capability._layout(self.condition, self.shape)
        for name, count in (("video", layout.num_condition_video_rows), ("audio", layout.num_condition_audio_rows)):
            indices = getattr(layout, name + "_indices")[count:]
            expected = indices.float().reshape(1, -1, 1).expand(1, -1, 4)
            torch.testing.assert_close(features[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(getattr(velocity, name), getattr(ordinary, name), rtol=0, atol=0)
            self.assertFalse(features[name].requires_grad)
            self.assertFalse(getattr(velocity, name).requires_grad)
        self.assertEqual(self.model.residual_head_feature_dims, {"video": 4, "audio": 4})
        self.assertEqual(self.model.residual_head_output_dims, {"video": 96, "audio": 32})
        self.assertFalse(self.model.transformer.proj_out._forward_pre_hooks)
        self.assertFalse(self.model.transformer.audio_proj_out._forward_pre_hooks)

    def test_hook_cleanup_on_failure(self):
        self.setup_inputs()
        self.model.transformer = PackedProjectionFixture()
        self.model.transformer.fail = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.model.predict_velocity_with_features(self.latents, self.sigma, self.condition)
        self.assertFalse(self.model.transformer.proj_out._forward_pre_hooks)
        self.assertFalse(self.model.transformer.audio_proj_out._forward_pre_hooks)
        self.model.transformer.fail = False
        _, features = self.model.predict_velocity_with_features(self.latents, self.sigma, self.condition)
        self.assertEqual(features["video"].shape[:2], self.latents.video.shape[:2])

    def test_sigma_uses_modality_shift_and_sp_rejected(self):
        self.setup_inputs()
        self.model.transformer = PackedProjectionFixture()
        sigma = self.model.residual_head_sigmas(self.sigma)
        expected = self.capability._modality_sigmas(self.sigma)
        torch.testing.assert_close(sigma["video"], expected[0])
        torch.testing.assert_close(sigma["audio"], expected[1])
        self.assertFalse(torch.equal(sigma["video"], sigma["audio"]))
        with patch("lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability.get_sequence_parallel_world_size", return_value=2):
            with self.assertRaisesRegex(ValueError, "sequence_parallel"):
                self.model.predict_velocity_with_features(self.latents, self.sigma, self.condition)

    def test_native_projection_features_reconstruct_exact_velocity(self):
        model = fixtures.model_wrapper()
        model.running_dtype = torch.float32
        model.text_dim = 12
        model.video_latent_channels = 2
        model.audio_latent_channels = 2
        model.transformer = _transformer_class()(
            num_attention_heads=2,
            attention_head_dim=8,
            hidden_size=16,
            num_layers=1,
            num_refiner_layers=1,
            ffn_dim=32,
            in_channels=2,
            audio_in_channels=2,
            patch_size=(1, 2, 2),
            text_dim=12,
            freq_dim=4,
            time_embed_hidden_dim=16,
            time_embed_dim=8,
            rope_freq_dim=1,
        ).eval()
        capability = model.ensure_capabilities().require(DistributionMatchingCapability)
        cached = fixtures.reference_condition()
        cached["prompt_embeds"] = torch.randn(1, 4, 12)
        cached["references"][0]["video_latents"] = torch.randn(6, 8)
        cached["references"][1]["audio_latents"] = torch.randn(4, 2)
        sample, condition = fixtures.Ref2AVCapabilityTest().prepare(capability, cached)
        shape = capability.latent_shape(sample, [{"value": [124, 64, 96]}], lambda x: x)
        latents = capability.initial_latents(shape, torch.float32, lambda x: x)
        sigma = torch.tensor([0.5])
        velocity, features = model.predict_velocity_with_features(latents, sigma, condition)
        with torch.no_grad():
            ordinary = capability.predict_velocity(latents, sigma, condition)
            for name, projection in (("video", model.transformer.proj_out), ("audio", model.transformer.audio_proj_out)):
                torch.testing.assert_close(getattr(velocity, name), getattr(ordinary, name), rtol=0, atol=0)
                torch.testing.assert_close(projection(features[name]), getattr(velocity, name))
        # Capturing no-grad fake features must not disable subsequent student gradients.
        predicted = capability.predict_velocity(latents, sigma, condition)
        (predicted.video.square().mean() + predicted.audio.square().mean()).backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.transformer.parameters()))


if __name__ == "__main__":
    unittest.main()
