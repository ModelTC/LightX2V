"""Temporary feature capture must not change Wan's normal prediction path."""

import types
import unittest

import torch
from torch import nn

from lightx2v_train.model_zoo.wan.wan_t2v import WanT2VModel


class WanHeadFeatureTests(unittest.TestCase):
    def make_model(self):
        model = object.__new__(WanT2VModel)
        model.use_causal_transformer = False
        model.sp_size = 1
        model.transformer = nn.Module()
        model.transformer.head = nn.Module()
        model.transformer.head.head = nn.Linear(3, 2)

        def predict(instance, latents, sigma, condition):
            return instance.transformer.head.head(latents)

        model.predict_denoiser_output = types.MethodType(predict, model)
        return model

    def test_velocity_unchanged_features_detached_and_hook_removed(self):
        model = self.make_model()
        latents = torch.randn(1, 4, 3, requires_grad=True)
        expected = model.predict_denoiser_output(latents, torch.tensor(0.5), {})
        velocity, features = model.predict_velocity_with_features(latents, torch.tensor(0.5), {})
        torch.testing.assert_close(velocity, expected)
        torch.testing.assert_close(features, latents)
        self.assertFalse(velocity.requires_grad)
        self.assertFalse(features.requires_grad)
        self.assertEqual(len(model.transformer.head.head._forward_pre_hooks), 0)

    def test_hook_removed_when_prediction_fails(self):
        model = self.make_model()

        def fail(*args):
            raise RuntimeError("test failure")

        model.predict_denoiser_output = fail
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            model.predict_velocity_with_features(torch.zeros(1, 1, 3), torch.tensor(0.5), {})
        self.assertEqual(len(model.transformer.head.head._forward_pre_hooks), 0)

    def test_unsupported_causal_and_sequence_parallel_rejected(self):
        model = self.make_model()
        for causal, sp_size in ((True, 1), (False, 2)):
            model.use_causal_transformer, model.sp_size = causal, sp_size
            with self.assertRaisesRegex(ValueError, "non-causal"):
                model.predict_velocity_with_features(torch.zeros(1, 1, 3), torch.tensor(0.5), {})


if __name__ == "__main__":
    unittest.main()
