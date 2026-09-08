import copy
import unittest

import torch

from lightx2v_train.model_zoo.native.minimax_h3.vae_adversarial import (
    LatentConditionedFramePatchDiscriminator,
    LatentConditionedVideoPatchDiscriminator,
    _per_sample_feature_distance,
    make_ensemble_seraena_correction_target,
)


class MiniMaxH3VAEAdversarialMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(71)
        self.video = torch.randn(2, 3, 7, 32, 32)
        self.latent = torch.randn(2, 24, 3, 2, 2)

    def _model(self, frame_wise):
        common = dict(
            base_channels=4,
            condition_channels=2,
            channel_multipliers=(1, 2),
            group_norm_groups=2,
            spectral_normalization=True,
        )
        if frame_wise:
            return LatentConditionedFramePatchDiscriminator(spatial_strides=(2, 2), **common)
        return LatentConditionedVideoPatchDiscriminator(temporal_strides=(1, 2), **common)

    def test_checkpoint_matches_outputs_gradients_and_spectral_buffers(self):
        for frame_wise in (False, True):
            for training in (False, True):
                with self.subTest(frame_wise=frame_wise, training=training):
                    eager = self._model(frame_wise).train(training)
                    checked = copy.deepcopy(eager)
                    checked.gradient_checkpointing = True
                    eager_input = self.video.clone().requires_grad_(True)
                    checked_input = self.video.clone().requires_grad_(True)
                    eager_logits = eager(eager_input, self.latent)
                    checked_logits = checked(checked_input, self.latent)
                    for expected, actual in zip(eager_logits, checked_logits, strict=True):
                        torch.testing.assert_close(actual, expected)
                    sum(value.square().mean() for value in eager_logits).backward()
                    sum(value.square().mean() for value in checked_logits).backward()
                    torch.testing.assert_close(checked_input.grad, eager_input.grad)
                    for expected, actual in zip(eager.parameters(), checked.parameters(), strict=True):
                        torch.testing.assert_close(actual.grad, expected.grad)
                    for name, value in eager.named_buffers():
                        torch.testing.assert_close(dict(checked.named_buffers())[name], value)
                    self.assertEqual(checked.training, training)
                    for block in checked.blocks:
                        self.assertEqual(block.convolution.training, training)

    def test_two_training_forwards_keep_each_spectral_norm_snapshot(self):
        eager = self._model(frame_wise=True)
        checked = copy.deepcopy(eager)
        checked.gradient_checkpointing = True
        eager_losses = []
        checked_losses = []
        for scale in (1.0, 0.7):
            eager_losses.append(sum(value.square().mean() for value in eager(self.video * scale, self.latent)))
            checked_losses.append(sum(value.square().mean() for value in checked(self.video * scale, self.latent)))
        sum(eager_losses).backward()
        sum(checked_losses).backward()
        for expected, actual in zip(eager.parameters(), checked.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        for name, value in eager.named_buffers():
            torch.testing.assert_close(dict(checked.named_buffers())[name], value)

    def test_checkpoint_saves_fewer_activations_for_a_64_frame_clip(self):
        video = self.video[:1].repeat(1, 1, 10, 1, 1)[:, :, :64]
        latent = self.latent[:1]
        for frame_wise in (False, True):
            with self.subTest(frame_wise=frame_wise):
                model = self._model(frame_wise)
                saved_counts = []
                for enabled in (False, True):
                    model.gradient_checkpointing = enabled
                    saved_numel = []

                    def pack(tensor):
                        saved_numel.append(tensor.numel())
                        return tensor

                    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                        logits = model(video, latent)
                    self.assertEqual(logits[0].shape[2], 64)
                    saved_counts.append(sum(saved_numel))
                    del logits
                self.assertLess(saved_counts[1], saved_counts[0] * 0.7)

    def test_serial_ensemble_matches_joint_graph_correction(self):
        branches = ((self._model(False).eval(), 1.0), (self._model(True).eval(), 0.75))
        for model, _ in branches:
            model.gradient_checkpointing = True
        real = self.video
        fake = torch.randn_like(real)
        leaf = fake.detach().requires_grad_(True)
        distances = []
        for model, _ in branches:
            with torch.no_grad():
                real_logits = model(real, self.latent)
            distances.append(_per_sample_feature_distance(real_logits, model(leaf, self.latent)))
        distance = sum(value * weight for value, (_, weight) in zip(distances, branches, strict=True)) / 1.75
        gradient = torch.autograd.grad(distance.sum(), leaf)[0]
        raw_rms = gradient.square().flatten(1).mean(1).sqrt()
        expected = (-gradient / raw_rms.clamp_min(1e-6).view(-1, 1, 1, 1, 1)).clamp(-3, 3) * 0.02

        result = make_ensemble_seraena_correction_target(
            branches, real, fake, self.latent, correction_scale=0.02, normalized_clamp=3.0
        )

        torch.testing.assert_close(result.correction, expected, atol=1e-7, rtol=1e-4)
        torch.testing.assert_close(result.feature_distance, distance.detach())
        torch.testing.assert_close(result.raw_correction_rms, raw_rms)
        self.assertFalse(result.target.requires_grad)
        self.assertTrue(all(parameter.grad is None for model, _ in branches for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
