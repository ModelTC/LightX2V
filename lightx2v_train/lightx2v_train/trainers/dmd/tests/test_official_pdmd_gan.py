"""Check loss signs and role isolation through the actual PDMD update loop."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lightx2v_train.trainers.dmd.dmd2_gan import Dmd2GanObjective, discriminator_logistic_loss, gan_sigmas, generator_logistic_loss
from lightx2v_train.trainers.dmd.dmd2_gan_head import H3GanHead, frozen_dmd2_gan_critic
from lightx2v_train.trainers.dmd.official_pdmd_core import full_rollout, rollout_grid, student_objective
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining
from lightx2v_train.trainers.dmd.tests.test_official_pdmd_core import ToyRoles, pack


class GanPDMDTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        roles = ToyRoles()
        self.roles = roles
        t = SimpleNamespace(**roles.capabilities)
        self.t = t
        t.config = {"seed": 17}
        t.dmd_config = {"gan": {"generator_weight": 0.005, "discriminator_weight": 0.01, "feature_block": 0}}
        t.trainable_params = list(roles.denoisers["student"].parameters())
        t.fake_trainable_params = list(roles.denoisers["fake"].parameters())
        t.dmd2_gan_head = H3GanHead(3, hidden_dim=8, temporal_bins=2)
        t.dmd2_gan_trainable_params = list(t.dmd2_gan_head.parameters())
        t.fake.predict_dmad_features = lambda x, sigma, condition, feature_block: {"video": roles.denoisers["fake"](x, sigma, condition).video}
        for name, params in (("optimizer", t.trainable_params), ("fake_optimizer", t.fake_trainable_params), ("dmd2_gan_optimizer", t.dmd2_gan_trainable_params)):
            setattr(t, name, torch.optim.AdamW(params, lr=0.001, betas=(0.0, 0.9), weight_decay=0))
        t.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer, lambda _: 1)
        t.fake_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.fake_optimizer, lambda _: 1)
        t.dmd2_gan_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(t.dmd2_gan_optimizer, lambda _: 1)
        t._set_student_gradient_sync = Mock()
        t._set_fake_gradient_sync = Mock()
        t._sync_sequence_parallel_grads = Mock()
        t._after_student_optimizer_step = Mock()
        t.max_grad_norm = 1.0
        t.num_inference_steps = 4
        noise = pack((torch.randn(1, 3, 2, 2, 2), torch.randn(2, 2, 3)))
        self.trajectory = full_rollout(t.student, noise, {"bias": 0.04}, rollout_grid(4, 12), rollout_grid(4, 3))
        self.real = SimpleNamespace(load=Mock(return_value=(self.trajectory.states[-1], {"bias": 0.31})))
        self.gan = Dmd2GanObjective(t, self.real)

    def test_logistic_signs_and_noise_levels(self):
        logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        generator_logistic_loss(logits).backward()
        self.assertTrue(bool((logits.grad < 0).all()))
        logits.grad = None
        discriminator_logistic_loss(logits, real=False).backward()
        self.assertTrue(bool((logits.grad > 0).all()))
        logits.grad = None
        discriminator_logistic_loss(logits, real=True).backward()
        self.assertTrue(bool((logits.grad < 0).all()))
        rng = torch.Generator().manual_seed(123)
        for _ in range(20):
            sigmas = gan_sigmas(rng, "cpu")
            self.assertEqual(sigmas.shape, (2,))
            self.assertEqual(sigmas[0], sigmas[1])
            self.assertTrue(bool(((sigmas >= 0) & (sigmas <= 0.999)).all()))

    def test_generator_gradient_reaches_student_only(self):
        t = self.t
        with frozen_dmd2_gan_critic(t):
            _, _, clean = student_objective(t.student, t.fake, t.teacher, self.trajectory, 2, 0.4, return_clean=True)
            loss, metrics = self.gan.generator(clean, self.trajectory.condition, 10, 0)
            loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in t.trainable_params), 0)
        for p in [*t.fake_trainable_params, *t.dmd2_gan_trainable_params, *self.roles.denoisers["teacher"].parameters()]:
            self.assertIsNone(p.grad)
        self.assertTrue(all(p.requires_grad for p in t.fake_trainable_params))
        self.assertIn("gan_g", metrics)

    def test_fake_then_student_updates_and_real_cursor(self):
        t = self.t
        engine = OfficialH3PdmdTraining.__new__(OfficialH3PdmdTraining)
        engine.trainer, engine.dmd, engine.gan = t, t.dmd_config, self.gan
        engine.critic_steps, engine.accum, engine.seed = 10, 1, 17
        engine.cached, engine.consumed_microbatches = [], 9
        engine.route_loss_scale = lambda _: 1.0
        engine._trajectory = lambda sample, iteration, slot: self.trajectory
        before = {name: [p.detach().clone() for p in model.parameters()] for name, model in self.roles.denoisers.items()}
        head_before = [p.detach().clone() for p in t.dmd2_gan_trainable_params]
        role, metrics = engine._update(iter([{}]), 9)
        self.assertEqual(role, "fake")
        self.assertEqual(engine.consumed_microbatches, 10)
        self.real.load.assert_called_once_with(t.student, 9)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before["fake"], t.fake_trainable_params)))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(head_before, t.dmd2_gan_trainable_params)))
        torch.testing.assert_close(t.trainable_params[0], before["student"][0])
        self.assertIn("gan_d_real", metrics)
        fake_before = [p.detach().clone() for p in t.fake_trainable_params]
        head_before = [p.detach().clone() for p in t.dmd2_gan_trainable_params]
        role, metrics = engine._update(iter(()), 10)
        self.assertEqual(role, "student")
        self.assertEqual(engine.consumed_microbatches, 10)
        self.assertEqual(engine.cached, [])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before["student"], t.trainable_params)))
        for a, b in zip(fake_before + head_before, t.fake_trainable_params + t.dmd2_gan_trainable_params):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(self.roles.denoisers["teacher"].weights, before["teacher"][0])
        self.assertIn("pdmd_loss", metrics)
        self.assertIn("gan_g", metrics)


if __name__ == "__main__":
    unittest.main()
