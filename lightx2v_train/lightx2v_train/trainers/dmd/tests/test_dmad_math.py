import unittest

import torch

from lightx2v_train.trainers.dmd.dmad_math import (
    DmadConfig,
    DualHeadCritic,
    GapRouter,
    PowerEMA,
    critic_loss,
    generator_loss,
    rollout_base_sigmas,
)


class DmadMathTests(unittest.TestCase):
    def test_linear_generator_and_bce(self):
        r = torch.tensor([2.0], requires_grad=True)
        t = torch.tensor([-1.0], requires_grad=True)
        w = torch.tensor(0.4, requires_grad=True)
        loss = generator_loss(r, t, w)
        self.assertAlmostEqual(float(loss), -1.6)
        loss.backward()
        self.assertEqual(float(r.grad), -1)
        self.assertAlmostEqual(float(t.grad), -0.4)
        self.assertIsNone(w.grad)
        logits = [torch.zeros(1, requires_grad=True) for _ in range(4)]
        loss = critic_loss(*logits)
        self.assertAlmostEqual(float(loss), 4 * float(torch.log(torch.tensor(2.0))))
        loss.backward()
        self.assertEqual([float(x.grad) for x in logits], [0.5, 0.5, -0.5, -0.5])

    def test_balanced_modalities(self):
        heads = DualHeadCritic(4)
        video, audio = torch.randn(1, 1, 4), torch.randn(1, 1, 4)
        a = heads(video, audio)
        b = heads(video.expand(1, 100, 4), audio)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)

    def test_frozen_heads_preserve_input_gradient(self):
        heads = DualHeadCritic(4).requires_grad_(False)
        v = torch.randn(1, 5, 4, requires_grad=True)
        a = torch.randn(1, 2, 4, requires_grad=True)
        generator_loss(*heads(v, a)).backward()
        self.assertGreater(float(v.grad.norm()), 0)
        self.assertGreater(float(a.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in heads.parameters()))

    def test_gap_warmup_and_normalization(self):
        c = DmadConfig(gap_bands=3, gap_ready_bands=3, gap_min_count=2)
        router = GapRouter(c)
        self.assertEqual(float(router.weight(0)), 1)
        for _ in range(2):
            for b in range(3):
                router.update(b, torch.tensor(float(b)), torch.tensor(0.0))
        weights = torch.stack([router.weight(b) for b in range(3)])
        self.assertGreater(float(weights[0]), float(weights[2]))
        self.assertAlmostEqual(float(weights.mean()), 1, places=6)
        clone = GapRouter(c)
        clone.load_state_dict(router.state_dict())
        torch.testing.assert_close(clone.weight(0), weights[0])

    def test_rollout_grid(self):
        for n in range(1, 5):
            grid = rollout_base_sigmas(n, 0.02, 0.98, device="cpu")
            self.assertEqual(len(grid), n + 1)
            self.assertEqual(float(grid[0]), 1)
            self.assertEqual(float(grid[-1]), 0)
            self.assertTrue(bool((grid[:-1] >= grid[1:]).all()))
        torch.testing.assert_close(rollout_base_sigmas(4, 0.02, 0.98, device="cpu", random_timesteps=False), torch.tensor([1, 0.75, 0.5, 0.25, 0]))

    def test_power_ema_roundtrip_and_export(self):
        model = torch.nn.Linear(2, 2)
        ema = PowerEMA(model, 6.94)
        with torch.no_grad():
            model.weight.fill_(3)
        ema.update()
        torch.testing.assert_close(ema.shadow["weight"], model.weight)
        with torch.no_grad():
            model.weight.fill_(5)
        ema.update()
        beta = 0.5**7.94
        torch.testing.assert_close(ema.shadow["weight"], torch.full_like(model.weight, 3 * beta + 5 * (1 - beta)))
        other = PowerEMA(model, 6.94)
        other.load_state_dict(ema.state_dict())
        with other.average_parameters():
            torch.testing.assert_close(model.weight, ema.shadow["weight"])
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))
        self.assertEqual(other.num_updates, 2)

    def test_invalid_options(self):
        for config in ({"gap_tau": 0}, {"gap_tau": float("nan")}, {"renoise_sigma_max": 1.0}, {"lambda_real": -1}, {"unread_option": 4}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DmadConfig.from_mapping(config)


if __name__ == "__main__":
    unittest.main()
