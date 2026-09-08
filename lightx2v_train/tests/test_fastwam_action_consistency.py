import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from lightx2v_train.trainers.fastwam_action_consistency.checkpoint import ActionConsistencyCheckpointManager
from lightx2v_train.trainers.fastwam_action_consistency.config import FastWAMActionConsistencyConfig
from lightx2v_train.trainers.fastwam_action_consistency.roles import ActionConsistencyRoles
from lightx2v_train.trainers.fastwam_action_consistency.trainer import FastWAMActionConsistencyTrainer, _masked_mse, _masked_pseudo_huber, shifted_consistency_pair
from torch import nn


class _Denoiser(nn.Module):
    def __init__(self, value, dynamic=False):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))
        self.dynamic = dynamic
        self.calls = []

    def forward(self, action, timestep, condition):
        self.calls.append((action.detach().clone(), timestep.detach().clone(), condition, torch.is_grad_enabled()))
        if self.dynamic:
            return self.value * action + (timestep / 1000).view(-1, 1, 1)
        return torch.ones_like(action) * self.value


def _trainer(sigma, source=None, steps=4):
    consistency = {"teacher_reference_steps": steps}
    if source is not None:
        consistency["flow_target"] = source
    config = {"training": {"student": {"train_type": "full", "optimizer": {}}, "action_consistency": consistency}}
    trainer = FastWAMActionConsistencyTrainer.__new__(FastWAMActionConsistencyTrainer)
    trainer.parsed = FastWAMActionConsistencyConfig.from_mapping(config)
    module = SimpleNamespace(train_action_scheduler=SimpleNamespace(num_train_timesteps=1000))
    trainer.model = SimpleNamespace(unwrap_module=lambda: module)
    trainer.student_denoiser = _Denoiser(0.2)
    trainer.teacher_denoiser = _Denoiser(0.3).requires_grad_(False)
    trainer.target_denoiser = _Denoiser(0.5).requires_grad_(False)
    trainer._sigma_pair = lambda action: (sigma, sigma * 0.5)
    return trainer


class FastWAMActionConsistencyTest(unittest.TestCase):
    def test_requested_configs_parse(self):
        root = Path(__file__).parents[1]
        for name, expected_rank, source in (
            ("libero_action_1step_consistency.yaml", 64, "data"),
            ("robotwin_action_1step_consistency.yaml", 128, "data"),
            ("robotwin_action_1step_consistency_teacher.yaml", 128, "teacher"),
        ):
            with (root / "configs/train/fastwam_action_dmd" / name).open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            parsed = FastWAMActionConsistencyConfig.from_mapping(config)
            self.assertEqual(config["training"]["method"], "fastwam_action_consistency")
            self.assertEqual(parsed.target_steps, 10)
            self.assertEqual(parsed.teacher_reference_steps, 10)
            self.assertEqual(parsed.flow_target, source)
            self.assertEqual(parsed.student.lora["rank"], expected_rank)
            self.assertAlmostEqual(parsed.flow_loss_weight, 0.2)
            if source == "teacher":
                self.assertTrue(config["training"]["output_dir"].endswith("_teacher"))
                self.assertTrue(config["logging"]["wandb"]["name"].endswith("-teacher"))

    def test_invalid_flow_target_is_rejected(self):
        for source in ("ema", "DATA", "", 1):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "flow_target"):
                _trainer(torch.tensor([0.5]), source)

    def test_default_and_data_preserve_loss_gradient_and_rng(self):
        action = torch.tensor([[[0.1], [0.4]], [[-0.2], [0.8]]])
        sigma = torch.tensor([0.8, 0.4])
        valid = torch.tensor([[True, False], [True, True]])
        for source in (None, "data"):
            with self.subTest(source=source):
                trainer = _trainer(sigma, source)
                torch.manual_seed(9)
                noise = torch.randn_like(action)
                expected_rng = torch.get_rng_state()
                value = torch.tensor(0.2, requires_grad=True)
                # Constant teacher=0.3, EMA=0.5: f_student - f_EMA has a closed form.
                difference = sigma.view(-1, 1, 1) * (0.3 - value) + (sigma * 0.5).view(-1, 1, 1) * 0.2
                expected_consistency = _masked_pseudo_huber(difference.expand_as(action), torch.zeros_like(action), valid, 0.001)
                expected_flow = _masked_mse(value.expand_as(action), noise - action, valid)
                expected = expected_consistency + 0.2 * expected_flow
                expected.backward()
                torch.manual_seed(9)
                loss, metrics = trainer._loss({"action": action}, None, valid)
                loss.backward()
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(metrics["consistency"], expected_consistency)
                torch.testing.assert_close(metrics["flow"], expected_flow)
                torch.testing.assert_close(trainer.student_denoiser.value.grad, value.grad)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng)
                self.assertEqual(len(trainer.teacher_denoiser.calls), 1)

    def test_teacher_rollout_matches_euler_endpoint_and_reuses_first_call(self):
        for dtype in (torch.float32, torch.bfloat16):
            for steps in (1, 4, 10):
                with self.subTest(dtype=dtype, steps=steps):
                    sigma = torch.tensor([0.25, 0.8], dtype=dtype)
                    trainer = _trainer(sigma, "teacher", steps)
                    trainer.teacher_denoiser.dynamic = True
                    trainer.teacher_denoiser.value.fill_(0.25)  # Exactly representable in BF16 and FP32.
                    noisy = torch.tensor([[[1.0, -0.5]], [[0.2, 0.7]]], dtype=dtype)
                    original = noisy.clone()
                    condition = object()
                    with torch.no_grad():
                        first = trainer.teacher_denoiser(noisy, sigma * 1000, condition)
                    first_before = first.clone()
                    target = trainer._teacher_flow_target(noisy, sigma, condition, first)
                    self.assertEqual(len(trainer.teacher_denoiser.calls), steps)
                    self.assertEqual(target.dtype, torch.float32)
                    self.assertFalse(target.requires_grad)
                    torch.testing.assert_close(noisy, original)
                    torch.testing.assert_close(first, first_before)

                    # Independently integrate the field; use endpoint displacement as the oracle.
                    state = noisy.float()
                    for index, (seen_action, seen_t, seen_condition, grad_enabled) in enumerate(trainer.teacher_denoiser.calls):
                        timestep = sigma * 1000 if index == 0 else (sigma.float() * (1 - index / steps) * 1000).to(dtype)
                        torch.testing.assert_close(seen_action, state.to(dtype))
                        torch.testing.assert_close(seen_t, timestep)
                        self.assertIs(seen_condition, condition)
                        self.assertFalse(grad_enabled)
                        velocity = 0.25 * state.to(dtype) + (timestep / 1000).view(-1, 1, 1)
                        state = state - (sigma.float() / steps).view(-1, 1, 1) * velocity.float()
                    expected = (noisy.float() - state) / sigma.float().view(-1, 1, 1)
                    torch.testing.assert_close(target, expected)
                    if steps == 1:
                        torch.testing.assert_close(target, first.float())

    def test_teacher_target_is_stable_at_zero_and_tiny_sigma(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                sigma = torch.tensor([0.0, 1e-12, 1e-6], dtype=dtype)
                trainer = _trainer(sigma, "teacher", steps=10)
                noisy = torch.ones(3, 2, 2, dtype=dtype)
                with torch.no_grad():
                    first = trainer.teacher_denoiser(noisy, sigma * 1000, None)
                target = trainer._teacher_flow_target(noisy, sigma, None, first)
                self.assertTrue(torch.isfinite(target).all())
                torch.testing.assert_close(target, first.float())

    def test_teacher_loss_mask_and_student_only_gradients(self):
        action = torch.tensor([[[0.1], [0.4]], [[-0.2], [0.8]]])
        sigma = torch.tensor([0.3, 0.9])
        valid = torch.tensor([[True, False], [True, True]])
        results = []
        for offset in (0.0, 100.0):
            trainer = _trainer(sigma, "teacher", steps=4)
            trainer.teacher_denoiser.dynamic = True
            condition = object()
            sample = action + (~valid).unsqueeze(-1) * offset
            torch.manual_seed(3)
            noise = torch.randn_like(action)
            expected_rng = torch.get_rng_state()
            torch.manual_seed(3)
            loss, metrics = trainer._loss({"action": sample}, condition, valid)
            loss.backward()
            results.append((loss.detach(), trainer.student_denoiser.value.grad.clone()))
            torch.testing.assert_close(torch.get_rng_state(), expected_rng)
            self.assertEqual(set(metrics), {"consistency", "flow"})
            calls = trainer.teacher_denoiser.calls
            self.assertEqual(len(calls), 4)
            torch.testing.assert_close(calls[0][0], (1 - sigma.view(-1, 1, 1)) * sample + sigma.view(-1, 1, 1) * noise)
            self.assertTrue(all(call[2] is condition and not call[3] for call in calls))
            self.assertEqual(len(trainer.student_denoiser.calls), 1)
            self.assertIsNone(trainer.teacher_denoiser.value.grad)
            self.assertIsNone(trainer.target_denoiser.value.grad)
            expected_target = torch.stack([0.3 * call[0] + (call[1] / 1000).view(-1, 1, 1) for call in calls]).mean(0)
            expected_flow = _masked_mse(torch.full_like(action, 0.2), expected_target, valid)
            torch.testing.assert_close(metrics["flow"], expected_flow)
            self.assertGreater(trainer.student_denoiser.value.grad.abs().item(), 0)
        torch.testing.assert_close(results[0], results[1])

    def test_shifted_pair_uses_two_step_stride(self):
        base = torch.tensor([1.0, 0.75, 0.5, 0.25])
        start, end = shifted_consistency_pair(base, shift=5.0, target_steps=2)
        expected_end = 5.0 * torch.tensor([0.5, 0.25, 0.0, 0.0]) / (1.0 + 4.0 * torch.tensor([0.5, 0.25, 0.0, 0.0]))
        torch.testing.assert_close(end, expected_end)
        self.assertTrue(torch.all(end <= start))

    def test_x0_consistency_identity_and_masked_loss(self):
        action = torch.randn(2, 3, 4)
        noise = torch.randn_like(action)
        sigma = torch.tensor([0.2, 0.9]).view(2, 1, 1)
        noisy = (1.0 - sigma) * action + sigma * noise
        predicted_x0 = noisy - sigma * (noise - action)
        torch.testing.assert_close(predicted_x0, action)

        changed_only_under_mask = action.clone()
        changed_only_under_mask[:, 1:] += 10.0
        valid = torch.tensor([[True, False, False], [True, False, False]])
        self.assertEqual(float(_masked_pseudo_huber(action, changed_only_under_mask, valid, 0.001)), 0.0)

    def test_ema_updates_without_changing_teacher(self):
        config = SimpleNamespace(train_type="full", lora=None)
        expert = nn.Linear(2, 2, bias=False)
        roles = ActionConsistencyRoles.build(expert, config)
        teacher_before = roles.teacher.weight.detach().clone()
        target_before = roles.target.weight.detach().clone()
        with torch.no_grad():
            roles.student.weight.add_(2.0)
        roles.update_target(0.5)
        torch.testing.assert_close(roles.target.weight, target_before + 1.0)
        torch.testing.assert_close(roles.teacher.weight, teacher_before)

    def test_checkpoint_round_trip_restores_online_and_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(train_type="full", lora=None)
            roles = ActionConsistencyRoles.build(nn.Linear(2, 2, bias=False), config)
            optimizer = torch.optim.AdamW(roles.student.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
            trainer = SimpleNamespace(
                config={"resume": {}},
                runtime_config={"training": {"method": "fastwam_action_consistency"}},
                output_dir=directory,
                save_total_limit=2,
                parsed=SimpleNamespace(student=config),
                roles=roles,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            manager = ActionConsistencyCheckpointManager(trainer)
            student_before = roles.student.weight.detach().clone()
            target_before = roles.target.weight.detach().clone()
            manager.save(7)
            with torch.no_grad():
                roles.student.weight.zero_()
                roles.target.weight.zero_()
            self.assertEqual(manager.load(str(Path(directory) / "checkpoint-000000007")), 7)
            torch.testing.assert_close(roles.student.weight, student_before)
            torch.testing.assert_close(roles.target.weight, target_before)


if __name__ == "__main__":
    unittest.main()
