"""CPU integration checks, with an optional independent released-engine oracle.

Set PDMD_REFERENCE_SOURCE to a reviewed official checkout to run the direct
oracle comparison. The normal tests do not require that external repository.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

import torch

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import MiniMaxH3JointLatents
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_ref2av_capability import model_wrapper
from lightx2v_train.trainers.dmd.official_pdmd_core import (
    critic_objective,
    critic_sigmas,
    full_rollout,
    interval_sigmas,
    rollout_grid,
    student_objective,
)


def pack(value):
    """Official [1,C,T,H,W]/[stereo,C,T] -> explicit-batch H3 tokens."""
    video, audio = value
    return MiniMaxH3JointLatents(video.permute(0, 2, 3, 4, 1).reshape(1, 8, 3), audio.permute(0, 2, 1).reshape(1, 6, 2), None)


def unpack(value):
    return value.video.reshape(1, 2, 2, 2, 3).permute(0, 4, 1, 2, 3), value.audio.reshape(2, 3, 2).permute(0, 2, 1)


class ToyDenoiser(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.calls = []

    def forward(self, latents, sigmas, condition):
        self.calls.append((MiniMaxH3JointLatents(latents.video.detach().clone(), latents.audio.detach().clone(), None), sigmas.detach().clone(), torch.is_grad_enabled(), self.training))
        result = []
        for index, (value, other) in enumerate(((latents.video, latents.audio), (latents.audio, latents.video))):
            spatial_bias = torch.linspace(-0.5, 0.8, value.numel(), dtype=torch.float32).reshape(value.shape)
            velocity = self.weights[index] * value + self.weights[index + 2] * spatial_bias + self.weights[index + 4] * other.mean()
            result.append(velocity + float(condition["bias"]) * sigmas[index])
        return MiniMaxH3JointLatents(*result, None)


class ToyRoles:
    def __init__(self):
        self.models, self.capabilities, self.denoisers = {}, {}, {}
        for name, weights in (
            ("student", [0.17, -0.13, 0.07, 0.14, 0.03, -0.04]),
            ("fake", [-0.03, 0.09, -0.11, 0.06, 0.08, 0.02]),
            ("teacher", [0.21, 0.05, 0.16, -0.09, -0.01, 0.05]),
        ):
            model = model_wrapper()
            denoiser = ToyDenoiser(weights)
            model.transformer = denoiser
            capability = MiniMaxH3DistributionMatchingCapability(
                model,
                {
                    "official_pdmd": True,
                    "video_loss_weight": 0.8,
                    "audio_loss_weight": 0.8,
                    "audio_dmd_loss_weight": 0.8,
                    "video_flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                },
            )
            capability.predict_velocity = denoiser
            self.models[name], self.denoisers[name], self.capabilities[name] = model, denoiser, capability

    def zero_grad(self):
        for denoiser in self.denoisers.values():
            denoiser.zero_grad(set_to_none=True)
            denoiser.calls.clear()


class OfficialToyBackend:
    def __init__(self, roles):
        self.roles = roles
        # The oracle resets adapters after its score calls; this toy has
        # independent role modules, so there are no shared adapters to reset.
        self.dit = torch.nn.Module()

    def predict(self, role, latents, sigmas, condition):
        name = "fake" if role == "critic" else role
        value = self.roles.denoisers[name](pack(latents), torch.stack(tuple(sigmas)), condition)
        return tuple(-tensor for tensor in unpack(value))


class OfficialPDMDCoreTest(unittest.TestCase):
    def setUp(self):
        self.roles = ToyRoles()
        self.student = self.roles.capabilities["student"]
        self.fake = self.roles.capabilities["fake"]
        self.teacher = self.roles.capabilities["teacher"]
        generator = torch.Generator().manual_seed(871)
        self.raw_noise = (torch.randn(1, 3, 2, 2, 2, generator=generator), torch.randn(2, 2, 3, generator=generator))
        self.fresh_raw = tuple(torch.randn(value.shape, generator=generator) for value in self.raw_noise)
        self.noise = pack(self.raw_noise)
        self.fresh = pack(self.fresh_raw)
        self.condition = {"bias": 0.04}

    def trajectory(self, video_shift=1.0, audio_shift=1.0):
        return full_rollout(self.student, self.noise, self.condition, rollout_grid(4, video_shift), rollout_grid(4, audio_shift))

    def test_full_rollout_runs_four_no_grad_eval_calls_and_keeps_stereo_in_batch(self):
        trajectory = self.trajectory()
        self.assertEqual(len(trajectory.states), 5)
        calls = self.roles.denoisers["student"].calls
        self.assertEqual(len(calls), 4)
        for index, (latents, sigmas, grad, training) in enumerate(calls):
            self.assertFalse(grad)
            self.assertFalse(training)
            self.assertEqual(latents.video.shape, (1, 8, 3))
            self.assertEqual(latents.audio.shape, (1, 6, 2))
            torch.testing.assert_close(sigmas, torch.full((2,), 1 - index / 4), rtol=0, atol=0)
        self.assertTrue(all(not tensor.requires_grad for state in trajectory.states for tensor in (state.video, state.audio)))
        self.assertFalse(torch.equal(trajectory.states[-1].video, trajectory.states[1].video))

    def test_all_exit_steps_share_query_ratio_and_only_student_receives_gradients(self):
        trajectory = self.trajectory(12.0, 3.0)
        ratio = 0.37
        for index in range(4):
            with self.subTest(index=index):
                self.roles.zero_grad()
                loss, stats = student_objective(self.student, self.fake, self.teacher, trajectory, index, ratio)
                loss.backward()
                self.assertIsNotNone(self.roles.denoisers["student"].weights.grad)
                self.assertGreater(self.roles.denoisers["student"].weights.grad.norm().item(), 0)
                self.assertIsNone(self.roles.denoisers["fake"].weights.grad)
                self.assertIsNone(self.roles.denoisers["teacher"].weights.grad)
                self.assertEqual([len(value.calls) for value in self.roles.denoisers.values()], [1, 1, 1])
                self.assertTrue(self.roles.denoisers["student"].calls[0][2])
                self.assertFalse(self.roles.denoisers["fake"].calls[0][2])
                self.assertFalse(self.roles.denoisers["teacher"].calls[0][2])
                physical = interval_sigmas(trajectory.video_grid, trajectory.audio_grid, index, ratio)
                query, query_sigmas, _, _ = self.roles.denoisers["fake"].calls[0]
                torch.testing.assert_close(query_sigmas, physical, rtol=0, atol=0)
                torch.testing.assert_close(stats["score_sigma_video"], physical[0], rtol=0, atol=0)
                torch.testing.assert_close(stats["score_sigma_audio"], physical[1], rtol=0, atol=0)
                for side, grid in enumerate((trajectory.video_grid, trajectory.audio_grid)):
                    recovered_ratio = (physical[side] - grid[index + 1]) / (grid[index] - grid[index + 1])
                    torch.testing.assert_close(recovered_ratio, torch.tensor(ratio))
                teacher_query = self.roles.denoisers["teacher"].calls[0][0]
                torch.testing.assert_close(query.video, teacher_query.video, rtol=0, atol=0)
                torch.testing.assert_close(query.audio, teacher_query.audio, rtol=0, atol=0)

    def test_critic_uses_full_endpoint_fresh_noise_and_only_fake_receives_gradients(self):
        trajectory = self.trajectory()
        self.roles.zero_grad()
        sigmas = torch.tensor([0.96, 0.87])
        loss = critic_objective(self.student, self.fake, trajectory, sigmas, self.fresh)
        loss.backward()
        self.assertIsNone(self.roles.denoisers["student"].weights.grad)
        self.assertIsNone(self.roles.denoisers["teacher"].weights.grad)
        self.assertGreater(self.roles.denoisers["fake"].weights.grad.norm().item(), 0)
        query, recorded_sigmas, grad, training = self.roles.denoisers["fake"].calls[0]
        self.assertTrue(grad)
        self.assertFalse(training)
        torch.testing.assert_close(recorded_sigmas, sigmas, rtol=0, atol=0)
        for index, side in enumerate(("video", "audio")):
            expected = (1 - sigmas[index]) * getattr(trajectory.states[-1], side) + sigmas[index] * getattr(self.fresh, side)
            torch.testing.assert_close(getattr(query, side), expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("PDMD_REFERENCE_SOURCE"), "set PDMD_REFERENCE_SOURCE to a reviewed official source checkout")
    def test_direct_released_engine_parity_rollout_all_exits_and_both_role_gradients(self):
        source = Path(os.environ["PDMD_REFERENCE_SOURCE"]) / "src"
        self.assertTrue((source / "pdmd" / "engine.py").is_file())
        sys.path.insert(0, str(source))
        try:
            engine = importlib.import_module("pdmd.engine")
            schedule = importlib.import_module("pdmd.schedule")
        finally:
            sys.path.remove(str(source))
        self.assertEqual(Path(engine.__file__).resolve().parent.parent, source.resolve())
        oracle_roles = ToyRoles()
        backend = OfficialToyBackend(oracle_roles)
        for seed in range(32):
            actual = critic_sigmas(torch.Generator().manual_seed(seed))
            expected = torch.stack(schedule.critic_times(torch.Generator().manual_seed(seed), "cpu"))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for shifts in ((1.0, 1.0), (12.0, 3.0)):
            grids = schedule.schedules(4, *shifts)
            actual_trajectory = self.trajectory(*shifts)
            reference = engine.rollout(backend, self.raw_noise, self.condition, *grids)
            for actual, expected in zip(actual_trajectory.states, reference.states):
                for actual_tensor, expected_tensor in zip(unpack(actual), expected):
                    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
            for sigmas in (torch.tensor([0.12, 0.31]), torch.tensor([0.96, 0.87])):
                self.roles.zero_grad()
                oracle_roles.zero_grad()
                actual = critic_objective(self.student, self.fake, actual_trajectory, sigmas, self.fresh)
                expected, _ = engine.critic_step_loss(backend, reference, tuple(sigmas), self.fresh_raw)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(self.roles.denoisers["fake"].weights.grad, oracle_roles.denoisers["fake"].weights.grad, rtol=2e-6, atol=2e-7)
            for index in range(4):
                for ratio in (0.0, 0.37, 1.0):
                    with self.subTest(shifts=shifts, index=index, ratio=ratio):
                        self.roles.zero_grad()
                        oracle_roles.zero_grad()
                        actual, stats = student_objective(self.student, self.fake, self.teacher, actual_trajectory, index, ratio)
                        expected, reference_stats = engine.student_step_loss(backend, reference, index, ratio)
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                        actual.backward()
                        expected.backward()
                        torch.testing.assert_close(self.roles.denoisers["student"].weights.grad, oracle_roles.denoisers["student"].weights.grad, rtol=2e-6, atol=2e-7)
                        for roles in (self.roles, oracle_roles):
                            self.assertIsNone(roles.denoisers["fake"].weights.grad)
                            self.assertIsNone(roles.denoisers["teacher"].weights.grad)
                        for side in ("video", "audio"):
                            torch.testing.assert_close(stats[f"dmd_{side}_direction_rms"], reference_stats[f"update_rms_{side}"], rtol=2e-6, atol=2e-7)
                        self.assertEqual(int(stats["dmd_nonfinite_update"]), reference_stats["nonfinite_update"])
                        for role in ("fake", "teacher"):
                            actual_query, actual_sigmas, _, _ = self.roles.denoisers[role].calls[0]
                            expected_query, expected_sigmas, _, _ = oracle_roles.denoisers[role].calls[0]
                            torch.testing.assert_close(actual_sigmas, expected_sigmas, rtol=0, atol=0)
                            torch.testing.assert_close(actual_query.video, expected_query.video, rtol=0, atol=0)
                            torch.testing.assert_close(actual_query.audio, expected_query.audio, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
