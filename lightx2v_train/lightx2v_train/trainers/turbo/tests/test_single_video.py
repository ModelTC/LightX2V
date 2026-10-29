"""Single-video Turbo objectives, CFG teachers and preview cadence."""

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lightx2v_train.trainers.dmd.trainer import DmdTrainer
from lightx2v_train.trainers.turbo.config import TurboConfig
from lightx2v_train.trainers.turbo.cursor import TurboCursorSampler
from lightx2v_train.trainers.turbo.sampler import TurboCondition, sample_turbo_latents
from lightx2v_train.trainers.turbo.tests.test_trainer import make_trainer, save_toy_roles
from lightx2v_train.trainers.turbo.trainer import TurboTrainer


class VideoCapability:
    device = torch.device("cpu")

    def __init__(self, value):
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.conditions = []

    def predict_velocity_at_sigmas(self, state, sigma, condition):
        assert sigma.ndim == 0
        self.conditions.append(condition)
        return self.weight * state.sin() + sigma + condition

    def predict_velocity_jvp(self, state, sigma, condition, tangent, time_tangent):
        primal, derivative = torch.func.jvp(lambda x, t: self.predict_velocity_at_sigmas(x, t, condition), (state, sigma), (tangent, time_tangent))
        return primal.detach(), derivative.detach()


class SingleVideoTurboTests(unittest.TestCase):
    def test_tensor_sampler_uses_scalar_physical_sigma_and_last_call_grad(self):
        capability = VideoCapability(0.1)
        state = torch.randn(1, 2, 3, 4, 4)
        output = sample_turbo_latents(capability, state, 0.0, num_steps=8, video_shift=5, with_grad=True)
        self.assertEqual(output.shape, state.shape)
        self.assertEqual(len(capability.conditions), 8)
        output.square().sum().backward()
        self.assertIsNotNone(capability.weight.grad)
        self.assertTrue(torch.isfinite(capability.weight.grad))

    def test_all_four_objectives_support_video_only_and_cfg_teacher(self):
        trainer = TurboTrainer.__new__(TurboTrainer)
        trainer.turbo = TurboConfig(num_inference_steps=2, dcm_steps=1, scm_steps=1, joint_steps=1, video_shift=5)
        trainer.student, trainer.fake, trainer.teacher = (VideoCapability(value) for value in (0.1, 0.2, 0.3))
        trainer.guidance_scale = 5.0
        condition = TurboCondition(0.5, -0.5)
        clean = torch.randn(1, 2, 3, 4, 4)
        for objective in (
            lambda: trainer._dcm_loss(clean, condition),
            lambda: trainer._scm_loss(clean, condition, 2),
            lambda: trainer._dmd_loss(clean, condition),
            lambda: trainer._critic_loss(clean, condition),
        ):
            loss, metrics = objective()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all("audio" not in key for key in metrics))
            loss.backward()
        self.assertTrue(all(item == 0.5 for item in trainer.student.conditions))
        self.assertTrue(all(item == 0.5 for item in trainer.fake.conditions))
        self.assertIn(-0.5, trainer.teacher.conditions)

    def test_absolute_single_device_cursor_resumes_across_shuffle_epochs(self):
        sampler = TurboCursorSampler(list(range(7)), seed=42)
        sampler.configure_from_microbatch_offset(start_microbatch=0)
        expected = []
        for epoch in range(4):
            sampler.set_epoch(epoch)
            expected.extend(list(sampler))
        sampler.configure_from_microbatch_offset(start_microbatch=5)
        actual = []
        for epoch in range(3):
            sampler.set_epoch(epoch)
            actual.extend(list(sampler))
        self.assertEqual(actual, expected[5:26])

    def test_preview_runs_at_zero_and_every_configured_student_interval(self):
        with TemporaryDirectory() as output, patch.object(DmdTrainer, "setup"), patch.object(DmdTrainer, "save_checkpoint", new=save_toy_roles):
            trainer = make_trainer(output)
            previews = []
            trainer.infer_every_iters = 2
            trainer.dataloader_val = None
            trainer.inferencer = SimpleNamespace(set_data=lambda loader: None)
            trainer.run_inference = previews.append
            trainer.train()
            self.assertEqual(previews, [0, 2, 4])

    def test_two_rank_cursor_partitions_global_shuffle_and_resumes(self):
        dataset = list(range(64))
        samplers = [TurboCursorSampler(dataset, seed=42, num_replicas=2, rank=rank) for rank in range(2)]
        batches = [list(sampler) for sampler in samplers]
        self.assertEqual(len(batches[0]), 32)
        self.assertEqual(set(batches[0]) & set(batches[1]), set())
        self.assertEqual(set(batches[0]) | set(batches[1]), set(dataset))
        for rank, sampler in enumerate(samplers):
            resumed = TurboCursorSampler(dataset, seed=42, num_replicas=2, rank=rank)
            resumed.configure_from_microbatch_offset(start_microbatch=11)
            self.assertEqual(list(resumed)[:21], batches[rank][11:])


if __name__ == "__main__":
    unittest.main()
