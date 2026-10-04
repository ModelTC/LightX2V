import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.nn.functional as F

from lightx2v_train.model_zoo.minimax_h3.capability_adapters.common import (
    MiniMaxH3JointLatents,
    MiniMaxH3LatentShape,
)
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
)
from lightx2v_train.trainers.dmd.trainer import DmdTrainer


class TinyH3Model:
    """Use the real H3 loss capability without allocating a DiT."""

    transformer_component = "transformer_ref"
    device = torch.device("cpu")

    def __init__(self, coefficient):
        self.config = {"training": {"dmd": {"num_inference_steps": 4, "num_frames": 124}}}
        self.transformer = torch.nn.Linear(1, 1, bias=False)
        self.transformer.weight.data.fill_(coefficient)

    def denoiser_module(self):
        return self.transformer


class H3TrainerIntegrationTest(unittest.TestCase):
    def fixture(self, projected=False):
        shape = MiniMaxH3LatentShape(124, 37, 4, 6, 207, (1, 2, 3), (1, 3, 2))
        generated = MiniMaxH3JointLatents(
            torch.tensor([[[0.4, -0.8, 1.2], [2.0, 0.25, -1.5]]], requires_grad=True),
            torch.tensor([[[0.6, -1.3], [1.1, 0.2], [-0.4, 0.7]]], requires_grad=True),
            shape,
        )
        noise = MiniMaxH3JointLatents(
            torch.tensor([[[-0.3, 0.4, -1.0], [0.9, -0.2, 0.6]]]),
            torch.tensor([[[0.8, -0.1], [-0.5, 1.2], [0.3, -0.9]]]),
            shape,
        )
        options = {
            "projected_dmd": projected,
            "video_loss_weight": 2.0,
            "audio_loss_weight": 3.0,
            "audio_dmd_loss_weight": 0.7,
        }
        student = MiniMaxH3DistributionMatchingCapability(TinyH3Model(0.1), options)
        fake = MiniMaxH3DistributionMatchingCapability(TinyH3Model(0.4), options)
        teacher = MiniMaxH3DistributionMatchingCapability(TinyH3Model(-0.2), options)

        def velocity(capability, latents):
            coefficient = capability.model.transformer.weight.reshape(())
            return MiniMaxH3JointLatents(
                coefficient * latents.video + 0.3,
                coefficient * latents.audio - 0.2,
                latents.shape,
            )

        fake.predict_velocity = Mock(side_effect=lambda latents, sigma, condition: velocity(fake, latents))
        teacher.predict_velocity = Mock(side_effect=lambda latents, sigma, condition: velocity(teacher, latents))
        student.random_noise_like = Mock(return_value=noise)
        student.student_regularization = Mock(wraps=student.student_regularization)
        trainer = DmdTrainer.__new__(DmdTrainer)
        trainer.student, trainer.fake, trainer.teacher = student, fake, teacher
        trainer.latent_dtype = torch.float32
        trainer.scheduler = object()
        trainer.guidance_scale, trainer.cfg_norm = 1.0, "none"
        trainer._sample_score_sigma = Mock(return_value=torch.tensor([0.35]))

        def rollout(condition, latent_shape, grad_enabled, xt):
            del condition, latent_shape, xt
            result = generated if grad_enabled else MiniMaxH3JointLatents(generated.video.detach(), generated.audio.detach(), shape)
            return result, 1000, 0

        trainer.run_back_simulation = Mock(side_effect=rollout)
        return trainer, generated, noise

    @staticmethod
    def endpoints(trainer, generated, noise):
        sigma = trainer._sample_score_sigma.return_value
        with torch.no_grad():
            renoised = trainer.student.add_noise(trainer.scheduler, generated, noise, sigma)
            fake = trainer.fake.predict_velocity(renoised, sigma, {})
            teacher = trainer.teacher.predict_velocity(renoised, sigma, {})
            return (
                trainer.student.x0_from_velocity(renoised, fake, sigma),
                trainer.student.x0_from_velocity(renoised, teacher, sigma),
            )

    @staticmethod
    def legacy_modality_loss(generated, fake, teacher):
        normalizer = (generated.detach() - teacher).abs().mean(dim=(1, 2), keepdim=True)
        direction = torch.nan_to_num((fake - teacher) / normalizer)
        return 0.5 * F.mse_loss(generated, (generated - direction).detach())

    def test_disabled_pdmd_matches_legacy_av_loss_and_gradients(self):
        trainer, generated, noise = self.fixture()
        result = trainer.forward_loss(generated.shape, ({}, None), "student")
        actual_gradients = torch.autograd.grad(result["loss"], (generated.video, generated.audio))
        fake, teacher = self.endpoints(trainer, generated, noise)
        expected = 2.0 * self.legacy_modality_loss(generated.video, fake.video, teacher.video)
        expected += 0.7 * self.legacy_modality_loss(generated.audio, fake.audio, teacher.audio)
        expected_gradients = torch.autograd.grad(expected, (generated.video, generated.audio))
        torch.testing.assert_close(result["loss"], expected, rtol=0, atol=0)
        for actual, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertIsNone(trainer.fake.model.transformer.weight.grad)
        self.assertIsNone(trainer.teacher.model.transformer.weight.grad)
        trainer.student.student_regularization.assert_called_once()

    def test_pdmd_changes_each_modality_direction_without_critic_gradients(self):
        baseline, baseline_generated, _ = self.fixture()
        baseline_result = baseline.forward_loss(baseline_generated.shape, ({}, None), "student")
        baseline_gradients = torch.autograd.grad(baseline_result["loss"], (baseline_generated.video, baseline_generated.audio))
        trainer, generated, noise = self.fixture(projected=True)
        result = trainer.forward_loss(generated.shape, ({}, None), "student")
        result["loss"].backward()
        fake, _ = self.endpoints(trainer, generated, noise)
        for gradient, baseline_gradient, estimate, student in zip(
            (generated.video.grad, generated.audio.grad),
            baseline_gradients,
            (fake.video, fake.audio),
            (generated.video, generated.audio),
        ):
            self.assertGreater((gradient - baseline_gradient).abs().max().item(), 1e-4)
            torch.testing.assert_close((gradient * (estimate - student.detach())).sum(), torch.tensor(0.0), rtol=0, atol=1e-6)
        self.assertIsNone(trainer.fake.model.transformer.weight.grad)
        self.assertIsNone(trainer.teacher.model.transformer.weight.grad)
        self.assertLess(result["loss"].item(), baseline_result["loss"].item())

    def test_fake_regression_and_gradients_are_unchanged_by_pdmd(self):
        losses, gradients = [], []
        for projected in (False, True):
            trainer, generated, noise = self.fixture(projected)
            actual = trainer.forward_loss(generated.shape, ({}, None), "fake")
            sigma = trainer._sample_score_sigma.return_value
            renoised = trainer.student.add_noise(trainer.scheduler, generated, noise, sigma)
            prediction = trainer.fake.predict_velocity(renoised, sigma, {})
            expected = 2.0 * F.mse_loss(prediction.video, generated.video - noise.video)
            expected += 3.0 * F.mse_loss(prediction.audio, generated.audio - noise.audio)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.backward()
            losses.append(actual.detach())
            gradients.append(trainer.fake.model.transformer.weight.grad.clone())
            self.assertIsNone(generated.video.grad)
            self.assertIsNone(generated.audio.grad)
            self.assertIsNone(trainer.teacher.model.transformer.weight.grad)
            trainer.student.student_regularization.assert_not_called()
            self.assertFalse(trainer.run_back_simulation.call_args.kwargs["grad_enabled"])
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)

    def test_diagnostics_are_detached_and_refreshed_per_forward(self):
        trainer, generated, _ = self.fixture(projected=True)
        first = trainer.forward_loss(generated.shape, ({}, None), "student")
        names = {
            "dmd_video",
            "dmd_audio",
            "dmd_video_normalizer",
            "dmd_audio_normalizer",
            "dmd_video_direction_rms",
            "dmd_audio_direction_rms",
        }
        self.assertEqual(set(first), names | {"loss", "dmd"})
        torch.testing.assert_close(first["dmd"], first["dmd_video"] + first["dmd_audio"])
        for name in names | {"dmd"}:
            self.assertFalse(first[name].requires_grad)
            self.assertEqual(first[name].dtype, torch.float32)
        with torch.no_grad():
            trainer.fake.model.transformer.weight.fill_(0.9)
        second = trainer.forward_loss(generated.shape, ({}, None), "student")
        self.assertNotEqual(first["dmd_video_direction_rms"].item(), second["dmd_video_direction_rms"].item())
        self.assertEqual(set(trainer.student.dmd_metrics()), names)

    def test_disabled_adv_and_memory_hooks_are_noops(self):
        trainer, generated, _ = self.fixture()
        student = trainer.student
        initial_state = student.adv_regularizer.state_dict()
        with patch("torch.cuda.reset_peak_memory_stats") as reset:
            self.assertIsNone(student.student_regularization(generated, None, {}, trainer.scheduler, lambda value: value))
            student.on_iteration_start(0)
            student.after_optimizer_step("fake")
            student.after_optimizer_step("student")
            student.on_iteration_end(1)
        reset.assert_not_called()
        self.assertEqual(student.adv_regularizer.state_dict(), initial_state)
        self.assertEqual(student.extra_training_state(), {})
        student.load_extra_training_state({})

    def test_route_scale_changes_accumulated_gradient_not_reported_loss(self):
        trainer = DmdTrainer.__new__(DmdTrainer)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        trainer.optimizer = torch.optim.SGD([parameter], lr=0.1)
        trainer.lr_scheduler = Mock()
        trainer.trainable_params = [parameter]
        trainer.max_grad_norm = 1e6
        trainer._set_student_gradient_sync = Mock()
        trainer._sync_sequence_parallel_grads = Mock()
        trainer._after_student_optimizer_step = Mock()
        trainer.real_data_fake_trick = SimpleNamespace(enabled_for=lambda region: False)
        trainer.diversity_trick = SimpleNamespace(enabled=False)
        sampler = SimpleNamespace(is_minimax_h3_task_cycle_sampler=True, microbatch_loss_scale=Mock(side_effect=[0.25, 2.0]))
        trainer.dataloader_train = SimpleNamespace(sampler=sampler)
        trainer._encode_conditions = Mock(return_value=({}, None))
        trainer._latent_shape = Mock(return_value=None)
        trainer.sample_initial_latents = Mock(return_value=None)
        trainer.forward_loss = Mock(
            side_effect=[
                {"loss": 2.0 * parameter, "dmd": torch.tensor(2.0), "dmd_video": torch.tensor(1.0)},
                {"loss": 3.0 * parameter, "dmd": torch.tensor(3.0), "dmd_video": torch.tensor(2.0)},
            ]
        )
        result = trainer._train_one_stage(iter([{}, {}]), "student", 2, outer_iteration=7)
        torch.testing.assert_close(parameter, torch.tensor(0.675))
        self.assertEqual(result["loss"], 2.5)
        self.assertEqual(result["dmd_video"], 1.5)
        self.assertEqual(
            sampler.microbatch_loss_scale.call_args_list,
            [
                call(outer_iteration=7, stage="student", micro_step=0, fake_update_index=0),
                call(outer_iteration=7, stage="student", micro_step=1, fake_update_index=0),
            ],
        )
        self.assertEqual(trainer._set_student_gradient_sync.call_args_list, [call(False), call(True)])
        trainer._after_student_optimizer_step.assert_called_once_with("main")
        trainer.lr_scheduler.step.assert_called_once()

    def test_resumed_sampler_is_configured_before_iteration(self):
        trainer = DmdTrainer.__new__(DmdTrainer)
        trainer._resolve_resume = Mock(return_value=("checkpoint-000000007", 7))
        trainer.setup = Mock()
        trainer.max_train_iters = 7
        trainer.gradient_accumulation_iters = 2
        trainer.fake_update_ratio = 5
        trainer.save_every_iters, trainer.save_total_limit = 0, 1
        trainer.training_config = {"method": "dmd"}
        trainer.student_train_type, trainer.fake_train_type = "lora", "full"
        trainer.train_log_every_iters, trainer.infer_every_iters = 1, 0
        trainer.diversity_trick = SimpleNamespace(enabled=False, config=SimpleNamespace(weight=0.0, teacher_inference_steps=4, anchor_step=0))
        trainer.real_data_fake_trick = SimpleNamespace(
            config=SimpleNamespace(
                regions={
                    "main": SimpleNamespace(enabled=False, weight=0.0, timestep_list=()),
                }
            )
        )
        events = Mock()
        sampler = SimpleNamespace(is_minimax_h3_ref_cost_sampler=True, configure=events.configure)
        trainer.dataloader_train = SimpleNamespace(sampler=sampler)
        trainer._iter_train_samples = events.iterate
        trainer._iter_train_samples.return_value = iter(())
        with tempfile.TemporaryDirectory() as directory:
            trainer.output_train_dir = directory
            trainer.train()
        self.assertEqual(
            events.mock_calls,
            [
                call.configure(start_iteration=7, gradient_accumulation_iters=2, fake_update_ratio=5),
                call.iterate(),
            ],
        )
        trainer.setup.assert_called_once_with(resume_ckpt_path="checkpoint-000000007")


if __name__ == "__main__":
    unittest.main()
