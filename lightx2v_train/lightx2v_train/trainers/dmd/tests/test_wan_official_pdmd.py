"""Wan's released PDMD objective, trajectory, CFG, cadence and resume boundary."""

import importlib
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch.utils.data import DataLoader, Dataset

from lightx2v_train.model_zoo.wan.capability_adapters.wan_distribution_matching_capability import WanDistributionMatchingCapability
from lightx2v_train.trainers.dmd.official_pdmd_core import PdmdCondition, critic_objective, critic_sigmas, full_rollout, joint_noise_like, rollout_grid, student_objective
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining, validate_official_pdmd_config


class ToyDenoiser(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(weights))
        self.calls = []

    def forward(self, state, sigma, condition):
        self.calls.append((state.detach().clone(), sigma.detach().clone(), condition, torch.is_grad_enabled(), self.training))
        pattern = torch.linspace(-0.4, 0.7, state.numel()).reshape_as(state)
        return self.weights[0] * state + self.weights[1] * pattern + condition["bias"] * sigma


class ModelStub(SimpleNamespace):
    pass


def roles():
    result = {}
    for name, weights in (("student", [0.17, 0.07]), ("fake", [-0.03, -0.11]), ("teacher", [0.21, 0.16])):
        denoiser = ToyDenoiser(weights)
        if name == "teacher":
            denoiser.requires_grad_(False)
        model = ModelStub(
            config={"model": {"capabilities": {"distribution_matching": {"official_pdmd": True, "video_loss_weight": 0.8}}}},
            device=torch.device("cpu"),
            running_dtype=torch.float32,
            use_causal_transformer=False,
            denoiser_module=lambda denoiser=denoiser: denoiser,
            predict_denoiser_output=denoiser,
        )
        result[name] = WanDistributionMatchingCapability(model)
        result[name]._test_model_keepalive = model
    return result


class Samples(Dataset):
    def __init__(self):
        self.samples = [{"id": index} for index in range(13)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return index


def trainer_fixture(output="unused"):
    values = roles()
    student, fake, teacher = (values[name] for name in ("student", "fake", "teacher"))
    student_parameters = list(student.denoiser().parameters())
    fake_parameters = list(fake.denoiser().parameters())
    optimizer = torch.optim.SGD(student_parameters, lr=0.01)
    fake_optimizer = torch.optim.SGD(fake_parameters, lr=0.01)
    config = {
        "seed": 42,
        "model": {"name": "wan_t2v", "capabilities": {"distribution_matching": {"official_pdmd": True}}},
        "training": {
            "method": "dmd",
            "gradient_accumulation_iters": 1,
            "teacher": {"guidance_scale": 5},
            "dmd": {"official_pdmd": True, "num_inference_steps": 8, "fake_update_ratio": 5, "update_order": "fake_first", "model_mode": "eval"},
        },
        "data": {"train": {"name": "video_dataset", "batch_size": 1}},
        "scheduler": {},
    }
    encoded = []

    def encode(sample):
        encoded.append(sample)
        return {"bias": 0.04, "id": sample}, {"bias": -0.02, "id": sample}

    def save_checkpoint(iteration, limit):
        (Path(output) / f"checkpoint-{iteration:09d}").mkdir(parents=True, exist_ok=True)

    trainer = SimpleNamespace(
        config=config,
        dmd_config=config["training"]["dmd"],
        fake_update_ratio=5,
        gradient_accumulation_iters=1,
        num_inference_steps=8,
        student=student,
        fake=fake,
        teacher=teacher,
        guidance_scale=5.0,
        cfg_norm="none",
        optimizer=optimizer,
        fake_optimizer=fake_optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        fake_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(fake_optimizer, lambda _: 1),
        trainable_params=student_parameters,
        fake_trainable_params=fake_parameters,
        max_grad_norm=1.0,
        _set_student_gradient_sync=lambda enabled: None,
        _set_fake_gradient_sync=lambda enabled: None,
        _sync_sequence_parallel_grads=lambda parameters: None,
        _after_student_optimizer_step=lambda role: None,
        _encode_conditions=encode,
        _latent_shape=lambda sample: (1, 1, 2, 2, 2),
        sample_initial_latents=lambda shape: torch.randn(shape),
        output_train_dir=output,
        save_total_limit=3,
        save_checkpoint=save_checkpoint,
        dataloader_train=DataLoader(Samples(), batch_size=1, collate_fn=lambda items: items[0]),
    )
    runner = OfficialH3PdmdTraining(trainer)
    runner.video_grid = rollout_grid(8, 5)
    runner.audio_grid = rollout_grid(8)
    return runner, trainer, encoded


class WanOfficialPdmdTests(unittest.TestCase):
    def test_native_noise_minus_clean_full_rollout_and_guided_teacher_only(self):
        values = roles()
        noise = torch.randn(1, 1, 2, 2, 2)
        grid = rollout_grid(8, 5)
        condition = PdmdCondition({"bias": 0.04}, {"bias": -0.02}, 5.0, "none")
        trajectory = full_rollout(values["student"], noise, condition, grid, grid)
        self.assertEqual(len(trajectory.states), 9)
        expected = noise.clone()
        for index, call in enumerate(values["student"].denoiser().calls):
            state, sigma, positive, grad, training = call
            torch.testing.assert_close(state, expected, rtol=0, atol=0)
            self.assertFalse(grad or training)
            velocity = values["student"].model.predict_denoiser_output(expected, sigma, positive)
            expected = expected + (grid[index + 1] - grid[index]) * velocity
            if index == 7:
                break
        torch.testing.assert_close(trajectory.states[-1], expected, rtol=0, atol=0)
        for value in values.values():
            value.denoiser().calls.clear()
        loss, metrics = student_objective(values["student"], values["fake"], values["teacher"], trajectory, 3, 0.37)
        loss.backward()
        self.assertEqual([len(value.denoiser().calls) for value in values.values()], [1, 1, 2])
        self.assertIsNotNone(values["student"].denoiser().weights.grad)
        self.assertIsNone(values["fake"].denoiser().weights.grad)
        self.assertIsNone(values["teacher"].denoiser().weights.grad)
        self.assertNotIn("score_sigma_audio", metrics)
        self.assertGreater(float(values["student"].denoiser().weights.grad.abs().sum()), 0)

    def test_five_fake_one_student_reuses_final_critic_trajectory(self):
        runner, trainer, encoded = trainer_fixture()
        samples = itertools.count()
        initial = trainer.student.denoiser().weights.detach().clone()
        for index in range(5):
            role, metrics = runner._update(samples, index)
            self.assertEqual(role, "fake")
            self.assertNotIn("score_sigma_audio", metrics)
            torch.testing.assert_close(trainer.student.denoiser().weights, initial, rtol=0, atol=0)
        self.assertEqual(encoded, list(range(5)))
        self.assertEqual(len(runner.cached), 1)
        self.assertEqual(runner.cached[0].condition.positive["id"], 4)
        fake = trainer.fake.denoiser().weights.detach().clone()
        self.assertEqual(runner._update(samples, 5)[0], "student")
        self.assertEqual(encoded, list(range(5)))
        self.assertEqual(runner.cached, [])
        torch.testing.assert_close(trainer.fake.denoiser().weights, fake, rtol=0, atol=0)
        self.assertFalse(torch.equal(trainer.student.denoiser().weights, initial))

    def test_video_only_sigma_is_uniform_without_audio_draws(self):
        for seed in range(12):
            actual = critic_sigmas(torch.Generator().manual_seed(seed), video_only=True)
            expected = torch.rand(1, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_weight_applied_after_clamp_and_legacy_recipe_rejected(self):
        value = roles()["student"]
        prediction = torch.full((1, 2), 100.0, requires_grad=True)
        loss = value.regression_loss(prediction, torch.zeros_like(prediction))
        self.assertEqual(float(loss), 4.0)
        loss.backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0)
        value.model.config["model"]["capabilities"]["distribution_matching"] = {"projected_dmd": True}
        with self.assertRaisesRegex(ValueError, "removed"):
            WanDistributionMatchingCapability(value.model)

    def test_generic_cursor_resume_and_euler_preview_cadence(self):
        with tempfile.TemporaryDirectory() as output:
            runner, trainer, encoded = trainer_fixture(output)
            trainer._resolve_resume = lambda: (None, 0)
            trainer.setup = Mock()
            trainer._iter_train_samples = lambda: itertools.cycle(trainer.dataloader_train)
            trainer.student.on_iteration_start = Mock()
            trainer.student.on_iteration_end = Mock()
            trainer.log_metrics = Mock()
            trainer.train_log_every_iters = 1
            trainer.max_train_iters = 12
            trainer.save_every_iters = 6
            trainer.infer_every_iters = 6
            trainer.infer_config = {"run_at_start": True}
            trainer.dataloader_val = object()
            trainer.inferencer = SimpleNamespace(set_data=Mock())
            trainer.run_inference = Mock()
            runner._validate_wan_residency = Mock()
            runner.run()
            self.assertEqual([call.args[0] for call in trainer.run_inference.call_args_list], [0, 6, 12])
            self.assertEqual(len(encoded), 10)
            self.assertEqual(trainer.lr_scheduler.last_epoch, 2)
            self.assertEqual(trainer.fake_lr_scheduler.last_epoch, 10)
            self.assertFalse(trainer.student.denoiser().training)
            resumed = OfficialH3PdmdTraining(trainer)
            resumed._load_cursor(Path(output) / "checkpoint-000000012", 12)
            self.assertEqual(resumed.consumed_microbatches, 10)
            self.assertEqual(resumed.cached, [])
            validate_official_pdmd_config(trainer.config)

    @unittest.skipUnless(os.environ.get("PDMD_REFERENCE_SOURCE"), "released PDMD source checkout required")
    def test_direct_released_engine_video_parity_all_exits_and_gradients(self):
        source = Path(os.environ["PDMD_REFERENCE_SOURCE"]) / "src"
        sys.path.insert(0, str(source))
        try:
            engine = importlib.import_module("pdmd.engine")
        finally:
            sys.path.remove(str(source))
        actual_roles, reference_roles = roles(), roles()
        condition = PdmdCondition({"bias": 0.04}, {"bias": -0.02}, 5.0, "none")

        class Backend:
            dit = torch.nn.Module()

            def predict(self, role, states, sigmas, query_condition):
                value = reference_roles["fake" if role == "critic" else role]
                if role == "teacher":
                    prediction = value.predict_guided_velocity(
                        states[0], sigmas[0].reshape(1), query_condition.positive, query_condition.negative, query_condition.guidance_scale, query_condition.cfg_norm
                    )
                else:
                    prediction = value.predict_velocity(states[0], sigmas[0].reshape(1), query_condition.positive)
                return prediction, torch.zeros_like(states[1])

        noise = torch.randn(1, 1, 2, 2, 2, generator=torch.Generator().manual_seed(87))
        fresh = joint_noise_like(noise, torch.Generator().manual_seed(89))
        grid = rollout_grid(8, 5)
        actual = full_rollout(actual_roles["student"], noise, condition, grid, grid)
        reference = engine.rollout(Backend(), (noise, torch.zeros_like(noise)), condition, grid, grid)
        for state, expected in zip(actual.states, reference.states):
            torch.testing.assert_close(state, expected[0], rtol=0, atol=0)
        for sigma in (0.12, 0.97):
            for collection in (actual_roles, reference_roles):
                for value in collection.values():
                    value.denoiser().zero_grad(set_to_none=True)
            actual_loss = critic_objective(actual_roles["student"], actual_roles["fake"], actual, torch.tensor([sigma]), fresh)
            expected, _ = engine.critic_step_loss(Backend(), reference, (torch.tensor(sigma), torch.tensor(sigma)), (fresh, torch.zeros_like(fresh)), weights=(0.8, 0.0))
            torch.testing.assert_close(actual_loss, expected, rtol=0, atol=0)
            actual_loss.backward()
            expected.backward()
            torch.testing.assert_close(actual_roles["fake"].denoiser().weights.grad, reference_roles["fake"].denoiser().weights.grad, rtol=0, atol=0)
        for index in range(8):
            for ratio in (0.0, 0.37, 1.0):
                with self.subTest(index=index, ratio=ratio):
                    for collection in (actual_roles, reference_roles):
                        for value in collection.values():
                            value.denoiser().zero_grad(set_to_none=True)
                    actual_loss, metrics = student_objective(actual_roles["student"], actual_roles["fake"], actual_roles["teacher"], actual, index, ratio)
                    expected, stats = engine.student_step_loss(Backend(), reference, index, ratio, weights=(0.8, 0.0))
                    torch.testing.assert_close(actual_loss, expected, rtol=0, atol=0)
                    actual_loss.backward()
                    expected.backward()
                    torch.testing.assert_close(actual_roles["student"].denoiser().weights.grad, reference_roles["student"].denoiser().weights.grad, rtol=0, atol=0)
                    torch.testing.assert_close(metrics["dmd_video_direction_rms"], stats["update_rms_video"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
