"""CPU integration checks for optimizer cadence, reuse and resume cursors."""

import copy
import itertools
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lightx2v_train.trainers.dmd.math import official_pdmd_critic_loss, official_pdmd_loss_with_stats
from lightx2v_train.trainers.dmd.official_pdmd_core import critic_updates_before, role_at, rollout_grid
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining, validate_official_pdmd_config


@dataclass(frozen=True)
class Joint:
    video: torch.Tensor
    audio: torch.Tensor
    shape: tuple = ()


class ToyCapability:
    device = torch.device("cpu")

    def __init__(self, value):
        self.parameter = torch.nn.Parameter(torch.tensor(value))
        self.calls = []

    def set_training(self, enabled):
        assert not enabled

    def predict_velocity(self, state, sigmas, condition):
        self.calls.append((condition["id"], torch.is_grad_enabled(), sigmas.clone()))
        pattern = torch.tensor([[[1.0, -0.7, 0.3, -0.5]]])
        return Joint(self.parameter * state.video + pattern, self.parameter * state.audio - pattern)

    @staticmethod
    def add_noise(scheduler, clean, noise, sigmas):
        return Joint((1 - sigmas[0]) * clean.video + sigmas[0] * noise.video, (1 - sigmas[1]) * clean.audio + sigmas[1] * noise.audio)

    @staticmethod
    def training_target(clean, noise):
        return Joint(clean.video - noise.video, clean.audio - noise.audio)

    @staticmethod
    def regression_loss(prediction, target):
        return 0.8 * (official_pdmd_critic_loss(prediction.video, target.video) + official_pdmd_critic_loss(prediction.audio, target.audio))

    @staticmethod
    def x0_from_velocity(state, velocity, sigmas):
        return Joint(state.video + sigmas[0] * velocity.video, state.audio + sigmas[1] * velocity.audio)

    @staticmethod
    def detach(state):
        return Joint(state.video.detach(), state.audio.detach())

    @staticmethod
    def dmd_loss(clean, fake, teacher):
        return 0.8 * (official_pdmd_loss_with_stats(clean.video, fake.video, teacher.video)[0] + official_pdmd_loss_with_stats(clean.audio, fake.audio, teacher.audio)[0])

    @staticmethod
    def dmd_metrics():
        return {}


def fixture(accum=1, output="unused"):
    config = {
        "seed": 42,
        "model": {"name": "minimax_h3_ref2av", "capabilities": {"distribution_matching": {"official_pdmd": True}}},
        "training": {
            "method": "dmd",
            "gradient_accumulation_iters": accum,
            "dmd": {"official_pdmd": True, "num_inference_steps": 4, "fake_update_ratio": 5, "update_order": "fake_first", "model_mode": "eval"},
        },
        "data": {"train": {"name": "minimax_h3_ref_cache_dataset"}},
        "scheduler": {},
    }
    student, fake, teacher = [ToyCapability(value) for value in (0.3, -0.2, 0.7)]
    teacher.parameter.requires_grad_(False)
    optimizer = torch.optim.SGD([student.parameter], lr=0.01)
    fake_optimizer = torch.optim.SGD([fake.parameter], lr=0.01)
    sync = []
    encoded = []

    def encode(sample):
        encoded.append(sample)
        return {"id": sample, "cached_bf16": torch.ones(2, dtype=torch.bfloat16)}, None

    def save_checkpoint(iteration, limit):
        path = Path(output) / f"checkpoint-{iteration:09d}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"student": student.parameter.detach(), "fake": fake.parameter.detach()}, path / "toy_weights.pt")

    trainer = SimpleNamespace(
        config=config,
        dmd_config=config["training"]["dmd"],
        fake_update_ratio=5,
        gradient_accumulation_iters=accum,
        num_inference_steps=4,
        student=student,
        fake=fake,
        teacher=teacher,
        optimizer=optimizer,
        fake_optimizer=fake_optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        fake_lr_scheduler=torch.optim.lr_scheduler.LambdaLR(fake_optimizer, lambda _: 1),
        trainable_params=[student.parameter],
        fake_trainable_params=[fake.parameter],
        max_grad_norm=1.0,
        _set_student_gradient_sync=lambda enabled: sync.append(("student", enabled)),
        _set_fake_gradient_sync=lambda enabled: sync.append(("fake", enabled)),
        _sync_sequence_parallel_grads=lambda params: None,
        _after_student_optimizer_step=lambda role: None,
        _encode_conditions=encode,
        _latent_shape=lambda sample: (),
        sample_initial_latents=lambda shape: Joint(torch.randn(1, 1, 4), torch.randn(1, 1, 4)),
        output_train_dir=output,
        save_total_limit=5,
        save_checkpoint=save_checkpoint,
    )
    runner = OfficialH3PdmdTraining(trainer)
    runner.video_grid = rollout_grid(4)
    runner.audio_grid = rollout_grid(4)
    return runner, trainer, encoded, sync


class OfficialPdmdTrainingTest(unittest.TestCase):
    def test_role_count_and_data_consumption(self):
        self.assertEqual([role_at(i) for i in range(12)], ["fake"] * 5 + ["student"] + ["fake"] * 5 + ["student"])
        self.assertEqual(critic_updates_before(2500), 2084)
        self.assertEqual(critic_updates_before(5), critic_updates_before(6))
        for bad in (-1,):
            with self.assertRaises(ValueError):
                role_at(bad)

    def test_trainer_dispatch_runs_single_update_loop_and_saves_final_cursor(self):
        from lightx2v_train.trainers.dmd.trainer import DmdTrainer

        with tempfile.TemporaryDirectory() as output:
            runner, trainer, encoded, _ = fixture(output=output)
            events = []
            sampler = SimpleNamespace(configure_from_microbatch_offset=lambda **kwargs: events.append(("sampler", kwargs)))
            trainer.dataloader_train = SimpleNamespace(sampler=sampler)
            trainer._resolve_resume = lambda: (None, 0)
            trainer.setup = lambda **kwargs: events.append(("setup", kwargs))
            trainer._iter_train_samples = lambda: itertools.count()
            trainer.student.on_iteration_start = Mock()
            trainer.student.on_iteration_end = Mock()
            trainer.log_metrics = Mock()
            trainer.train_log_every_iters = 1
            trainer.max_train_iters = 6
            trainer.save_every_iters = 0
            DmdTrainer.train(trainer)
            self.assertEqual([event[0] for event in events], ["sampler", "setup"])
            self.assertEqual(events[0][1]["start_microbatch"], 0)
            self.assertEqual(encoded, list(range(5)))
            self.assertEqual(trainer.student.on_iteration_start.call_count, 6)
            self.assertEqual(trainer.lr_scheduler.last_epoch, 1)
            self.assertEqual(trainer.fake_lr_scheduler.last_epoch, 5)
            self.assertEqual(trainer.log_metrics.call_count, 6)
            checkpoint = Path(output) / "checkpoint-000000006"
            runner._load_cursor(checkpoint, 6)
            self.assertEqual(runner.consumed_microbatches, 5)
            self.assertEqual(runner.cached, [])

    def test_six_updates_reuse_last_critic_batch_and_only_step_correct_role(self):
        runner, trainer, encoded, sync = fixture(accum=2)
        student_start = trainer.student.parameter.detach().clone()
        samples = itertools.count()
        for iteration in range(5):
            role, metrics = runner._update(samples, iteration)
            self.assertEqual(role, "fake")
            torch.testing.assert_close(trainer.student.parameter, student_start, rtol=0, atol=0)
            self.assertEqual(len(encoded), 2 * (iteration + 1))
            self.assertEqual(len(runner.cached), 2 if iteration == 4 else 0)
        fake_end = trainer.fake.parameter.detach().clone()
        self.assertEqual([tr.condition["id"] for tr in runner.cached], [8, 9])
        self.assertTrue(all(not state.video.requires_grad for tr in runner.cached for state in tr.states))
        role, metrics = runner._update(samples, 5)
        self.assertEqual(role, "student")
        self.assertEqual(encoded, list(range(10)))
        self.assertEqual(runner.consumed_microbatches, 10)
        torch.testing.assert_close(trainer.fake.parameter, fake_end, rtol=0, atol=0)
        self.assertFalse(torch.equal(trainer.student.parameter, student_start))
        self.assertEqual(trainer.lr_scheduler.last_epoch, 1)
        self.assertEqual(trainer.fake_lr_scheduler.last_epoch, 5)
        self.assertEqual(runner.cached, [])
        self.assertEqual(sync, [("fake", x) for _ in range(5) for x in (False, True)] + [("student", False), ("student", True)])
        differentiable_student = [call for call in trainer.student.calls if call[1]]
        self.assertEqual([call[0] for call in differentiable_student], [8, 9])
        self.assertEqual(float(differentiable_student[0][2][0]), float(differentiable_student[1][2][0]))
        self.assertIsNone(trainer.teacher.parameter.grad)
        self.assertIsNone(trainer.fake.parameter.grad)

    def test_resume_before_student_reuses_serialized_trajectory_without_reading(self):
        with tempfile.TemporaryDirectory() as output:
            original, trainer, _, _ = fixture(accum=2, output=output)
            samples = itertools.count()
            for iteration in range(5):
                original._update(samples, iteration)
            original._save(5)
            resumed, other, encoded, _ = fixture(accum=2, output=output)
            path = Path(output) / "checkpoint-000000005"
            resumed._load_cursor(path, 5)
            with torch.no_grad():
                other.student.parameter.copy_(trainer.student.parameter)
                other.fake.parameter.copy_(trainer.fake.parameter)
            self.assertEqual(resumed.cached[0].condition["cached_bf16"].dtype, torch.bfloat16)
            role_a, metrics_a = original._update(samples, 5)
            role_b, metrics_b = resumed._update(iter(()), 5)
            self.assertEqual(role_a, role_b)
            self.assertEqual(metrics_a, metrics_b)
            torch.testing.assert_close(trainer.student.parameter, other.student.parameter, rtol=0, atol=0)
            self.assertEqual(encoded, [])
            self.assertEqual(resumed.consumed_microbatches, 10)

    def test_resume_rejects_missing_marker_recipe_changes_and_missing_cached_state(self):
        with tempfile.TemporaryDirectory() as output:
            runner, trainer, _, _ = fixture(output=output)
            samples = itertools.count()
            for iteration in range(5):
                runner._update(samples, iteration)
            path = Path(output) / "checkpoint-000000005"
            runner._save(5)
            changed, _, _, _ = fixture(accum=2, output=output)
            with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
                changed._load_cursor(path, 5)
            sidecar = path / "official_pdmd.rank00000.pt"
            state = torch.load(sidecar, weights_only=False)
            state["cached"] = []
            torch.save(state, sidecar)
            with self.assertRaisesRegex(RuntimeError, "trajectory missing"):
                runner._load_cursor(path, 5)
            (path / "official_pdmd.complete").unlink()
            with self.assertRaisesRegex(RuntimeError, "complete official-PDMD"):
                runner._load_cursor(path, 5)

    def test_nonfinite_gradients_fail_before_optimizer_step(self):
        runner, trainer, _, _ = fixture()
        initial = trainer.fake.parameter.detach().clone()
        handle = trainer.fake.parameter.register_hook(lambda gradient: gradient * float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "nonfinite fake gradient"):
            runner._update(itertools.count(), 0)
        torch.testing.assert_close(trainer.fake.parameter, initial, rtol=0, atol=0)
        handle.remove()

    def test_unsupported_features_rejected_without_loading_model(self):
        runner, trainer, _, _ = fixture()
        validate_official_pdmd_config(trainer.config)
        for name, value in (("residual_head", {"enabled": True}), ("update_order", "student_first")):
            config = copy.deepcopy(trainer.config)
            config["training"]["dmd"][name] = value
            with self.assertRaises(ValueError):
                validate_official_pdmd_config(config)


if __name__ == "__main__":
    unittest.main()
