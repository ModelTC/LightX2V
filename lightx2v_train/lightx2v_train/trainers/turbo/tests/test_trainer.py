"""Exercise the real Turbo loop and resume contract with coupled CPU models.

Only heavyweight role loading and DMD's model/optimizer serializer are replaced.
The curriculum, objectives, true joint JVP, optimizer steps, cursor validation,
completion markers and RNG streams all execute production Turbo code.
"""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock, patch

import torch

from lightx2v_train.trainers.dmd.trainer import DmdTrainer
from lightx2v_train.trainers.turbo.config import TurboConfig
from lightx2v_train.trainers.turbo.trainer import TurboTrainer


class Joint(NamedTuple):
    video: torch.Tensor
    audio: torch.Tensor
    shape: object


class ToyCapability(torch.nn.Module):
    """Small, genuinely coupled AV velocity model with an exact forward JVP."""

    def __init__(self, values, *, frozen=False):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(values), requires_grad=not frozen)
        self.device = torch.device("cpu")
        self.started, self.ended, self.forward_modes = [], [], []

    def set_training(self, enabled):
        self.train(enabled)

    def on_iteration_start(self, iteration):
        self.started.append(iteration)

    def on_iteration_end(self, iteration):
        self.ended.append(iteration)

    def extract_turbo_latents(self, sample, shape):
        assert sample["clean"].shape == shape
        return sample["clean"]

    def _forward(self, video, audio, sigmas, condition):
        # Match the real H3 boundary: stable trig math outside, FP32 native RF.
        video, audio, sigmas = video.float(), audio.float(), sigmas.float()
        return (
            self.weights[0] * video.tanh() + 0.07 * audio.mean() + 0.1 * sigmas[0] + condition,
            self.weights[1] * audio.sin() - 0.03 * video.mean() - 0.15 * sigmas[1] - condition,
        )

    def predict_velocity_at_sigmas(self, state, sigma_pair, condition):
        self.forward_modes.append(torch.is_grad_enabled())
        video, audio = self._forward(state.video, state.audio, sigma_pair, condition)
        return Joint(video, audio, state.shape)

    def predict_velocity_jvp(self, state, sigma_pair, condition, tangent, sigma_tangent):
        with torch.no_grad():
            primal, derivative = torch.func.jvp(
                lambda video, audio, time: self._forward(video, audio, time, condition),
                (state.video, state.audio, sigma_pair),
                (tangent.video, tangent.audio, sigma_tangent),
            )
        return Joint(*(item.detach() for item in primal), state.shape), Joint(*(item.detach() for item in derivative), state.shape)


class CursorSampler:
    def __init__(self):
        self.cursor = 0
        self.configuration = None

    def configure_from_microbatch_offset(self, **options):
        self.configuration = options
        self.cursor = options["start_microbatch"]

    def cursor_microbatch_loss_scale(self, cursor):
        # Nontrivial deterministic route weights also have to survive resume.
        return 0.9 + 0.1 * (cursor % 3)


def data_stream(trainer):
    sampler = trainer.dataloader_train.sampler
    while True:
        index = sampler.cursor
        sampler.cursor += 1
        trainer.read_ids.append(index)
        generator = torch.Generator().manual_seed(9000 + index)
        clean = Joint(torch.randn(1, 2, 2, generator=generator), torch.randn(1, 3, 2, generator=generator), "toy_av")
        yield {"clean": clean, "condition": 0.01 * (index % 7), "index": index}


def state_snapshot(trainer):
    return copy.deepcopy(
        {
            "student": trainer.student.state_dict(),
            "fake": trainer.fake.state_dict(),
            "teacher": trainer.teacher.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "fake_optimizer": trainer.fake_optimizer.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
            "fake_scheduler": trainer.fake_lr_scheduler.state_dict(),
            "ema": trainer.ema,
            "ema_updates": trainer.ema_updates,
            "consumed_microbatches": trainer.consumed_microbatches,
            "fake_updates": trainer.fake_updates,
        }
    )


def save_toy_roles(trainer, iteration, save_total_limit):
    directory = Path(trainer.output_train_dir) / f"checkpoint-{iteration:09d}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state_snapshot(trainer), directory / "toy_state.pt")


def load_toy_roles(trainer, directory):
    state = torch.load(Path(directory) / "toy_state.pt", map_location="cpu", weights_only=False)
    for name in ("student", "fake", "teacher"):
        getattr(trainer, name).load_state_dict(state[name])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.fake_optimizer.load_state_dict(state["fake_optimizer"])
    trainer.lr_scheduler.load_state_dict(state["scheduler"])
    trainer.fake_lr_scheduler.load_state_dict(state["fake_scheduler"])
    trainer.ema = state["ema"]
    trainer.ema_updates = state["ema_updates"]
    # Cursor counters intentionally come only from the real Turbo sidecar.


def scheduler(trainer, optimizer, *, num_warmup_steps=0, num_training_steps=5):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - 0.3 * min(step, num_training_steps) / num_training_steps)


def update_ema(trainer, role):
    assert role == "main"
    with torch.no_grad():
        trainer.ema.mul_(0.8).add_(trainer.student.weights, alpha=0.2)
    trainer.ema_updates += 1


def make_trainer(output, *, auto_resume=False, accumulation=2):
    trainer = TurboTrainer.__new__(TurboTrainer)
    options = {"dcm_steps": 2, "scm_steps": 1, "joint_steps": 2, "tangent_warmup_steps": 1, "num_inference_steps": 8}
    trainer.public_config = {
        "seed": 123,
        "model": {"name": "minimax_h3_ref2av"},
        "training": {"method": "turbo", "max_train_iters": 5, "turbo": options, "output_dir": str(output), "gradient_accumulation_iters": accumulation},
        "data": {"train": {"name": "toy_clean_av", "data_path": "toy", "batch_size": 1, "num_workers": 0}},
        "scheduler": {"time_shift_settings": {"do_time_shift": False}},
    }
    trainer.config = copy.deepcopy(trainer.public_config)
    trainer.turbo = TurboConfig.from_mapping(trainer.public_config)
    trainer.student = ToyCapability([0.2, -0.15])
    trainer.fake = ToyCapability([0.1, -0.08])
    trainer.teacher = ToyCapability([0.35, -0.25], frozen=True)
    trainer.trainable_params = list(trainer.student.parameters())
    trainer.fake_trainable_params = list(trainer.fake.parameters())
    trainer.optimizer = torch.optim.AdamW(trainer.trainable_params, lr=1e-3, weight_decay=0.01)
    trainer.fake_optimizer = torch.optim.AdamW(trainer.fake_trainable_params, lr=2e-3, weight_decay=0.01)
    trainer._build_lr_scheduler = MethodType(scheduler, trainer)
    trainer.lr_scheduler = trainer._build_lr_scheduler(trainer.optimizer, num_training_steps=5)
    trainer.fake_lr_scheduler = trainer._build_lr_scheduler(trainer.fake_optimizer, num_training_steps=8)
    trainer._load_resume_state = MethodType(load_toy_roles, trainer)
    trainer._after_student_optimizer_step = MethodType(update_ema, trainer)
    trainer.ema, trainer.ema_updates = trainer.student.weights.detach().clone(), 0
    trainer.consumed_microbatches, trainer.fake_updates = 0, 0
    trainer.gradient_accumulation_iters = accumulation
    trainer.fake_update_ratio = trainer.turbo.fake_update_ratio
    trainer.max_train_iters = trainer.turbo.total_steps
    trainer.num_inference_steps = trainer.turbo.num_inference_steps
    trainer.max_grad_norm = 1.0
    trainer.auto_resume = auto_resume
    trainer.output_train_dir = str(output)
    trainer.save_every_iters, trainer.save_total_limit, trainer.train_log_every_iters = 1, 20, 1
    trainer.student_sync, trainer.fake_sync, trainer.read_ids, trainer.metrics = [], [], [], []
    trainer._set_student_gradient_sync = trainer.student_sync.append
    trainer._set_fake_gradient_sync = trainer.fake_sync.append
    trainer._sync_sequence_parallel_grads = lambda parameters: None
    trainer._encode_conditions = lambda sample: (sample["condition"], None)
    trainer._latent_shape = lambda sample: sample["clean"].shape
    trainer._iter_train_samples = lambda: data_stream(trainer)
    trainer.log_metrics = lambda values, step: trainer.metrics.append((step, values))
    trainer.dataloader_train = SimpleNamespace(sampler=CursorSampler(), dataset=SimpleNamespace(samples=[{"type": "metadata", "row": {"index": index}} for index in range(32)]))
    trainer.route_loss_scale = lambda cursor: 1.0
    return trainer


class TurboTrainerTests(unittest.TestCase):
    def setUp(self):
        self.setup_patch = patch.object(DmdTrainer, "setup", autospec=True)
        self.save_patch = patch.object(DmdTrainer, "save_checkpoint", new=save_toy_roles)
        self.setup_patch.start()
        self.save_patch.start()
        self.addCleanup(self.setup_patch.stop)
        self.addCleanup(self.save_patch.stop)

    def assert_nested_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right, strict=True):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_curriculum_loop_updates_data_sync_and_ema(self):
        with TemporaryDirectory() as output:
            trainer = make_trainer(output)
            initial_student = trainer.student.weights.detach().clone()
            initial_fake = trainer.fake.weights.detach().clone()
            initial_teacher = trainer.teacher.weights.detach().clone()
            trainer.train()
            self.assertEqual(trainer.read_ids, list(range(26)))
            self.assertEqual(trainer.consumed_microbatches, 26)
            self.assertEqual(trainer.fake_updates, 8)
            self.assertEqual(trainer.ema_updates, 5)
            self.assertEqual(int(trainer.optimizer.state[trainer.student.weights]["step"]), 5)
            self.assertEqual(int(trainer.fake_optimizer.state[trainer.fake.weights]["step"]), 8)
            self.assertEqual(trainer.student_sync, [False, True] * 3 + [False, False, False, True] * 2)
            self.assertEqual(trainer.fake_sync, [False, True] * 8)
            self.assertEqual(trainer.student.started, list(range(5)))
            self.assertEqual(trainer.student.ended, list(range(1, 6)))
            self.assertFalse(torch.equal(initial_student, trainer.student.weights))
            self.assertFalse(torch.equal(initial_fake, trainer.fake.weights))
            torch.testing.assert_close(initial_teacher, trainer.teacher.weights, atol=0, rtol=0)
            self.assertIsNone(trainer.teacher.weights.grad)
            self.assertTrue(all(not item.training for item in (trainer.student, trainer.fake, trainer.teacher)))
            metrics = dict(trainer.metrics)
            for step in (1, 2):
                self.assertIn("train/turbo_dcm_video", metrics[step])
            self.assertIn("train/turbo_scm_video", metrics[3])
            self.assertNotIn("train/turbo_dmd_video", metrics[3])
            for step in (4, 5):
                self.assertIn("train/turbo_scm_video", metrics[step])
                self.assertIn("train/turbo_dmd_video", metrics[step])
                self.assertIn("train/turbo_fake_critic_video", metrics[step])
            self.assertEqual(trainer.dataloader_train.sampler.configuration["data_consumption_mode"], "turbo_student_and_fake_fresh")
            self.assertEqual(trainer.dataloader_train.sampler.configuration["student_data_source"], "cached_clean_av")
            self.assertTrue((Path(output) / "checkpoint-000000005/turbo.complete.json").is_file())

    def test_role_gradients_do_not_pollute_other_optimizers(self):
        with TemporaryDirectory() as output:
            trainer = make_trainer(output)
            trainer.setup()
            samples = trainer._iter_train_samples()
            fake_before, teacher_before = trainer.fake.weights.detach().clone(), trainer.teacher.weights.detach().clone()
            trainer._optimizer_step(samples, "student", "joint", 3)
            self.assertIsNone(trainer.fake.weights.grad)
            self.assertIsNone(trainer.teacher.weights.grad)
            torch.testing.assert_close(trainer.fake.weights, fake_before, atol=0, rtol=0)
            torch.testing.assert_close(trainer.teacher.weights, teacher_before, atol=0, rtol=0)
            student_before = trainer.student.weights.detach().clone()
            trainer._optimizer_step(samples, "fake", "joint", 3)
            self.assertIsNone(trainer.student.weights.grad)
            self.assertIsNone(trainer.teacher.weights.grad)
            torch.testing.assert_close(trainer.student.weights, student_before, atol=0, rtol=0)
            self.assertEqual(trainer.fake_updates, 1)

    def test_resume_at_every_stage_boundary_and_joint_group_is_bit_exact(self):
        with TemporaryDirectory() as root:
            full = make_trainer(Path(root) / "full")
            full.train()
            expected = state_snapshot(full)
            for boundary in (2, 3, 4):
                with self.subTest(boundary=boundary):
                    output = Path(root) / f"split{boundary}"
                    first = make_trainer(output)
                    first.max_train_iters = boundary
                    first.train()
                    second = make_trainer(output, auto_resume=True)
                    torch.manual_seed(987654)  # Resume cannot depend on global RNG.
                    second.train()
                    self.assert_nested_equal(expected, state_snapshot(second))
                    self.assertEqual(first.read_ids + second.read_ids, full.read_ids)
                    self.assertEqual(second.dataloader_train.sampler.configuration["start_microbatch"], first.consumed_microbatches)

    def test_resume_rejects_bad_recipe_cursor_or_completion_metadata(self):
        import json

        with TemporaryDirectory() as output:
            original = make_trainer(output)
            original.max_train_iters = 2
            original.train()
            directory = Path(output) / "checkpoint-000000002"
            sidecar = directory / "turbo.rank00000.json"
            marker = directory / "turbo.complete.json"
            sidecar_text, marker_text = sidecar.read_text(), marker.read_text()
            resumed = make_trainer(output, auto_resume=True)
            resumed.public_config["training"]["turbo"]["audio_shift"] = 6.0
            with self.assertRaisesRegex(RuntimeError, "recipe, counters or data cursor"):
                resumed._resume_cursor(str(directory), 2)
            resumed = make_trainer(output, auto_resume=True)
            resumed.dataloader_train.dataset.samples.reverse()
            with self.assertRaisesRegex(RuntimeError, "recipe, counters or data cursor"):
                resumed._resume_cursor(str(directory), 2)
            resumed = make_trainer(output, auto_resume=True)
            bad = json.loads(sidecar_text)
            bad["consumed_microbatches"] += 1
            sidecar.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "recipe, counters or data cursor"):
                resumed._resume_cursor(str(directory), 2)
            sidecar.write_text(sidecar_text)
            bad = json.loads(marker_text)
            bad["world_size"] = 32
            marker.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, "completion metadata mismatch"):
                resumed._resume_cursor(str(directory), 2)
            marker.unlink()
            with self.assertRaisesRegex(RuntimeError, "Not a complete Turbo checkpoint"):
                resumed._resume_cursor(str(directory), 2)

    def test_auto_resume_skips_newer_incomplete_group_and_rejects_pdmd_only_output(self):
        with TemporaryDirectory() as output:
            trainer = make_trainer(output)
            trainer.max_train_iters = 2
            trainer.train()
            (Path(output) / "checkpoint-000000003").mkdir()
            with self.assertRaisesRegex(RuntimeError, "requires a fresh output directory"):
                make_trainer(output)._resolve_turbo_resume()
            resumed = make_trainer(output, auto_resume=True)
            checkpoint, iteration = resumed._resolve_turbo_resume()
            self.assertEqual(iteration, 2)
            self.assertTrue(checkpoint.endswith("checkpoint-000000002"))
        with TemporaryDirectory() as output:
            (Path(output) / "checkpoint-000000001").mkdir()
            with self.assertRaisesRegex(RuntimeError, "No complete Turbo checkpoint"):
                make_trainer(output, auto_resume=True)._resolve_turbo_resume()

    def test_completed_resume_retries_failed_final_preview_without_updates(self):
        with TemporaryDirectory() as output:
            original = make_trainer(output)
            original.infer_every_iters = original.max_train_iters
            original.infer_config = {"run_at_start": False}
            original.inferencer = SimpleNamespace(set_data=Mock())
            original.dataloader_val = object()
            original.run_inference = Mock(side_effect=RuntimeError("failed final preview"))
            with self.assertRaisesRegex(RuntimeError, "failed final preview"):
                original.train()
            original.run_inference.assert_called_once_with(5)
            self.assertTrue((Path(output) / "checkpoint-000000005/turbo.complete.json").is_file())
            expected = state_snapshot(original)

            resumed = make_trainer(output, auto_resume=True)
            resumed.infer_every_iters = resumed.max_train_iters
            resumed.infer_config = {"run_at_start": False}
            resumed.inferencer = SimpleNamespace(set_data=Mock())
            resumed.dataloader_val = object()
            resumed.run_inference = Mock()
            resumed._optimizer_step = Mock(side_effect=AssertionError("completed resume must not update"))
            resumed.train()
            resumed.run_inference.assert_called_once_with(5)
            resumed._optimizer_step.assert_not_called()
            self.assertEqual(resumed.read_ids, [])
            self.assert_nested_equal(expected, state_snapshot(resumed))

    def test_checkpoint_rejects_incomplete_optimizer_group(self):
        with TemporaryDirectory() as output:
            trainer = make_trainer(output)
            with self.assertRaisesRegex(RuntimeError, "incomplete Turbo student/fake update group"):
                trainer.save_checkpoint(1, 1)
            self.assertFalse(list(Path(output).iterdir()))

    def test_checkpoint_failure_preserves_old_complete_then_limit_one_prunes_safely(self):
        with TemporaryDirectory() as output:
            trainer = make_trainer(output)
            trainer.save_total_limit = 1
            trainer.max_train_iters = 1
            trainer.train()
            old = Path(output) / "checkpoint-000000001"
            unrelated = Path(output) / "checkpoint-000000999"
            unrelated.mkdir()
            (unrelated / "trainer_state.json").write_text('{"iteration": 999}')
            # Complete the next real optimizer group before injecting its save
            # failure; the cursor validator must never permit half-groups.
            trainer._optimizer_step(trainer._iter_train_samples(), "student", "dcm", 1)

            def fail_after_dmd_marker(owner, iteration, save_total_limit):
                self.assertIsNone(save_total_limit)
                incomplete = Path(owner.output_train_dir) / f"checkpoint-{iteration:09d}"
                incomplete.mkdir(exist_ok=True)
                (incomplete / "trainer_state.json").write_text('{"iteration": 2}')
                raise OSError("injected checkpoint serialization failure")

            with patch.object(DmdTrainer, "save_checkpoint", new=fail_after_dmd_marker):
                with self.assertRaisesRegex(OSError, "injected checkpoint"):
                    trainer.save_checkpoint(2, 1)
            self.assertTrue((old / "turbo.complete.json").is_file())
            trainer.auto_resume = True
            self.assertEqual(trainer._resolve_turbo_resume()[1], 1)
            trainer.max_train_iters = 2
            trainer.train()
            self.assertFalse(old.exists())
            latest = Path(output) / "checkpoint-000000002"
            self.assertTrue((latest / "turbo.complete.json").is_file())
            complete = [path for path in Path(output).glob("checkpoint-*") if (path / "turbo.complete.json").is_file()]
            self.assertEqual(complete, [latest])
            self.assertTrue(unrelated.is_dir())
            self.assertEqual(trainer._resolve_turbo_resume()[1], 2)


if __name__ == "__main__":
    unittest.main()
