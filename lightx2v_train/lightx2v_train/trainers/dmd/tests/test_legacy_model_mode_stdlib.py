"""Exercise production mode/order branches without importing PyTorch/CUDA.

Only the actual method AST is loaded; mocks replace tensor/model work, not the
mode decisions being tested. Tensor numerics remain covered by the H3 trainer
integration suite in environments with PyTorch installed.
"""

import ast
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, call

DMD_ROOT = Path(__file__).resolve().parents[1]


def load_class(filename, class_name, methods=None, namespace=None):
    path = DMD_ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    if methods is not None:
        node.bases = []
        node.decorator_list = []
        node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in methods]
    module = ast.Module(body=[node], type_ignores=[])
    env = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[class_name]


class FakeAutograd:
    float32 = "fp32"

    def __init__(self):
        self.enabled = True

    @contextmanager
    def _context(self, enabled):
        previous, self.enabled = self.enabled, enabled
        try:
            yield
        finally:
            self.enabled = previous

    def no_grad(self):
        return self._context(False)

    def enable_grad(self):
        return self._context(True)


class LegacyModelModeTest(unittest.TestCase):
    def schedule(self, **options):
        config_type = load_class("config.py", "DmdScheduleConfig", namespace={"dataclass": dataclass, "Optional": Optional})
        return config_type.from_mapping({"scheduler": {}}, dmd_config=options, student_config={}, num_inference_steps=8)

    def trainer(self):
        autograd = FakeAutograd()
        trainer_type = load_class(
            "trainer.py",
            "DmdTrainer",
            methods={"run_back_simulation", "forward_loss", "_train_iteration"},
            namespace={"torch": autograd, "broadcast_sequence_parallel_value": lambda value: value},
        )
        return trainer_type(), autograd

    def test_config_defaults_and_explicit_legacy_mode(self):
        self.assertEqual(self.schedule().model_mode, "eval")
        self.assertEqual(self.schedule(model_mode="legacy_train").model_mode, "legacy_train")
        with self.assertRaisesRegex(ValueError, "model_mode"):
            self.schedule(model_mode="train")

    def test_rollout_mode_is_independent_of_exit_step_autograd(self):
        for mode in (None, "eval", "legacy_train"):
            for grad_enabled in (False, True):
                with self.subTest(mode=mode, grad_enabled=grad_enabled):
                    trainer, autograd = self.trainer()
                    if mode is not None:
                        trainer.dmd_model_mode = mode
                    trainer.latent_dtype = "fp32"
                    trainer._prepare_sampling_schedule = Mock()
                    trainer._prepare_timestep_lookup = Mock()
                    trainer.sample_end_step = Mock(return_value=2)
                    trainer._sample_synced_int = Mock(return_value=2)
                    trainer._denoised_timestep_window = Mock(return_value=(1000, 0))
                    trainer.scheduler = SimpleNamespace(num_inference_steps=8, sigma_at=lambda *args, **kwargs: "sigma")
                    trainer.student = SimpleNamespace(
                        device="cpu",
                        latent_hw=lambda shape: None,
                        set_training=Mock(),
                        step=lambda scheduler, velocity, index, sample: (sample, velocity),
                        to_dtype=lambda value, dtype: value,
                    )
                    flags = []
                    trainer._predict_velocity = lambda *args: flags.append(autograd.enabled) or "velocity"
                    output = trainer.run_back_simulation({}, None, grad_enabled, xt="initial")
                    self.assertEqual(output, ("velocity", 1000, 0))
                    trainer.student.set_training.assert_called_once_with(mode == "legacy_train")
                    self.assertEqual(flags, [False, False, grad_enabled])
                    if grad_enabled:
                        trainer.sample_end_step.assert_called_once_with()
                    else:
                        trainer._sample_synced_int.assert_called_once_with(0, 8)

    def forward_fixture(self, mode):
        trainer, _ = self.trainer()
        trainer.dmd_model_mode = mode
        trainer.latent_dtype, trainer.scheduler = "fp32", object()
        trainer.guidance_scale, trainer.cfg_norm = 1.0, "none"
        trainer.run_back_simulation = Mock(return_value=("generated", 1000, 0))
        trainer._sample_score_sigma = Mock(return_value="sigma")
        trainer._predict_velocity = Mock(return_value="velocity")
        trainer.student = SimpleNamespace(
            device="cpu",
            latent_hw=lambda shape: None,
            random_noise_like=Mock(return_value="noise"),
            add_noise=Mock(return_value="noised"),
            training_target=Mock(return_value="target"),
            regression_loss=Mock(return_value=7),
            x0_from_velocity=Mock(return_value="x0"),
            dmd_loss=Mock(return_value=SimpleNamespace(detach=lambda: "detached")),
            student_regularization=Mock(return_value=None),
            dmd_metrics=Mock(return_value={}),
        )
        trainer.fake = SimpleNamespace(set_training=Mock())
        trainer.teacher = SimpleNamespace(set_training=Mock(), predict_guided_velocity=Mock(return_value="teacher_velocity"))
        return trainer

    def test_fake_fit_switches_but_teacher_fake_scoring_remain_eval(self):
        for mode in ("eval", "legacy_train"):
            with self.subTest(mode=mode):
                trainer = self.forward_fixture(mode)
                self.assertEqual(trainer.forward_loss(None, ({}, None), "fake"), 7)
                trainer.fake.set_training.assert_called_once_with(mode == "legacy_train")
                trainer.teacher.set_training.assert_not_called()
                self.assertFalse(trainer.run_back_simulation.call_args.kwargs["grad_enabled"])
                trainer.fake.set_training.reset_mock()
                trainer.forward_loss(None, ({}, None), "student")
                trainer.fake.set_training.assert_called_once_with(False)
                trainer.teacher.set_training.assert_called_once_with(False)
                self.assertTrue(trainer.run_back_simulation.call_args.kwargs["grad_enabled"])

    def test_old_student_then_five_independent_critic_stages(self):
        trainer, _ = self.trainer()
        trainer.dmd_update_order, trainer.fake_update_ratio = "student_first", 5
        trainer._train_one_stage = Mock(side_effect=[{"loss": 10}] + [{"loss": index, "fake_real": 0} for index in range(5)])
        samples = iter(range(6))
        student, fake, fake_real = trainer._train_iteration(samples, 1, 4)
        self.assertEqual((student, fake, fake_real), ({"loss": 10}, 2, 0))
        self.assertEqual(
            trainer._train_one_stage.call_args_list,
            [
                call(samples, stage="student", grad_accum_iters=1, outer_iteration=4),
                *[call(samples, stage="fake", grad_accum_iters=1, outer_iteration=4, fake_update_index=index) for index in range(5)],
            ],
        )

    def test_checkpoint_records_and_validates_model_mode(self):
        manager_type = load_class(
            "checkpoint.py",
            "DmdCheckpointManager",
            methods={"_extra_checkpoint_metadata", "_validate_checkpoint_state", "_require_checkpoint_keys"},
            namespace={"logger": SimpleNamespace(warning=Mock())},
        )
        manager = manager_type()
        manager.student = SimpleNamespace(extra_checkpoint_metadata=lambda: {}, legacy_extra_checkpoint_metadata=lambda: {})
        manager.dataloader_train = SimpleNamespace(sampler=None)
        manager.trainer_name, manager.dmd_model_mode = "dmd", "legacy_train"
        metadata = manager._extra_checkpoint_metadata()
        self.assertEqual(metadata["dmd_model_mode"], "legacy_train")
        manager.config = {}
        manager.checkpoint_version_key, manager.checkpoint_version = "version", 2
        manager._active_role_runtimes = lambda: []
        manager._validate_checkpoint_metadata = Mock()
        manager._validate_residual_head_state = Mock()
        manager._validate_student_ema_state = Mock()
        manager._validate_h3_checkpoint_metadata = Mock()
        manager._trick_checkpoint_metadata = manager._extra_checkpoint_metadata
        state = dict(metadata, version=2)
        manager._validate_checkpoint_state(state, "state.pt", "checkpoint")
        for saved in ("eval", None):
            changed = dict(state)
            if saved is None:
                changed.pop("dmd_model_mode")
            else:
                changed["dmd_model_mode"] = saved
            with self.assertRaisesRegex(RuntimeError, "dmd_model_mode"):
                manager._validate_checkpoint_state(changed, "state.pt", "checkpoint")
        manager.dmd_model_mode = "eval"
        changed = dict(state)
        changed.pop("dmd_model_mode")
        manager._validate_checkpoint_state(changed, "state.pt", "checkpoint")


if __name__ == "__main__":
    unittest.main()
