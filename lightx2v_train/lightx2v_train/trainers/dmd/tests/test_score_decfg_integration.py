"""Score-only DeCFG across real H3 role setup, dCM and official PDMD updates."""

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from lightx2v_train.model_zoo.capability_adapters.common import CommonTrainableCapability
from lightx2v_train.model_zoo.minimax_h3.capability_adapters import MiniMaxH3DistributionMatchingCapability
from lightx2v_train.model_zoo.minimax_h3.tests.test_minimax_h3_mixed_rank_lora import _owner, _transformer
from lightx2v_train.model_zoo.minimax_h3.tests.test_training_adapter import _expected_state, _preset
from lightx2v_train.model_zoo.minimax_h3.training_adapter import AdapterOptions
from lightx2v_train.model_zoo.native.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from lightx2v_train.trainers.dmd.official_pdmd_core import full_rollout, rollout_grid
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime
from lightx2v_train.trainers.dmd.tests.test_official_pdmd_training import Joint, fixture
from lightx2v_train.trainers.dmd.trainer import DmdTrainer
from lightx2v_train.trainers.turbo.tests.test_trainer import make_trainer


class ScoreDeCFGIntegrationTest(unittest.TestCase):
    def test_score_only_lora_warmstart_rejects_old_full_base_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _owner(_transformer().float())
            original = copy.deepcopy(model.transformer.state_dict())
            preset_path = root / "preset.safetensors"
            factors = _preset(model.transformer, preset_path)
            model._training_adapter_options = AdapterOptions.from_mapping({"enabled": True, "path": str(preset_path)})
            checkpoint = root / "model_state.pt"
            torch.save(_expected_state(model.transformer, factors), checkpoint)
            owner = object.__new__(DmdTrainer)
            owner.model = model
            owner.parallel = SimpleNamespace(state_module=lambda: model.transformer)
            owner.student_checkpoint_path = str(checkpoint)
            owner.student_checkpoint_strict = True
            owner.student_train_type = "lora"
            with patch.object(owner, "_load_student_checkpoint", wraps=owner._load_student_checkpoint) as load, patch.object(_DmdRuntime, "setup") as setup:
                with self.assertRaisesRegex(ValueError, "original-base learned student LoRA export"):
                    owner.setup()
            load.assert_not_called()
            setup.assert_not_called()
            for name, parameter in model.transformer.named_parameters():
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)

    def test_full_warmstart_keeps_unmodified_wan_and_explicit_student_preset_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _transformer().float()
            preset_path = root / "preset.safetensors"
            factors = _preset(baseline, preset_path)
            adapted = _expected_state(baseline, factors)
            checkpoint = root / "model_state.pt"
            torch.save(adapted, checkpoint)
            # Neither a model without DeCFG (including Wan), nor the explicit
            # legacy student-preset route, changes its existing load behavior.
            for route, options in (
                ("wan_without_preset_api", None),
                ("disabled_preset", None),
                ("explicit_student_preset", {"enabled": True, "path": str(preset_path), "roles": ["student", "fake", "teacher"]}),
            ):
                with self.subTest(route=route):
                    model = _owner(copy.deepcopy(baseline))
                    if route == "wan_without_preset_api":
                        model = SimpleNamespace(transformer=model.transformer)
                    else:
                        model._training_adapter_options = AdapterOptions.from_mapping(options)
                    owner = object.__new__(DmdTrainer)
                    owner.model = model
                    owner.parallel = SimpleNamespace(state_module=lambda: model.transformer)
                    owner.student_checkpoint_path = str(checkpoint)
                    owner.student_checkpoint_strict = True
                    owner.student_train_type = "lora"
                    events = []
                    owner._setup_fake_real_resources = Mock(side_effect=RuntimeError("stop after base setup"))
                    with (
                        patch.object(owner, "_load_student_checkpoint", wraps=owner._load_student_checkpoint) as load,
                        patch.object(_DmdRuntime, "setup", side_effect=lambda **kwargs: events.append("parallel_ready")),
                    ):
                        with self.assertRaisesRegex(RuntimeError, "stop after base setup"):
                            owner.setup()
                    load.assert_called_once_with(str(checkpoint), strict=True)
                    self.assertEqual(events, ["parallel_ready"])
                    for name, parameter in model.transformer.named_parameters():
                        torch.testing.assert_close(parameter, adapted[name], rtol=0, atol=0)

    def test_real_h3_student_base_stays_original_teacher_freezes_and_full_fake_learns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = _transformer().float()
            original = copy.deepcopy(baseline.state_dict())
            preset_path = root / "preset.safetensors"
            factors = _preset(baseline, preset_path)
            expected = _expected_state(baseline, factors)
            options = AdapterOptions.from_mapping({"enabled": True, "path": str(preset_path)})
            self.assertEqual(options.roles, ("fake", "teacher"))
            owners = {}
            for role in ("student", "teacher", "fake"):
                owner = _owner(copy.deepcopy(baseline))
                owner._training_adapter_options = options
                owner._training_adapter_role = None
                owner._training_adapter_applied = False
                owner._stream_load_pending = False
                owner.prepare_training_adapter_role(role)
                owners[role] = owner
                reference = original if role == "student" else expected
                for name, parameter in owner.transformer.named_parameters():
                    torch.testing.assert_close(parameter, reference[name], rtol=0, atol=0)
                self.assertEqual(owner._training_adapter_applied, role != "student")

            student, teacher, fake = (owners[role] for role in ("student", "teacher", "fake"))
            patterns = {"adaln_proj.linear": 16, "norm_out.linear": 16}
            CommonTrainableCapability(student).configure(
                "lora",
                {
                    "rank": 128,
                    "alpha": 128,
                    "param_dtype": "fp32",
                    "target_modules": [*MiniMaxH3DistributionMatchingCapability._DEFAULT_LORA_TARGETS, *patterns],
                    "rank_pattern": patterns,
                    "alpha_pattern": patterns,
                },
            )
            CommonTrainableCapability(fake).configure("full", {})
            self.assertTrue(all(not parameter.requires_grad for parameter in teacher.transformer.parameters()))
            self.assertFalse(teacher.transformer.training)
            self.assertTrue(all(parameter.requires_grad for parameter in fake.transformer.parameters()))
            self.assertFalse(any("lora_" in name for name, _ in fake.transformer.named_parameters()))
            for name, parameter in student.transformer.named_parameters():
                self.assertEqual(parameter.requires_grad, ".lora_" in name)
                if ".lora_" not in name:
                    torch.testing.assert_close(parameter, original[name.replace(".base_layer.", ".")], rtol=0, atol=0)

            layout = build_packed_sequence(torch.ones(4, dtype=torch.long), 2, 2, 2, 3, patch_size=(1, 1, 1))
            times, indices = build_row_timesteps(layout, 0.5, 0.25)
            video, audio = fake.transformer(
                hidden_states=torch.randn(1, 8, 2),
                audio_hidden_states=torch.randn(1, 6, 2),
                encoder_hidden_states=torch.randn(1, 4, 12),
                timestep=times,
                timestep_indices=indices,
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
            loss = video.float().square().mean() + audio.float().square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in fake.transformer.parameters()))
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in fake.transformer.parameters() if parameter.grad is not None))
            torch.optim.SGD(fake.transformer.parameters(), lr=0.02).step()
            updated = copy.deepcopy(fake.transformer.state_dict())
            self.assertTrue(any(not torch.equal(updated[name], expected[name]) for name in expected))
            fake.prepare_training_adapter_role("fake")
            for name, parameter in fake.transformer.named_parameters():
                torch.testing.assert_close(parameter, updated[name], rtol=0, atol=0)
            for name, parameter in teacher.transformer.named_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad)

            student.save_lora_weights(str(root / "student_ema"))
            metadata = json.loads((root / "student_ema/training_adapter.json").read_text())
            self.assertFalse(metadata["enabled"])
            self.assertFalse(metadata["inference_requires_preset"])
            self.assertEqual(metadata["export_semantics"], "learned_lora_only")
            self.assertEqual(metadata["score_training_adapter"]["roles"], ["fake", "teacher"])

    def test_dcm_optimizer_uses_adapted_teacher_and_original_student_without_fake_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = make_trainer(directory, accumulation=1)
            with torch.no_grad():
                trainer.teacher.weights.add_(0.1)
                trainer.fake.weights.add_(0.1)
            student_before = trainer.student.weights.detach().clone()
            teacher_before = trainer.teacher.weights.detach().clone()
            fake_before = trainer.fake.weights.detach().clone()
            fake_optimizer_before = copy.deepcopy(trainer.fake_optimizer.state_dict())
            metrics = trainer._optimizer_step(trainer._iter_train_samples(), "student", "dcm", 0)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            self.assertFalse(torch.equal(trainer.student.weights, student_before))
            torch.testing.assert_close(trainer.teacher.weights, teacher_before, rtol=0, atol=0)
            torch.testing.assert_close(trainer.fake.weights, fake_before, rtol=0, atol=0)
            self.assertEqual(trainer.fake_optimizer.state_dict(), fake_optimizer_before)
            self.assertEqual(trainer.fake_updates, 0)
            self.assertEqual(trainer.ema_updates, 1)
            self.assertEqual(trainer.student.forward_modes, [False, True])
            self.assertEqual(trainer.teacher.forward_modes, [False])
            self.assertEqual(trainer.fake.forward_modes, [])
            self.assertIsNone(trainer.teacher.weights.grad)
            self.assertIsNone(trainer.fake.weights.grad)

            # The teacher's preset affects the real dCM target. It is not a
            # metadata-only role label or an unused model initialization.
            unadapted = make_trainer(directory, accumulation=1)
            other = unadapted._optimizer_step(unadapted._iter_train_samples(), "student", "dcm", 0)
            self.assertNotEqual(metrics["loss"], other["loss"])

    def test_official_pdmd_uses_score_models_but_rollout_and_inference_need_only_student(self):
        runner, trainer, encoded, _ = fixture(accum=1)
        trainer.num_inference_steps = 8
        runner.video_grid, runner.audio_grid = rollout_grid(8, 12), rollout_grid(8, 3)
        with torch.no_grad():
            trainer.teacher.parameter.add_(0.1)
            trainer.fake.parameter.add_(0.1)
        original_student = trainer.student.parameter.detach().clone()
        adapted_teacher = trainer.teacher.parameter.detach().clone()
        initialized_fake = trainer.fake.parameter.detach().clone()
        samples = itertools.count()
        for iteration in range(5):
            role, metrics = runner._update(samples, iteration)
            self.assertEqual(role, "fake")
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))
            torch.testing.assert_close(trainer.student.parameter, original_student, rtol=0, atol=0)
        self.assertEqual(trainer.teacher.calls, [])
        self.assertFalse(torch.equal(trainer.fake.parameter, initialized_fake))
        self.assertEqual(len(trainer.student.calls), 5 * 8)
        self.assertTrue(all(not call[1] for call in trainer.student.calls))
        self.assertEqual([trajectory.condition["id"] for trajectory in runner.cached], [4])
        trained_fake = trainer.fake.parameter.detach().clone()
        role, metrics = runner._update(samples, 5)
        self.assertEqual(role, "student")
        self.assertEqual(encoded, list(range(5)))
        self.assertFalse(torch.equal(trainer.student.parameter, original_student))
        torch.testing.assert_close(trainer.fake.parameter, trained_fake, rtol=0, atol=0)
        torch.testing.assert_close(trainer.teacher.parameter, adapted_teacher, rtol=0, atol=0)
        self.assertIsNone(trainer.teacher.parameter.grad)
        self.assertEqual(sum(call[1] for call in trainer.student.calls), 1)
        self.assertEqual(sum(call[1] for call in trainer.fake.calls), 5)
        self.assertEqual(len(trainer.teacher.calls), 1)
        self.assertFalse(trainer.teacher.calls[0][1])
        self.assertEqual((trainer.lr_scheduler.last_epoch, trainer.fake_lr_scheduler.last_epoch), (1, 5))
        self.assertEqual(runner.cached, [])

        initial = Joint(torch.randn(1, 1, 4), torch.randn(1, 1, 4))
        before = full_rollout(trainer.student, initial, {"id": 99}, runner.video_grid, runner.audio_grid).states[-1]
        teacher_calls, fake_calls = len(trainer.teacher.calls), len(trainer.fake.calls)
        with torch.no_grad():
            trainer.teacher.parameter.add_(100)
            trainer.fake.parameter.sub_(100)
        after = full_rollout(trainer.student, initial, {"id": 99}, runner.video_grid, runner.audio_grid).states[-1]
        torch.testing.assert_close(before.video, after.video, rtol=0, atol=0)
        torch.testing.assert_close(before.audio, after.audio, rtol=0, atol=0)
        self.assertEqual((len(trainer.teacher.calls), len(trainer.fake.calls)), (teacher_calls, fake_calls))


if __name__ == "__main__":
    unittest.main()
