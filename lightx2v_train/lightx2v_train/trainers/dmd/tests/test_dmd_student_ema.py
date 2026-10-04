"""CPU integration checks for student EMA trainer and checkpoint hooks."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime
from lightx2v_train.trainers.dmd.student_ema import StudentWeightEMA
from lightx2v_train.trainers.dmd.tests import test_dmd_checkpoint_metadata as checkpoint_metadata


class DmdStudentEMAIntegrationTest(unittest.TestCase):
    def fixture(self, updates=2):
        owner, manager, _ = checkpoint_metadata.DmdCheckpointMetadataTest().fixture()
        with torch.no_grad():
            owner.model.weight.fill_(1)
            owner.model.bias.fill_(2)
        owner.student_ema_config = {"enabled": True, "decay": 0.5, "use_for_inference": True}
        owner.student_ema = StudentWeightEMA(owner.model, decay=0.5)
        for _ in range(updates):
            with torch.no_grad():
                for parameter in owner.model.parameters():
                    parameter.add_(2)
            owner.student_ema.update()
        return owner, manager

    def assert_ema_equal(self, actual, expected):
        self.assertEqual(actual["decay"], expected["decay"])
        self.assertEqual(actual["num_updates"], expected["num_updates"])
        self.assertEqual(actual["shadow"].keys(), expected["shadow"].keys())
        for name, value in expected["shadow"].items():
            torch.testing.assert_close(actual["shadow"][name], value, rtol=0, atol=0)

    def test_student_optimizer_updates_ema_once_after_accumulation_not_fake(self):
        owner, _ = self.fixture(updates=0)
        initial = copy.deepcopy(owner.model.state_dict())
        owner.trainable_params = list(owner.model.parameters())
        fake_parameter = torch.nn.Parameter(torch.ones(1))
        owner.fake_trainable_params = [fake_parameter]
        owner.optimizer = torch.optim.SGD(owner.trainable_params, lr=0.2)
        owner.fake_optimizer = torch.optim.SGD(owner.fake_trainable_params, lr=0.4)
        owner.ida_trick = SimpleNamespace(after_student_step=Mock())
        owner.student.after_optimizer_step = Mock()
        owner.diversity_trick = SimpleNamespace(enabled=False)
        owner.real_data_fake_trick = SimpleNamespace(enabled_for=lambda region: False)
        owner._encode_conditions = Mock(return_value=("positive", None))
        owner._latent_shape = Mock(return_value=(1, 1))
        owner.sample_initial_latents = Mock(return_value=torch.zeros(1, 1))
        owner._set_student_gradient_sync = Mock()
        owner._set_fake_gradient_sync = Mock()
        owner._sync_sequence_parallel_grads = Mock()
        owner.max_grad_norm = 100.0
        owner.forward_loss = lambda *args, stage, **kwargs: (sum(parameter.sum() for parameter in owner.model.parameters()) if stage == "student" else fake_parameter.sum())
        with patch.object(owner.student_ema, "update", wraps=owner.student_ema.update) as update:
            owner._train_one_stage(iter([{}, {}, {}]), "student", 3)
            update.assert_called_once_with()
            self.assertEqual(owner.student_ema.num_updates, 1)
            for name, parameter in owner.model.named_parameters():
                torch.testing.assert_close(parameter, initial[name] - 0.2)
                torch.testing.assert_close(owner.student_ema.shadow[name], initial[name] - 0.1)
            owner._train_one_stage(iter([{}, {}, {}]), "fake", 3)
            update.assert_called_once_with()
            self.assertEqual(owner.student_ema.num_updates, 1)
        owner.student.after_optimizer_step.assert_called_once_with("main")
        self.assertEqual(owner.ida_trick.after_student_step.call_count, 1)
        torch.testing.assert_close(fake_parameter, torch.tensor([0.6]))

    def test_inference_installs_ema_and_restores_online_on_exception(self):
        owner, _ = self.fixture()
        online = copy.deepcopy(owner.model.state_dict())
        averaged = copy.deepcopy(owner.student_ema.state_dict())

        def preview(iteration):
            self.assertEqual(iteration, 7)
            for name, parameter in owner.model.named_parameters():
                torch.testing.assert_close(parameter, averaged["shadow"][name], rtol=0, atol=0)
            raise RuntimeError("preview failed")

        with patch.object(_DmdRuntime, "run_inference", side_effect=preview) as inference:
            with self.assertRaisesRegex(RuntimeError, "preview failed"):
                owner.run_inference(7)
            inference.assert_called_once_with(7)
        for name, parameter in owner.model.named_parameters():
            torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
        self.assert_ema_equal(owner.student_ema.state_dict(), averaged)

    def test_inference_can_explicitly_keep_online_weights(self):
        owner, _ = self.fixture()
        owner.student_ema_config["use_for_inference"] = False
        online = copy.deepcopy(owner.model.state_dict())

        def preview(iteration):
            for name, parameter in owner.model.named_parameters():
                torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
            return "online preview"

        with patch.object(_DmdRuntime, "run_inference", side_effect=preview):
            self.assertEqual(owner.run_inference(7), "online preview")
        self.assertEqual(owner.student_ema.num_updates, 2)

    def test_single_checkpoint_restores_ema_and_exports_averaged_weights(self):
        owner, manager = self.fixture()
        online = copy.deepcopy(owner.model.state_dict())
        averaged = copy.deepcopy(owner.student_ema.state_dict())

        def save_weights(model, directory, *, role):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            if role == "student":
                torch.save(model.state_dict(), path / "weights.pt")

        def load_weights(model, directory, *, role):
            if role == "student":
                model.load_state_dict(torch.load(Path(directory) / "weights.pt", weights_only=True))

        owner._save_model_weights = Mock(side_effect=save_weights)
        owner._load_model_weights = Mock(side_effect=load_weights)
        with tempfile.TemporaryDirectory() as directory, patch.object(DmdCheckpointManager, "_parallel", return_value=owner.parallel):
            owner.output_train_dir = directory
            manager.save_checkpoint(7, 2)
            checkpoint = Path(directory) / "checkpoint-000000007"
            state = torch.load(checkpoint / "training_state.pt", weights_only=False)
            self.assertEqual(state["student_ema_config"], owner.student_ema_config)
            self.assert_ema_equal(state["student_ema"], averaged)
            online_export = torch.load(checkpoint / "weights.pt", weights_only=True)
            ema_export = torch.load(checkpoint / "student_ema/weights.pt", weights_only=True)
            for name, parameter in owner.model.named_parameters():
                torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
                torch.testing.assert_close(online_export[name], online[name], rtol=0, atol=0)
                torch.testing.assert_close(ema_export[name], averaged["shadow"][name], rtol=0, atol=0)
            with torch.no_grad():
                for parameter in owner.model.parameters():
                    parameter.fill_(-20)
                for shadow in owner.student_ema.shadow.values():
                    shadow.zero_()
            owner.student_ema.num_updates = 0
            manager._load_single_process_state(str(checkpoint))
        self.assert_ema_equal(owner.student_ema.state_dict(), averaged)
        for name, parameter in owner.model.named_parameters():
            torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
        owner.optimizer.load_state_dict.assert_called_once_with({"step": 7})

    def test_ema_weight_export_failure_restores_online_weights(self):
        owner, manager = self.fixture()
        online = copy.deepcopy(owner.model.state_dict())
        averaged = copy.deepcopy(owner.student_ema.state_dict())

        def save_weights(model, directory, *, role):
            if Path(directory).name == "student_ema":
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, averaged["shadow"][name], rtol=0, atol=0)
                raise RuntimeError("EMA export failed")

        owner._save_model_weights = Mock(side_effect=save_weights)
        with tempfile.TemporaryDirectory() as directory:
            owner.output_train_dir = directory
            with self.assertRaisesRegex(RuntimeError, "EMA export failed"):
                manager.save_checkpoint(7, 2)
        for name, parameter in owner.model.named_parameters():
            torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
        self.assert_ema_equal(owner.student_ema.state_dict(), averaged)

    def test_distributed_checkpoint_path_restores_metadata_count_and_shadow(self):
        owner, manager = self.fixture()
        averaged = copy.deepcopy(owner.student_ema.state_dict())
        online = copy.deepcopy(owner.model.state_dict())
        parallel = SimpleNamespace(state_module=lambda: owner.model)
        owner.parallel = parallel
        module = "lightx2v_train.trainers.dmd.checkpoint"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(DmdCheckpointManager, "_parallel", return_value=parallel),
            patch(f"{module}.get_state_dict", return_value=({}, {})),
            patch(f"{module}.set_state_dict") as set_state,
        ):
            checkpoint = Path(directory) / "checkpoint-000000007"
            checkpoint.mkdir()
            manager._save_distributed_state(str(checkpoint), 7)
            trainer_state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
            self.assertEqual(trainer_state["student_ema_config"], owner.student_ema_config)
            self.assertEqual(trainer_state["student_ema"], {"decay": 0.5, "num_updates": 2})
            with torch.no_grad():
                for shadow in owner.student_ema.shadow.values():
                    shadow.fill_(-30)
            owner.student_ema.num_updates = 0
            manager._load_distributed_state(str(checkpoint))
            self.assertEqual(set_state.call_count, 2)
        self.assert_ema_equal(owner.student_ema.state_dict(), averaged)
        for name, parameter in owner.model.named_parameters():
            torch.testing.assert_close(parameter, online[name], rtol=0, atol=0)
        owner.fake_lr_scheduler.load_state_dict.assert_called_once_with({"step": 7})

    def test_resume_rejects_missing_ema_or_changed_recipe_before_restoring(self):
        owner, manager = self.fixture()
        state = {
            "iteration": 7,
            "world_size": 1,
            "dmd_checkpoint_version": 2,
            "student_train_type": "lora",
            "fake_train_type": "full",
            **manager._trick_checkpoint_metadata(),
            "student_ema": copy.deepcopy(owner.student_ema.state_dict()),
        }
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-000000007")
        before = copy.deepcopy(owner.student_ema.state_dict())
        for field in ("student_ema", "student_ema_config", "decay", "use_for_inference", "enabled"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(state)
                if field in ("student_ema", "student_ema_config"):
                    invalid.pop(field)
                else:
                    invalid["student_ema_config"][field] = 0.9 if field == "decay" else False
                with self.assertRaisesRegex(RuntimeError, "student_ema"):
                    manager._validate_checkpoint_state(invalid, "training_state.pt", "checkpoint-000000007")
                self.assert_ema_equal(owner.student_ema.state_dict(), before)
        owner._load_model_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
