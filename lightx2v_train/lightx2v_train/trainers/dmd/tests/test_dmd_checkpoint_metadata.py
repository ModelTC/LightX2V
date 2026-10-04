import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from lightx2v_train.data.minimax_h3_cache_dataset import MiniMaxH3ReferenceCostSampler
from lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability import (
    MiniMaxH3DistributionMatchingCapability,
)
from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager
from lightx2v_train.trainers.dmd.trainer import DmdTrainer


class DmdCheckpointMetadataTest(unittest.TestCase):
    def fixture(self, **options):
        model = torch.nn.Linear(1, 1)
        model.transformer_component = "transformer"
        model.device = torch.device("cpu")
        model.config = {"training": {"dmd": {"num_inference_steps": 4, "num_frames": 124}}}
        owner = DmdTrainer.__new__(DmdTrainer)
        owner.config, owner.training_config = {}, {}
        owner.student = MiniMaxH3DistributionMatchingCapability(model, options)
        owner.dataloader_train = SimpleNamespace(sampler=None)
        owner.gradient_accumulation_iters, owner.fake_update_ratio = 2, 5
        owner.student_train_type, owner.fake_train_type = "lora", "full"
        owner.model, owner.fake_model = model, object()
        owner.parallel = SimpleNamespace(is_fsdp=lambda: False)
        owner.optimizer, owner.fake_optimizer = Mock(), Mock()
        owner.lr_scheduler, owner.fake_lr_scheduler = Mock(), Mock()
        for resource in (owner.optimizer, owner.fake_optimizer, owner.lr_scheduler, owner.fake_lr_scheduler):
            resource.state_dict.return_value = {"step": 7}
        roles = {
            "student": SimpleNamespace(
                model=model,
                train_type="lora",
                optimizer=owner.optimizer,
                scheduler=owner.lr_scheduler,
                spec=SimpleNamespace(optimizer_attribute="optimizer", scheduler_attribute="lr_scheduler"),
            ),
            "fake": SimpleNamespace(
                model=owner.fake_model,
                train_type="full",
                optimizer=owner.fake_optimizer,
                scheduler=owner.fake_lr_scheduler,
                spec=SimpleNamespace(optimizer_attribute="fake_optimizer", scheduler_attribute="fake_lr_scheduler"),
            ),
        }
        owner.role_registry = SimpleNamespace(
            runtimes=lambda: roles,
            weight_directory_name=lambda role: "fake_weights" if role == "fake" else None,
        )
        owner._save_model_weights, owner._load_model_weights = Mock(), Mock()
        manager = DmdCheckpointManager(owner)
        state = {
            "iteration": 7,
            "world_size": 1,
            "dmd_checkpoint_version": 2,
            "student_train_type": "lora",
            "fake_train_type": "full",
            **manager._trick_checkpoint_metadata(),
        }
        return owner, manager, state

    @staticmethod
    def validate(manager, state):
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-000000007")

    def test_current_recipe_is_valid_and_legacy_baseline_remains_resumable(self):
        _, manager, state = self.fixture()
        self.validate(manager, state)
        state.pop("minimax_h3_distribution_matching")
        state.pop("minimax_h3_per_modality_normalization")
        self.validate(manager, state)

    def test_h3_transition_guards_do_not_change_generic_checkpoint_behavior(self):
        owner, manager, state = self.fixture()
        owner.student = SimpleNamespace(extra_checkpoint_metadata=lambda: {}, legacy_extra_checkpoint_metadata=lambda: {})
        state["student_sparse_attention"] = {"enabled": True}
        state["minimax_h3_route_sampling"] = {"route_mode": "stratified"}
        self.validate(manager, state)

    def test_legacy_pdmd_transition_requires_explicit_permission(self):
        owner, manager, state = self.fixture(projected_dmd=True)
        state.pop("minimax_h3_distribution_matching")
        with self.assertRaisesRegex(RuntimeError, "projected_dmd.*allow_distribution_matching_transition"):
            self.validate(manager, state)
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        self.validate(manager, state)

    def test_legacy_normalization_changes_are_not_silent(self):
        for options in (
            {"dmd_normalization": False},
            {"dmd_normalization_epsilon": 1e-6},
            {"dmd_reduction": "sum"},
        ):
            with self.subTest(options=options):
                owner, manager, state = self.fixture(**options)
                state.pop("minimax_h3_per_modality_normalization")
                with self.assertRaisesRegex(RuntimeError, "per_modality_normalization"):
                    self.validate(manager, state)
                owner.config["resume"] = {"allow_distribution_matching_transition": True}
                self.validate(manager, state)

    def test_saved_geometry_topology_and_loss_recipe_are_validated(self):
        _, manager, state = self.fixture()
        changed_values = (
            ("minimax_h3_target_geometry", "fallback_num_frames", 141),
            ("minimax_h3_parallel_topology", "sequence_parallel_size", 2),
            ("minimax_h3_distribution_matching", "video_flow_shift", 12.0),
            ("minimax_h3_distribution_matching", "audio_dmd_loss_weight", 0.5),
        )
        for key, field, value in changed_values:
            with self.subTest(key=key, field=field):
                saved = copy.deepcopy(state)
                saved[key][field] = value
                with self.assertRaisesRegex(RuntimeError, key):
                    self.validate(manager, saved)

    def test_transition_does_not_disable_required_role_or_version_checks(self):
        owner, manager, state = self.fixture(projected_dmd=True)
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        for key in ("dmd_checkpoint_version", "fake_train_type"):
            with self.subTest(key=key):
                missing = dict(state)
                missing.pop(key)
                with self.assertRaisesRegex(RuntimeError, "missing required state"):
                    self.validate(manager, missing)
        changed = dict(state, fake_train_type="lora")
        with self.assertRaisesRegex(RuntimeError, "fake_train_type"):
            self.validate(manager, changed)

    def test_transition_does_not_disable_base_world_size_or_iteration_checks(self):
        owner, manager, state = self.fixture()
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        for key, value in (("world_size", 2), ("iteration", 8)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(RuntimeError, key):
                    self.validate(manager, dict(state, **{key: value}))

    def test_transition_does_not_disable_existing_trick_checks(self):
        owner, manager, state = self.fixture()
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        owner.diversity_trick = SimpleNamespace(checkpoint_metadata=lambda: {"diversity_enabled": False})
        with self.assertRaisesRegex(RuntimeError, "missing required state"):
            self.validate(manager, state)
        with self.assertRaisesRegex(RuntimeError, "diversity_enabled"):
            self.validate(manager, dict(state, diversity_enabled=True))
        self.validate(manager, dict(state, diversity_enabled=False))

    def test_reference_sampler_recipe_is_part_of_checkpoint_validation(self):
        owner, manager, _ = self.fixture()
        rows = []
        for index, orientation in enumerate(("landscape", "portrait") * 2):
            rows.append(
                {
                    "type": "metadata",
                    "base_dir": ".",
                    "row": {
                        "condition_path": f"condition_{index}.pt",
                        "target_orientation": orientation,
                        "reference_image_count": 1,
                        "reference_video_count": 0,
                        "reference_audio_count": 0,
                        "packed_sequence_tokens_124": 100 + index,
                    },
                }
            )
        owner.dataloader_train.sampler = MiniMaxH3ReferenceCostSampler(
            SimpleNamespace(samples=rows),
            num_replicas=2,
            rank=0,
            seed=42,
            image_counts=(1,),
        )
        state = {
            "iteration": 7,
            "world_size": 1,
            "dmd_checkpoint_version": 2,
            "student_train_type": "lora",
            "fake_train_type": "full",
            **manager._trick_checkpoint_metadata(),
        }
        self.assertEqual(state["minimax_h3_route_sampling"]["gradient_accumulation_iters"], 2)
        self.assertEqual(state["minimax_h3_route_sampling"]["fake_update_ratio"], 5)
        self.validate(manager, state)
        owner.gradient_accumulation_iters = 1
        with self.assertRaisesRegex(RuntimeError, "minimax_h3_route_sampling"):
            self.validate(manager, state)
        owner.gradient_accumulation_iters = 2
        owner.dataloader_train.sampler.seed = 43
        with self.assertRaisesRegex(RuntimeError, "minimax_h3_route_sampling"):
            self.validate(manager, state)

    def test_saved_sampler_or_sparse_attention_cannot_be_silently_removed(self):
        for key, value in (
            ("minimax_h3_route_sampling", {"route_mode": "ref_cost_bucket"}),
            ("student_sparse_attention", {"enabled": True}),
        ):
            with self.subTest(key=key):
                owner, manager, state = self.fixture()
                state[key] = value
                with self.assertRaisesRegex(RuntimeError, key):
                    self.validate(manager, state)
                owner.config["resume"] = {"allow_distribution_matching_transition": True}
                self.validate(manager, state)

    def test_missing_cost_or_stratified_sampler_requires_deliberate_transition(self):
        for route_mode in ("ref_cost_bucket", "stratified", "homogeneous"):
            with self.subTest(route_mode=route_mode):
                owner, manager, state = self.fixture()
                owner.dataloader_train.sampler = SimpleNamespace(
                    is_minimax_h3_task_cycle_sampler=True,
                    checkpoint_metadata=lambda **kwargs: {"route_mode": route_mode},
                )
                if route_mode == "homogeneous":
                    self.validate(manager, state)
                else:
                    with self.assertRaisesRegex(RuntimeError, "minimax_h3_route_sampling"):
                        self.validate(manager, state)
                    owner.config["resume"] = {"allow_distribution_matching_transition": True}
                    self.validate(manager, state)

    def test_legacy_topology_is_dense_sp1_only_even_with_objective_transition(self):
        owner, manager, state = self.fixture()
        state.pop("minimax_h3_parallel_topology")
        self.validate(manager, state)
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        module = "lightx2v_train.model_zoo.minimax_h3.capability_adapters.minimax_h3_distribution_matching_capability"
        with patch(f"{module}.get_sequence_parallel_world_size", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "without MiniMax-H3 parallel-topology"):
                self.validate(manager, state)

    def test_saved_topology_is_not_an_objective_transition(self):
        owner, manager, state = self.fixture()
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        state["minimax_h3_parallel_topology"]["sequence_parallel_size"] = 2
        with self.assertRaisesRegex(RuntimeError, "minimax_h3_parallel_topology"):
            self.validate(manager, state)

    def test_legacy_dense_attention_requires_deliberate_sparse_transition(self):
        owner, manager, state = self.fixture(student_sparse_attention={"enabled": True})
        state.pop("student_sparse_attention")
        with self.assertRaisesRegex(RuntimeError, "student_sparse_attention"):
            self.validate(manager, state)
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        self.validate(manager, state)

    def test_legacy_fixed_geometry_requires_deliberate_transition(self):
        owner, manager, state = self.fixture(fixed_num_frames=141)
        state.pop("minimax_h3_target_geometry")
        with self.assertRaisesRegex(RuntimeError, "minimax_h3_target_geometry"):
            self.validate(manager, state)
        owner.config["resume"] = {"allow_distribution_matching_transition": True}
        self.validate(manager, state)

    def test_single_process_checkpoint_persists_and_restores_extra_training_state(self):
        owner, manager, _ = self.fixture(adaptive_video_regularization={"enabled": True})
        owner.student.adv_regularizer.regression_ema[0] = 2.5
        owner.student.adv_regularizer.regression_updates[0] = 4
        with tempfile.TemporaryDirectory() as directory, patch.object(DmdCheckpointManager, "_parallel", return_value=owner.parallel):
            owner.output_train_dir = directory
            manager.save_checkpoint(7, 2)
            checkpoint = Path(directory) / "checkpoint-000000007"
            state = torch.load(checkpoint / "training_state.pt", weights_only=False)
            self.assertEqual(state["adaptive_video_regularization"]["regression_ema"][0], 2.5)
            owner.student.adv_regularizer.regression_ema[0] = None
            owner.student.adv_regularizer.regression_updates[0] = 0
            manager._load_single_process_state(str(checkpoint))
        self.assertEqual(owner.student.adv_regularizer.regression_ema[0], 2.5)
        self.assertEqual(owner.student.adv_regularizer.regression_updates[0], 4)
        owner.optimizer.load_state_dict.assert_called_once_with({"step": 7})
        owner.fake_lr_scheduler.load_state_dict.assert_called_once_with({"step": 7})

    def test_distributed_checkpoint_persists_and_restores_extra_training_state(self):
        owner, manager, _ = self.fixture(adaptive_video_regularization={"enabled": True})
        owner.student.adv_regularizer.regression_ema[2] = 1.25
        owner.student.adv_regularizer.regression_updates[2] = 9
        parallel = SimpleNamespace(state_module=lambda: torch.nn.Linear(1, 1))
        owner.parallel = parallel
        module = "lightx2v_train.trainers.dmd.checkpoint"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(DmdCheckpointManager, "_parallel", return_value=parallel),
            patch(f"{module}.get_state_dict", return_value=({}, {})),
            patch(f"{module}.set_state_dict") as set_state,
            patch(f"{module}.dcp.save") as save,
            patch(f"{module}.dcp.load") as load,
        ):
            checkpoint = Path(directory) / "checkpoint-000000007"
            checkpoint.mkdir()
            manager._save_distributed_state(str(checkpoint), 7)
            owner.student.adv_regularizer.regression_ema[2] = None
            owner.student.adv_regularizer.regression_updates[2] = 0
            manager._load_distributed_state(str(checkpoint))
        self.assertEqual(owner.student.adv_regularizer.regression_ema[2], 1.25)
        self.assertEqual(owner.student.adv_regularizer.regression_updates[2], 9)
        save.assert_called_once()
        load.assert_called_once()
        self.assertEqual(set_state.call_count, 2)


if __name__ == "__main__":
    unittest.main()
