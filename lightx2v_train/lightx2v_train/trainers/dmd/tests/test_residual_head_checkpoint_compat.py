"""Narrow compatibility for the two newly introduced head recipe defaults."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager
from lightx2v_train.trainers.dmd.residual_head import ResidualHeadConfig


class ResidualHeadCheckpointCompatibilityTest(unittest.TestCase):
    def fixture(self, **config_updates):
        metadata = {
            **ResidualHeadConfig(enabled=True, **config_updates).checkpoint_metadata(),
            "feature_dim": 4,
            "latent_channels": 2,
            "patch_size": [1, 2, 2],
        }
        owner = SimpleNamespace(
            config={"resume": {"allow_distribution_matching_transition": True}},
            trainer_name="dmd",
            residual_head=SimpleNamespace(checkpoint_metadata=lambda: metadata),
            student=SimpleNamespace(extra_checkpoint_metadata=lambda: {}, legacy_extra_checkpoint_metadata=lambda: {}),
            dataloader_train=SimpleNamespace(sampler=None),
            role_registry=SimpleNamespace(runtimes=lambda: {}),
            _validate_checkpoint_metadata=Mock(),
        )
        manager = DmdCheckpointManager(owner)
        state = {
            "dmd_checkpoint_version": manager.checkpoint_version,
            **copy.deepcopy(manager._trick_checkpoint_metadata()),
            "residual_head_state": {},
        }
        return owner, manager, state, metadata

    @staticmethod
    def validate(manager, state):
        manager._validate_checkpoint_state(state, "training_state.pt", "checkpoint-000001000")

    @staticmethod
    def make_legacy(state):
        for key in ("fit_grad_accum_steps", "gate_mode"):
            state["residual_head_config"].pop(key, None)

    def test_legacy_full_accum_one_passes_without_modifying_metadata(self):
        _, manager, state, current = self.fixture()
        self.make_legacy(state)
        original_state, original_current = copy.deepcopy(state), copy.deepcopy(current)
        manager._validate_residual_head_state(state, "training_state.pt")
        self.validate(manager, state)
        self.assertEqual(state, original_state)
        self.assertEqual(current, original_current)

    def test_legacy_cannot_become_accum_four_or_calibrated_even_with_transition(self):
        for options in ({"fit_grad_accum_steps": 4}, {"gate_mode": "calibrated"}, {"fit_grad_accum_steps": 4, "gate_mode": "calibrated"}):
            with self.subTest(options=options):
                _, manager, state, _ = self.fixture(**options)
                self.make_legacy(state)
                with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
                    manager._validate_residual_head_state(state, "training_state.pt")
                with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
                    self.validate(manager, state)

    def test_current_recipe_round_trip(self):
        for options in ({}, {"fit_grad_accum_steps": 4, "gate_mode": "calibrated"}):
            with self.subTest(options=options):
                _, manager, state, _ = self.fixture(**options)
                self.validate(manager, copy.deepcopy(state))

    def test_partial_legacy_metadata_receives_only_missing_default(self):
        for key in ("fit_grad_accum_steps", "gate_mode"):
            with self.subTest(key=key):
                _, manager, state, _ = self.fixture()
                state["residual_head_config"].pop(key)
                self.validate(manager, state)

    def test_all_other_missing_or_mismatching_fields_remain_strict(self):
        for key, value in (("fit_steps", 7), ("learning_rate", 0.2), ("feature_dim", 99), ("patch_size", [2, 2, 2]), ("gate_mode", None), ("fit_grad_accum_steps", None)):
            with self.subTest(key=key, value=value):
                _, manager, state, _ = self.fixture()
                state["residual_head_config"][key] = value
                with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
                    self.validate(manager, state)
        _, manager, state, _ = self.fixture()
        state["residual_head_config"].pop("fit_steps")
        with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
            self.validate(manager, state)
        state["residual_head_config"]["fit_steps"] = 5
        state["residual_head_config"]["unknown_recipe"] = 1
        with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
            self.validate(manager, state)

    def test_generic_comparison_normalizes_both_sides_and_remains_strict(self):
        _, manager, state, current = self.fixture()
        # Test the generic metadata comparison independently of the dedicated
        # residual-head validation. A transition flag cannot bypass either.
        with patch.object(DmdCheckpointManager, "_validate_residual_head_state"):
            self.make_legacy(state)
            self.validate(manager, state)
            state["residual_head_config"]["fit_grad_accum_steps"] = 4
            with self.assertRaisesRegex(RuntimeError, "residual_head_config"):
                self.validate(manager, state)
            state["residual_head_config"]["fit_grad_accum_steps"] = 1
            current.pop("gate_mode")
            state["residual_head_config"]["gate_mode"] = "full"
            self.validate(manager, state)


if __name__ == "__main__":
    unittest.main()
