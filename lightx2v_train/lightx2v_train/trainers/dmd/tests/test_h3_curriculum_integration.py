"""Trainer integration contracts for fixed presets and sequential data."""

import copy
import unittest
from unittest.mock import Mock, patch

from lightx2v_train.trainers.dmd.checkpoint import DmdCheckpointManager
from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
from lightx2v_train.trainers.dmd.runtime import _DmdRuntime


class H3CurriculumIntegrationTests(unittest.TestCase):
    @staticmethod
    def config(model="minimax_h3_t2av", kind="fl2av"):
        return {
            "model": {"name": model, "capabilities": {"distribution_matching": {"official_pdmd": True}}},
            "training": {"method": "dmd", "dmd": {"official_pdmd": True, "update_order": "fake_first"}},
            "data": {"train": {"name": "minimax_h3_curriculum_cache_dataset", "kind": kind}},
        }

    def test_official_loop_accepts_both_matching_curricula_and_rejects_cross_task_use(self):
        validate_official_pdmd_config(self.config())
        validate_official_pdmd_config(self.config("minimax_h3_ref2av", "ref2av"))
        for model, kind in (("minimax_h3_t2av", "ref2av"), ("minimax_h3_ref2av", "fl2av")):
            with self.subTest(model=model), self.assertRaises(ValueError):
                validate_official_pdmd_config(self.config(model, kind))

    def test_frozen_preset_is_immutable_even_when_objective_transition_is_allowed(self):
        check = DmdCheckpointManager._validate_h3_checkpoint_metadata
        # Absence in historical checkpoints means disabled.
        check({}, {"minimax_h3_training_adapter": {"enabled": False}}, "checkpoint", False)
        preset = {"enabled": True, "sha256": "a", "scale": 1.0, "roles": ["student", "fake"]}
        current = {"minimax_h3_training_adapter": preset}
        check(copy.deepcopy(current), current, "checkpoint", False)
        for state in ({}, {"minimax_h3_training_adapter": {**preset, "scale": 0.5}}, {"minimax_h3_training_adapter": {**preset, "sha256": "b"}}):
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, "frozen-base adapter"):
                check(state, current, "checkpoint", True)
        with self.assertRaisesRegex(RuntimeError, "frozen-base adapter"):
            check(current, {}, "checkpoint", True)

    def test_curriculum_order_and_data_identity_cannot_silently_resume_as_another_stream(self):
        check = DmdCheckpointManager._validate_h3_checkpoint_metadata
        sampler = {"route_mode": "new_then_old", "new_fingerprint": "new", "old_fingerprint": "old", "repeat": "both"}
        metadata = {"minimax_h3_route_sampling": sampler}
        check(copy.deepcopy(metadata), metadata, "checkpoint", False)
        for changed in ({}, {"minimax_h3_route_sampling": {**sampler, "new_fingerprint": "another"}}, {"minimax_h3_route_sampling": {**sampler, "repeat": "old_only"}}):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, "curriculum or data fingerprint"):
                check(metadata, changed, "checkpoint", True)

    def test_phase_logging_tracks_consumed_data_and_does_not_repeat_within_phase(self):
        owner = _DmdRuntime.__new__(_DmdRuntime)
        owner.student = Mock()
        owner.negative_prompt = None
        owner.guidance_scale = 1.0
        with patch("lightx2v_train.trainers.dmd.runtime.is_main_process", return_value=True), patch("lightx2v_train.trainers.dmd.runtime.logger") as log:
            for source in ("new", "new", "old", "old", "new"):
                owner._encode_conditions({"meta": {"curriculum_source": [source]}})
        self.assertEqual([call.args[1] for call in log.info.call_args_list], ["new", "old", "new"])
        self.assertEqual(owner.student.encode_conditions.call_count, 5)


if __name__ == "__main__":
    unittest.main()
