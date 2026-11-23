"""Shift transitions must preserve PDMD updates, optimizer state and cursors."""

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from lightx2v_train.trainers.dmd.official_pdmd_core import rollout_grid
from lightx2v_train.trainers.dmd.official_pdmd_training import validate_official_pdmd_config
from lightx2v_train.trainers.dmd.rollout_schedule import rollout_shifts
from lightx2v_train.trainers.dmd.tests.test_official_pdmd_training import fixture


def scheduled_fixture(output, split=10):
    runner, trainer, _, _ = fixture(output=output)
    trainer.dmd_config.update(official_rollout_video_shift=6., official_rollout_audio_shift=3.,
                             rollout_shift_schedule=[
                                 {"start_update": 0, "video_shift": 6., "audio_shift": 3.},
                                 {"start_update": split, "video_shift": 12., "audio_shift": 3.},
                             ])
    trainer.config["model"]["capabilities"]["distribution_matching"].update(video_flow_shift=6., audio_flow_shift=3.)
    validate_official_pdmd_config(trainer.config)
    for name, parameter in (("optimizer", trainer.student.parameter), ("fake_optimizer", trainer.fake.parameter)):
        setattr(trainer, name, torch.optim.AdamW([parameter], lr=.001, betas=(0., .9)))
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)
    trainer.fake_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.fake_optimizer, lambda _: 1.)
    trainer.dataloader_train = SimpleNamespace(sampler=SimpleNamespace(configure_from_microbatch_offset=Mock()))
    trainer._resolve_resume = lambda: (None, 0)
    trainer._iter_train_samples = lambda: itertools.count(runner.consumed_microbatches)
    trainer.student.on_iteration_start = Mock()
    trainer.student.on_iteration_end = Mock()
    trainer.log_metrics = Mock()
    trainer.train_log_every_iters = 1000
    trainer.max_train_iters = 12
    trainer.save_every_iters = 0
    trainer.setup = Mock()
    save_weights = trainer.save_checkpoint

    def save(iteration, limit):
        save_weights(iteration, limit)
        state = {name: getattr(trainer, name).state_dict() for name in
                 ("optimizer", "fake_optimizer", "lr_scheduler", "fake_lr_scheduler")}
        state["video_shift"] = trainer.student.video_shift
        torch.save(state, Path(output) / f"checkpoint-{iteration:09d}" / "toy_training.pt")

    trainer.save_checkpoint = save
    return runner, trainer


class RolloutScheduleTest(unittest.TestCase):
    def test_exact_experiment_boundary_and_schedule_is_immutable(self):
        runner, trainer = scheduled_fixture("unused", split=1000)
        recipe = copy.deepcopy(runner.recipe())
        for completed, expected in ((0, 6), (999, 6), (1000, 12), (2999, 12)):
            runner._set_rollout_schedule(completed)
            self.assertEqual(trainer.student.video_shift, expected)
            torch.testing.assert_close(runner.video_grid, rollout_grid(4, expected))
            torch.testing.assert_close(runner.audio_grid, rollout_grid(4, 3))
            self.assertEqual(runner.recipe(), recipe)

    def test_resume_on_both_sides_and_before_student_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as root:
            full, expected = scheduled_fixture(str(Path(root) / "full"))
            full.run()
            for stop in (9, 10, 11):
                output = str(Path(root) / str(stop))
                first, trainer = scheduled_fixture(output)
                trainer.max_train_iters = stop
                first.run()
                path = Path(output) / f"checkpoint-{stop:09d}"
                resumed, actual = scheduled_fixture(output)
                actual._resolve_resume = lambda: (path, stop)

                def restore(resume_ckpt_path):
                    state = torch.load(resume_ckpt_path / "toy_training.pt", weights_only=True)
                    self.assertEqual(actual.student.video_shift, state.pop("video_shift"))
                    weights = torch.load(resume_ckpt_path / "toy_weights.pt", weights_only=True)
                    with torch.no_grad():
                        actual.student.parameter.copy_(weights["student"])
                        actual.fake.parameter.copy_(weights["fake"])
                    for name, value in state.items():
                        getattr(actual, name).load_state_dict(value)

                actual.setup = restore
                resumed.run()
                for role in ("student", "fake"):
                    torch.testing.assert_close(getattr(actual, role).parameter, getattr(expected, role).parameter, rtol=0, atol=0)
                self.assertEqual(resumed.consumed_microbatches, full.consumed_microbatches)
                for name in ("optimizer", "fake_optimizer"):
                    a, b = getattr(actual, name).state_dict(), getattr(expected, name).state_dict()
                    self.assertEqual(a["param_groups"], b["param_groups"])
                    for index in a["state"]:
                        for key in a["state"][index]:
                            torch.testing.assert_close(a["state"][index][key], b["state"][index][key], rtol=0, atol=0)
                receipt = json.loads((Path(output) / "checkpoint-000000012/rollout_schedule.json").read_text())
                self.assertEqual(receipt["last_update_shifts"], [12., 3.])

    def test_rejects_transition_between_fake_and_student(self):
        with self.assertRaisesRegex(ValueError, "cached fake trajectory"):
            scheduled_fixture("unused", split=1001)

    def test_resume_rejects_changed_schedule(self):
        with tempfile.TemporaryDirectory() as output:
            runner, trainer = scheduled_fixture(output)
            trainer.max_train_iters = 10
            runner.run()
            changed, _ = scheduled_fixture(output, split=12)
            with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
                changed._load_cursor(Path(output) / "checkpoint-000000010", 10)

    def test_actual_h3_checkpoint_metadata_restores_last_completed_phase(self):
        from lightx2v_train.trainers.dmd.tests.test_dmd_checkpoint_metadata import DmdCheckpointMetadataTest

        for completed in (1000, 1001, 1005, 3000):
            checks = DmdCheckpointMetadataTest()
            owner, manager, state = checks.fixture(official_pdmd=True)
            runner, trainer = scheduled_fixture("unused", split=1000)
            trainer.student = owner.student
            runner._set_rollout_schedule(completed - 1)
            state.update(manager._trick_checkpoint_metadata())
            owner.student.video_shift = 6.
            runner.active_rollout_shifts = None
            runner._set_rollout_schedule(completed - 1)
            checks.validate(manager, state)
            self.assertEqual(owner.student.video_shift, 6. if completed == 1000 else 12.)


if __name__ == "__main__":
    unittest.main()
