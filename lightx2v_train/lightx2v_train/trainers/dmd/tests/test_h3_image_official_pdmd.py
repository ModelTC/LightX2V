"""H3-Image uses a physical two-step grid and the single-modality PDMD loop."""

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from lightx2v_train.trainers.dmd.official_pdmd_core import rollout_grid
from lightx2v_train.trainers.dmd.official_pdmd_training import OfficialH3PdmdTraining, validate_official_pdmd_config
from lightx2v_train.trainers.dmd.tests.test_wan_official_pdmd import trainer_fixture


def image_fixture(output="unused"):
    runner, trainer, encoded = trainer_fixture(output)
    trainer.config["model"]["name"] = "minimax_h3_image"
    trainer.config["training"]["teacher"]["guidance_scale"] = 1.0
    trainer.config["data"]["train"]["name"] = "minimax_h3_image_cache_dataset"
    trainer.config["data"]["train"]["shuffle"] = False
    trainer.dmd_config.update(num_inference_steps=2, official_rollout_sigmas=[1.0, 0.75, 0.0])
    trainer.num_inference_steps = 2
    trainer.guidance_scale = 1.0
    encode = trainer._encode_conditions
    trainer._encode_conditions = lambda sample: (encode(sample)[0], None)
    runner = OfficialH3PdmdTraining(trainer)
    runner.video_grid = rollout_grid(2, sigmas=trainer.dmd_config["official_rollout_sigmas"])
    runner.audio_grid = runner.video_grid
    return runner, trainer, encoded


class H3ImageOfficialPdmdTests(unittest.TestCase):
    def test_explicit_physical_schedule_is_not_shifted_again(self):
        grid = rollout_grid(2, shift=12.0, sigmas=[1.0, 0.75, 0.0])
        torch.testing.assert_close(grid, torch.tensor([1.0, 0.75, 0.0]), rtol=0, atol=0)
        torch.testing.assert_close(grid, rollout_grid(2, shift=3.0), rtol=0, atol=0)
        for invalid in ([1.0, 0.0], [1.0, 0.75, 0.1], [1.0, 1.0, 0.0],
                        [1.0, float("nan"), 0.0], [1.0, 1.1, 0.0], [[1.0, 0.75, 0.0]]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Explicit rollout"):
                rollout_grid(2, sigmas=invalid)

    def test_image_rejects_implicit_grid_cfg_and_multi_request_batches(self):
        _, trainer, _ = image_fixture()
        validate_official_pdmd_config(trainer.config)
        trainer.config["training"]["teacher"]["guidance_scale"] = 2.0
        with self.assertRaisesRegex(ValueError, "guidance_scale"):
            validate_official_pdmd_config(trainer.config)
        trainer.config["training"]["teacher"]["guidance_scale"] = 1.0
        trainer.config["data"]["train"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "singleton"):
            validate_official_pdmd_config(trainer.config)
        trainer.config["data"]["train"]["batch_size"] = 1
        del trainer.dmd_config["official_rollout_sigmas"]
        with self.assertRaisesRegex(ValueError, "explicit physical"):
            validate_official_pdmd_config(trainer.config)

    def test_two_step_rollouts_five_critic_updates_and_one_student_update(self):
        runner, trainer, encoded = image_fixture()
        samples = itertools.count()
        initial = trainer.student.denoiser().weights.detach().clone()
        for iteration in range(5):
            role, metrics = runner._update(samples, iteration)
            self.assertEqual(role, "fake")
            self.assertNotIn("score_sigma_audio", metrics)
        self.assertEqual(len(trainer.student.denoiser().calls), 10)
        self.assertEqual(len(runner.cached[0].states), 3)
        self.assertEqual(encoded, list(range(5)))
        torch.testing.assert_close(trainer.student.denoiser().weights, initial, rtol=0, atol=0)
        role, metrics = runner._update(samples, 5)
        self.assertEqual(role, "student")
        self.assertEqual(len(trainer.teacher.denoiser().calls), 1)
        self.assertNotIn("score_sigma_audio", metrics)
        self.assertFalse(torch.equal(trainer.student.denoiser().weights, initial))

    def test_image_cache_cursor_and_checkpoint_fingerprint(self):
        with tempfile.TemporaryDirectory() as output:
            runner, trainer, encoded = image_fixture(output)
            trainer._resolve_resume = lambda: (None, 0)
            trainer.setup = Mock()
            trainer._iter_train_samples = lambda: itertools.cycle(trainer.dataloader_train)
            trainer.student.on_iteration_start = Mock()
            trainer.student.on_iteration_end = Mock()
            trainer.log_metrics = Mock()
            trainer.train_log_every_iters = 1
            trainer.max_train_iters = 6
            trainer.save_every_iters = 6
            runner.run()
            self.assertEqual(encoded, list(range(5)))
            self.assertTrue(hasattr(trainer.dataloader_train.sampler, "configure_from_microbatch_offset"))
            torch.testing.assert_close(runner.video_grid, torch.tensor([1.0, 0.75, 0.0]))
            checkpoint = Path(output) / "checkpoint-000000006"
            restored = OfficialH3PdmdTraining(trainer)
            restored._load_cursor(checkpoint, 6)
            self.assertEqual(restored.consumed_microbatches, 5)
            trainer.dataloader_train.dataset.samples[0] = {"id": "changed-condition"}
            with self.assertRaisesRegex(RuntimeError, "recipe mismatch"):
                restored._load_cursor(checkpoint, 6)


if __name__ == "__main__":
    unittest.main()
