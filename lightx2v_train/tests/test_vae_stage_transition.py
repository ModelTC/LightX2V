import unittest

import torch

from lightx2v_train.trainers.vae.distillation import VAEDistillationTrainer


class _DistillationCapability:
    def compute_loss(self, sample, context):
        self.sample = sample
        self.context = context
        return "loss-result"


class VAEStageTransitionTest(unittest.TestCase):
    def _trainer(self, offset=1400):
        trainer = object.__new__(VAEDistillationTrainer)
        trainer.running_dtype = torch.bfloat16
        trainer.current_train_iteration = 3600
        trainer.stage_iteration_offset = offset
        trainer._micro_step = 0
        trainer.adversarial = None
        trainer.schedule_fingerprint = "schedule"
        trainer.vae_distillation = _DistillationCapability()
        return trainer

    def test_stage_offset_does_not_change_checkpoint_iteration(self):
        trainer = self._trainer()

        result = trainer.compute_loss_on_sample({"sample": 1})

        self.assertEqual(result, "loss-result")
        self.assertEqual(trainer.current_train_iteration, 3600)
        self.assertEqual(trainer.vae_distillation.context.iteration, 5000)

    def test_legacy_checkpoint_can_start_an_explicit_stage_offset(self):
        trainer = self._trainer()

        trainer._load_extra_checkpoint_state({"vae_distillation_schedule": "schedule"})

    def test_offset_is_saved_and_must_match_on_later_resumes(self):
        trainer = self._trainer()
        state = trainer._extra_checkpoint_state()

        self.assertEqual(state["vae_distillation_stage_iteration_offset"], 1400)
        trainer._load_extra_checkpoint_state(state)

        state["vae_distillation_stage_iteration_offset"] = 0
        with self.assertRaisesRegex(RuntimeError, "different VAE distillation stage iteration offset"):
            trainer._load_extra_checkpoint_state(state)


if __name__ == "__main__":
    unittest.main()
