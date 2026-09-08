import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lightx2v_train.trainers.vae.adversarial import VAEAdversarialObjective
from lightx2v_train.runtime.checkpoint import find_latest_checkpoint


class VAEAdversarialTrainerStateTest(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "seed": 7,
            "crop_size": 8,
            "max_frames": 2,
            "discriminator_warmup_iters": 10,
            "discriminator": {
                "condition_channels": 2,
                "base_channels": 2,
                "channel_multipliers": (1,),
                "temporal_strides": (1,),
                "group_norm_groups": 1,
                "spectral_normalization": False,
            },
            "optimizer": {
                "learning_rate": 1e-3,
                "adam_beta1": 0.5,
                "adam_beta2": 0.9,
                "weight_decay": 0.0,
            },
        }

    @staticmethod
    def _sample(seed):
        generator = torch.Generator().manual_seed(seed)
        return (
            torch.rand(1, 3, 2, 8, 8, generator=generator, requires_grad=True),
            torch.rand(1, 3, 2, 8, 8, generator=generator),
            torch.rand(1, 4, 1, 1, 1, generator=generator),
        )

    def test_gradient_accumulation_and_state_dict_round_trip(self):
        accumulated = VAEAdversarialObjective(
            self._config(),
            device=torch.device("cpu"),
            latent_channels=4,
            gradient_accumulation_iters=2,
        )
        batched = VAEAdversarialObjective(
            self._config(),
            device=torch.device("cpu"),
            latent_channels=4,
            gradient_accumulation_iters=1,
        )

        samples = [self._sample(31), self._sample(37)]
        initial = [parameter.detach().clone() for parameter in accumulated.module.parameters()]
        for fake, real, latent in samples:
            generator_loss, _ = accumulated(fake, real, latent, stage_step=0)
            (generator_loss / 2).backward()
        self.assertEqual(accumulated.discriminator_steps, 0)
        accumulated.step()
        self.assertEqual(accumulated.discriminator_steps, 1)

        fake = torch.cat([sample[0].detach() for sample in samples]).requires_grad_(True)
        real = torch.cat([sample[1] for sample in samples])
        latent = torch.cat([sample[2] for sample in samples])
        generator_loss, _ = batched(fake, real, latent, stage_step=0)
        generator_loss.backward()
        batched.step()

        for accumulated_parameter, batched_parameter in zip(
            accumulated.module.parameters(),
            batched.module.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(accumulated_parameter, batched_parameter)
        self.assertTrue(
            any(
                not torch.equal(before, after)
                for before, after in zip(initial, accumulated.module.parameters(), strict=True)
            )
        )

        saved = copy.deepcopy(accumulated.state_dict())
        restored = VAEAdversarialObjective(
            self._config(),
            device=torch.device("cpu"),
            latent_channels=4,
            gradient_accumulation_iters=2,
        )
        restored.load_state_dict(saved)

        self.assertEqual(restored.discriminator_steps, accumulated.discriminator_steps)
        for expected, actual in zip(
            accumulated.module.parameters(),
            restored.module.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(restored.optimizer.state_dict()["param_groups"], saved["optimizer"]["param_groups"])
        for parameter, restored_parameter in zip(
            accumulated.optimizer.state.values(),
            restored.optimizer.state.values(),
            strict=True,
        ):
            self.assertEqual(parameter.keys(), restored_parameter.keys())
            for name in parameter:
                torch.testing.assert_close(restored_parameter[name], parameter[name])

    def test_auto_resume_ignores_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            complete = Path(directory) / "checkpoint-000000010"
            complete.mkdir()
            (complete / "training_state.pt").touch()
            incomplete = Path(directory) / "checkpoint-000000020"
            incomplete.mkdir()
            (incomplete / "trainer_state.pt").touch()
            (incomplete / ".incomplete").touch()

            checkpoint, iteration = find_latest_checkpoint(directory)

        self.assertEqual(checkpoint, str(complete))
        self.assertEqual(iteration, 10)

    def test_weighted_crop_specs_select_the_enabled_profile(self):
        config = self._config()
        config.pop("crop_size")
        config.pop("max_frames")
        config["crop_specs"] = [
            {"crop_size": 8, "max_frames": 1, "weight": 0.0},
            {"crop_size": 16, "max_frames": 2, "weight": 1.0},
        ]
        objective = VAEAdversarialObjective(
            config,
            device=torch.device("cpu"),
            latent_channels=4,
            gradient_accumulation_iters=1,
        )

        self.assertEqual(objective._sample_crop_spec(torch.device("cpu")), (16, 2))


if __name__ == "__main__":
    unittest.main()
