"""Turbo recipe accounting and real stochastic sampler behavior on CPU."""

import math
import unittest
from typing import NamedTuple
from unittest.mock import patch

import torch

from lightx2v_train.trainers.turbo.config import TurboConfig
from lightx2v_train.trainers.turbo.core import rf_to_trig_time, shift_rf_time, trig_to_rf_state
from lightx2v_train.trainers.turbo.sampler import sample_turbo_latents


def recipe():
    return {
        "training": {"method": "turbo", "max_train_iters": 7000},
        "model": {"name": "minimax_h3_ref2av"},
    }


class Joint(NamedTuple):
    video: torch.Tensor
    audio: torch.Tensor
    shape: object


class TinyCapability:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def predict_velocity_at_sigmas(self, state, sigma_pair, condition):
        self.calls.append((torch.is_grad_enabled(), state, sigma_pair, condition))
        return Joint(self.parameter * state.video + sigma_pair[0], self.parameter * state.audio - sigma_pair[1], state.shape)


class TurboConfigTests(unittest.TestCase):
    def test_full_target_cm_train_supports_matching_native_t2av_and_ref_tasks(self):
        for name, task in (("minimax_h3_t2av", "t2av"), ("minimax_h3_ref2av", "ref2av")):
            with self.subTest(name=name):
                config = recipe()
                config["model"]["name"] = name
                config["training"].update(max_train_iters=1500, turbo={"dcm_steps": 1500, "scm_steps": 0, "joint_steps": 0})
                config["data"] = {"train": {"name": "minimax_h3_turbo_cm_dataset", "tasks": [task], "batch_size": 1, "drop_last": False, "native_geometry": True}}
                parsed = TurboConfig.from_mapping(config)
                self.assertEqual(parsed.stage_at(1499), "dcm")
                self.assertEqual(parsed.fake_updates_before(1500), 0)
                config["data"]["train"]["tasks"] = ["ref2av" if task == "t2av" else "t2av"]
                with self.assertRaisesRegex(ValueError, "tasks="):
                    TurboConfig.from_mapping(config)

    def test_default_curriculum_and_exact_data_accounting(self):
        config = TurboConfig.from_mapping(recipe())
        for step, stage in ((0, "dcm"), (1999, "dcm"), (2000, "scm"), (2499, "scm"), (2500, "joint"), (6999, "joint")):
            self.assertEqual(config.stage_at(step), stage)
        for cursor, fake, consumed in ((0, 0, 0), (2000, 0, 2000), (2500, 0, 2500), (2501, 4, 2505), (7000, 18000, 25000)):
            self.assertEqual(config.fake_updates_before(cursor), fake)
            self.assertEqual(config.consumed_microbatches(cursor), consumed)
            self.assertEqual(config.consumed_microbatches(cursor, 2), 2 * consumed)
        self.assertEqual(config.consumed_microbatches(7000) * 32, 800000)
        self.assertEqual(config.as_dict()["num_inference_steps"], 8)
        for invalid in (-1, 7000):
            with self.assertRaises(ValueError):
                config.stage_at(invalid)
        for invalid in (-1, 7001):
            with self.assertRaises(ValueError):
                config.fake_updates_before(invalid)

    def test_strict_option_types_and_ranges(self):
        cases = (
            {"unknown_option": 1},
            {"num_inference_steps": True},
            {"num_inference_steps": 8.0},
            {"dcm_steps": -1},
            {"dcm_total_steps": 0},
            {"dcm_skip_steps": 49},
            {"fake_update_ratio": 0},
            {"video_shift": "12"},
            {"video_shift": True},
            {"video_shift": 0},
            {"audio_shift": math.inf},
            {"loss_scale": math.nan},
            {"generator_time_std": -0.5},
            {"generator_time_mean": "-0.8"},
            {"score_time_mean": True},
        )
        for options in cases:
            with self.subTest(options=options):
                config = recipe()
                config["training"]["turbo"] = options
                with self.assertRaises(ValueError):
                    TurboConfig.from_mapping(config)

    def test_incompatible_pdmd_and_double_shift_recipes_rejected(self):
        cases = []
        config = recipe()
        config["training"]["method"] = "dmd"
        cases.append(config)
        config = recipe()
        config["training"]["max_train_iters"] = 6999
        cases.append(config)
        config = recipe()
        config["training"]["dmd"] = {}
        cases.append(config)
        config = recipe()
        config["training"]["student"] = {"checkpoint_path": "/old/pdmd/adapter.safetensors"}
        cases.append(config)
        config = recipe()
        config["training"]["teacher"] = {"guidance_scale": 3}
        cases.append(config)
        config = recipe()
        config["model"]["name"] = "wan_t2v"
        cases.append(config)
        config = recipe()
        config["scheduler"] = {"time_shift_settings": {"do_time_shift": True}}
        cases.append(config)
        config = recipe()
        config["distributed"] = {"sequence_parallel": {"enabled": True, "size": 2}}
        cases.append(config)
        for option in ("official_pdmd", "projected_dmd"):
            config = recipe()
            config["model"]["capabilities"] = {"distribution_matching": {option: True}}
            cases.append(config)
        config = recipe()
        config["model"]["capabilities"] = {"distribution_matching": {"video_flow_shift": 6}}
        cases.append(config)
        config = recipe()
        config["resume"] = {"allow_distribution_matching_transition": True}
        cases.append(config)
        for option, values in (("gradient_accumulation_iters", (0, True, "2")), ("save_total_limit", (0, False, 1.5))):
            for value in values:
                config = recipe()
                config["training"][option] = value
                cases.append(config)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                TurboConfig.from_mapping(config)


class TurboSamplerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(120)

    def noise(self):
        return Joint(torch.randn(1, 3, 2), torch.randn(1, 4, 2), {"test": "joint"})

    def test_eight_calls_only_final_parameter_grad_and_shared_condition(self):
        capability = TinyCapability()
        condition, noise = object(), self.noise()
        result = sample_turbo_latents(capability, noise, condition, with_grad=True)
        self.assertEqual(len(capability.calls), 8)
        self.assertEqual([call[0] for call in capability.calls], [False] * 7 + [True])
        self.assertTrue(all(call[3] is condition for call in capability.calls))
        self.assertTrue(all(call[2].dtype == torch.float32 for call in capability.calls))
        self.assertTrue(result.video.requires_grad)
        self.assertTrue(result.audio.requires_grad)
        self.assertEqual(result.video.dtype, torch.float64)
        self.assertIs(result.shape, noise.shape)
        (result.video.sum() + result.audio.sum()).backward()
        self.assertTrue(bool(torch.isfinite(capability.parameter.grad)))
        self.assertNotEqual(float(capability.parameter.grad), 0)
        base = torch.linspace(1, 0, 9, dtype=torch.float64)
        expected = torch.stack((shift_rf_time(base, 12), shift_rf_time(base, 3)), dim=1).float()
        for index, call in enumerate(capability.calls):
            torch.testing.assert_close(call[2], expected[index])

    def test_each_intermediate_step_draws_new_video_and_audio_noise(self):
        capability, noise = TinyCapability(), self.noise()
        condition = object()
        draws = []

        def record_noise(reference):
            draw = Joint(torch.full_like(reference.video, len(draws) + 0.25), torch.full_like(reference.audio, -len(draws) - 0.75), reference.shape)
            draws.append(draw)
            return draw

        with patch("lightx2v_train.trainers.turbo.sampler.noise_like", side_effect=record_noise):
            sample_turbo_latents(capability, noise, condition, num_steps=3)
        self.assertEqual(len(draws), 2)
        for modality_index, modality in enumerate(("video", "audio")):
            shift = (12, 3)[modality_index]
            theta = rf_to_trig_time(shift_rf_time(torch.tensor(2 / 3, dtype=torch.float64), shift))
            first_state = getattr(noise, modality)
            first_sigma = capability.calls[0][2][modality_index]
            velocity = capability.parameter.detach() * first_state + (first_sigma if modality == "video" else -first_sigma)
            # At theta=pi/2, x0 = noise + native cleanward velocity.
            first_clean = first_state + velocity
            next_trig = theta.cos() * first_clean.double() + theta.sin() * getattr(draws[0], modality)
            expected_rf = trig_to_rf_state(next_trig, theta)
            torch.testing.assert_close(getattr(capability.calls[1][1], modality), expected_rf)

    def test_inference_matches_training_value_with_same_rng_and_parameters(self):
        noise, condition = self.noise(), object()
        train, inference = TinyCapability(), TinyCapability()
        torch.manual_seed(921)
        train_value = sample_turbo_latents(train, noise, condition, with_grad=True)
        torch.manual_seed(921)
        inference_value = sample_turbo_latents(inference, noise, condition, with_grad=False)
        torch.testing.assert_close(train_value.video, inference_value.video, atol=0, rtol=0)
        torch.testing.assert_close(train_value.audio, inference_value.audio, atol=0, rtol=0)
        self.assertFalse(inference_value.video.requires_grad)
        self.assertFalse(inference_value.audio.requires_grad)
        self.assertTrue(all(not call[0] for call in inference.calls))

    def test_single_step_has_one_forward_without_renoising(self):
        capability = TinyCapability()
        with patch("lightx2v_train.trainers.turbo.sampler.noise_like") as noise_fn:
            result = sample_turbo_latents(capability, self.noise(), None, num_steps=1, with_grad=True)
        self.assertEqual(len(capability.calls), 1)
        noise_fn.assert_not_called()
        self.assertTrue(result.video.requires_grad)

    def test_invalid_step_count_rejected(self):
        for steps in (0, -1):
            with self.assertRaises(ValueError):
                sample_turbo_latents(TinyCapability(), self.noise(), None, num_steps=steps)


if __name__ == "__main__":
    unittest.main()
