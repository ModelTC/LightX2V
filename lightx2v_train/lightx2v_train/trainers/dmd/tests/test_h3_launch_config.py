"""CPU checks for the H3 recipe and its shifted score-noise distribution."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from omegaconf.errors import InterpolationKeyError

from lightx2v_train.runtime.config import load_config
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.trainers.dmd.config import DmdConfig
from lightx2v_train.trainers.dmd.score_sampling import (
    ScoreSigmaContext,
    build_score_sigma_sampler,
)

TRAIN_ROOT = Path(__file__).resolve().parents[4]
CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_ref2av_dmd_lora_match124_image_audio_1to5_uniform_fsdp32_32gpu_full_fake.yaml"
T2AV_CONFIG = TRAIN_ROOT / "configs/train/dmd/minimax_h3_t2av_dmd_lora.yaml"
LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_ref2av_fsdp32_32gpu_acp.sh"
T2AV_LAUNCHER = TRAIN_ROOT / "scripts/run_minimax_h3_t2av_dmd.sh"


class H3LaunchConfigTests(unittest.TestCase):
    def config(self, projected="false", path=CONFIG):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/path/to/model",
                "H3_REF2AV_CACHE": "/path/to/cache/metadata.jsonl",
                "H3_REF2AV_DMD_OUTPUT": "/path/to/output",
                "H3_PDMD": "false" if projected is None else projected,
            },
        ):
            if projected is None:
                os.environ.pop("H3_PDMD", None)
            return load_config(str(path))

    def test_projection_selects_student_then_one_critic_in_both_configs(self):
        # Direct load_config, not a launcher override: train.py sees this cadence.
        for path in (CONFIG, T2AV_CONFIG):
            with self.subTest(path=path):
                baseline, projected = self.config(path=path), self.config("true", path)
                self.assertIs(baseline["model"]["capabilities"]["distribution_matching"].pop("projected_dmd"), False)
                self.assertIs(projected["model"]["capabilities"]["distribution_matching"].pop("projected_dmd"), True)
                self.assertEqual(baseline["training"]["dmd"]["update_order"], "student_first")
                self.assertEqual(projected["training"]["dmd"]["update_order"], "student_first")
                self.assertEqual(DmdConfig.from_mapping(baseline).fake_update_ratio, 5)
                self.assertEqual(DmdConfig.from_mapping(projected).fake_update_ratio, 1)
                baseline["training"]["dmd"].pop("fake_update_ratio")
                projected["training"]["dmd"].pop("fake_update_ratio")
                self.assertEqual(baseline, projected)

    def test_unset_projection_preserves_baseline_and_invalid_values_fail(self):
        for path in (CONFIG, T2AV_CONFIG):
            with self.subTest(path=path):
                self.assertEqual(self.config(None, path), self.config("false", path))
                with self.assertRaises(InterpolationKeyError):
                    self.config("not-a-boolean", path)

    def test_other_ref2av_recipe_settings_are_retained(self):
        baseline = self.config()
        parsed = DmdConfig.from_mapping(baseline)
        self.assertEqual(parsed.num_inference_steps, 8)
        self.assertEqual(parsed.fake_update_ratio, 5)
        self.assertEqual(parsed.student_lora["alpha"], 8)
        self.assertEqual(parsed.latent_dtype, torch.float32)
        self.assertEqual(parsed.guidance_scale, 1.0)
        self.assertEqual(parsed.student["optimizer"]["learning_rate"], 5e-5)
        self.assertEqual(parsed.fake["optimizer"]["learning_rate"], 4e-7)
        self.assertEqual(baseline["model"]["name"], "minimax_h3_ref2av")
        self.assertEqual(baseline["model"]["transformer_param_dtype"], "bf16")
        self.assertEqual(baseline["model"]["fake"]["transformer_param_dtype"], "fp32")

    def test_score_distribution_matches_shift_then_clamp(self):
        config = self.config()
        options = config["model"]["capabilities"]["distribution_matching"]
        video_shift = options["video_flow_shift"]
        audio_shift = options["audio_flow_shift"]
        self.assertEqual((video_shift, audio_shift), (12.0, 3.0))
        audio_video_ratio = audio_shift / video_shift
        self.assertEqual(audio_video_ratio, 0.25)
        scheduler = DMDFlowMatchingScheduler(config)
        context = ScoreSigmaContext(None, None, 1000, torch.device("cpu"), scheduler)
        sampler = build_score_sigma_sampler(config["training"]["dmd"]["score_sampling"], use_rollout_min=False, use_rollout_max=False)
        for unit in (0.0, 0.0001, 0.01, 0.2353, 0.9999, 1.0):
            with self.subTest(unit=unit), patch("torch.rand", return_value=torch.tensor([unit])):
                base = sampler.sample(context)
                actual_video = video_shift * base / (1 + (video_shift - 1) * base)
                quantized = torch.ceil(torch.tensor([unit]) * 1000) / 1000
                expected_video = (video_shift * quantized / (1 + (video_shift - 1) * quantized)).clamp(0.02, 1.0)
                torch.testing.assert_close(actual_video, expected_video)
                actual_audio = audio_shift * base / (1 + (audio_shift - 1) * base)
                mapped_audio = audio_video_ratio * expected_video / (1 + (audio_video_ratio - 1) * expected_video)
                torch.testing.assert_close(actual_audio, mapped_audio)

    def test_eight_step_schedule_is_unshifted_and_terminates_at_zero(self):
        config = self.config()
        scheduler = DMDFlowMatchingScheduler(config)
        scheduler.set_timesteps(config["training"]["dmd"]["num_inference_steps"], device="cpu")
        torch.testing.assert_close(scheduler.sigmas, torch.linspace(1, 0, 9))

    def test_t2av_uses_eight_steps_and_retains_modality_shifts(self):
        config = self.config(path=T2AV_CONFIG)
        parsed = DmdConfig.from_mapping(config)
        self.assertEqual(parsed.num_inference_steps, 8)
        options = config["model"]["capabilities"]["distribution_matching"]
        self.assertEqual((options["video_flow_shift"], options["audio_flow_shift"]), (6.0, 3.0))
        scheduler = DMDFlowMatchingScheduler(config)
        scheduler.set_timesteps(parsed.num_inference_steps, device="cpu")
        torch.testing.assert_close(scheduler.sigmas, torch.linspace(1, 0, 9))

    def test_invalid_sampling_settings_fail(self):
        for change in ({"video_flow_shift": 0}, {"discrete_samples": -1}, {"min_sigma": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_score_sigma_sampler({"type": "h3_shifted_uniform", **change}, use_rollout_min=False, use_rollout_max=False)

    def test_launchers_dry_run_and_validate_actual_config_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partition = root / "model/transformer_ref"
            partition.mkdir(parents=True)
            (partition / "config.json").write_text("{}")
            manifest = root / "metadata.jsonl"
            manifest.write_text("{}\n")
            environment = {
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29599",
                "H3_CODE_ROOT": str(TRAIN_ROOT.parent),
                "H3_PYTHON": sys.executable,
                "H3_MODEL_PATH": str(root / "model"),
                "H3_REF2AV_CACHE": str(manifest),
                "H3_REF2AV_DMD_OUTPUT": str(root / "outputs"),
                "H3_REF2AV_EXPECTED_ROWS": "1",
                "H3_PDMD": "true",
            }
            environment.pop("H3_CONFIG_PATH", None)
            environment.pop("KERNELS_CACHE", None)
            environment.pop("H3_KERNEL_SNAPSHOT", None)
            for launcher, config_path in ((LAUNCHER, CONFIG), (T2AV_LAUNCHER, T2AV_CONFIG)):
                for projected, ratio in (("false", 5), ("true", 1)):
                    with self.subTest(launcher=launcher, projected=projected):
                        environment["H3_PDMD"] = projected
                        result = subprocess.run(["bash", str(launcher), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
                        if launcher == LAUNCHER:
                            self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
                        self.assertIn(f"PDMD={projected}", result.stdout)
                        self.assertIn("update_order=student_first", result.stdout)
                        self.assertIn(f"fake_update_ratio={ratio}", result.stdout)
                        self.assertNotIn("Traceback", result.stderr)

                # H3_CONFIG_PATH must not silently retain the former 1:5 recipe.
                custom = self.config("true", config_path)
                custom["training"]["dmd"]["fake_update_ratio"] = 5
                custom_path = root / "custom.yaml"
                custom_path.write_text(yaml.safe_dump(custom))
                override_environment = {**environment, "H3_CONFIG_PATH": str(custom_path), "H3_PDMD": "true"}
                result = subprocess.run(["bash", str(launcher), "--dry-run"], env=override_environment, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires update_order=student_first and fake_update_ratio=1", result.stderr)

                custom["training"]["dmd"]["fake_update_ratio"] = 1
                custom["training"]["dmd"]["update_order"] = "fake_first"
                custom_path.write_text(yaml.safe_dump(custom))
                result = subprocess.run(["bash", str(launcher), "--dry-run"], env=override_environment, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires update_order=student_first and fake_update_ratio=1", result.stderr)

                custom["model"]["capabilities"]["distribution_matching"]["projected_dmd"] = False
                custom_path.write_text(yaml.safe_dump(custom))
                result = subprocess.run(["bash", str(launcher), "--dry-run"], env=override_environment, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("disagrees with the selected config", result.stderr)

                result = subprocess.run(["bash", str(launcher), "--dry-run"], env={**environment, "H3_PDMD": "yes"}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("H3_PDMD must be true or false", result.stderr)
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
