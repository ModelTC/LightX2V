"""CPU checks for the H3 recipe and its shifted score-noise distribution."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

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


class H3LaunchConfigTests(unittest.TestCase):
    def config(self, projected="false"):
        with patch.dict(
            os.environ,
            {
                "H3_MODEL_PATH": "/path/to/model",
                "H3_REF2AV_CACHE": "/path/to/cache/metadata.jsonl",
                "H3_REF2AV_DMD_OUTPUT": "/path/to/output",
                "H3_PDMD": projected,
            },
        ):
            return load_config(str(CONFIG))

    def test_projection_is_a_boolean_and_only_objective_toggle(self):
        baseline, projected = self.config(), self.config("true")
        options = baseline["model"]["capabilities"]["distribution_matching"]
        self.assertIs(options["projected_dmd"], False)
        self.assertIs(projected["model"]["capabilities"]["distribution_matching"].pop("projected_dmd"), True)
        options.pop("projected_dmd")
        self.assertEqual(baseline, projected)
        parsed = DmdConfig.from_mapping(baseline)
        self.assertEqual(parsed.num_inference_steps, 8)
        self.assertEqual(parsed.fake_update_ratio, 5)
        self.assertEqual(parsed.student_lora["alpha"], 8)
        self.assertEqual(parsed.latent_dtype, torch.float32)
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
        config = load_config(str(T2AV_CONFIG))
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

    def test_launcher_dry_run_without_starting_distributed_training(self):
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
            environment.pop("KERNELS_CACHE", None)
            environment.pop("H3_KERNEL_SNAPSHOT", None)
            result = subprocess.run(["bash", str(LAUNCHER), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
            self.assertIn("--nnodes=4 --nproc_per_node=8", result.stdout)
            self.assertIn("PDMD=true", result.stdout)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
