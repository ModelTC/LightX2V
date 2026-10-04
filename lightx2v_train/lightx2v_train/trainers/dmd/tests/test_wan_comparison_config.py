import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lightx2v_train.runtime.config import load_config
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.trainers.dmd.score_sampling import ScoreSigmaContext, build_score_sigma_sampler

CONFIG = Path(__file__).resolve().parents[4] / "configs/train/dmd/wan2_1_t2v_1_3b_pdmd_comparison_fsdp2.yaml"


class WanComparisonConfigTest(unittest.TestCase):
    def config(self, projected="false"):
        with patch.dict(
            os.environ,
            {
                "WAN_DMD_MODEL": "/path/to/model",
                "WAN_DMD_PROMPTS": "/path/to/prompts.txt",
                "WAN_DMD_OUTPUT": "/path/to/output",
                "WAN_DMD_PROJECTED": projected,
            },
        ):
            return load_config(str(CONFIG))

    def test_only_projection_differs_between_experiments(self):
        baseline, projected = self.config(), self.config("true")
        baseline_options = baseline["model"]["capabilities"]["distribution_matching"]
        projected_options = projected["model"]["capabilities"]["distribution_matching"]
        self.assertIs(baseline_options.pop("projected_dmd"), False)
        self.assertIs(projected_options.pop("projected_dmd"), True)
        self.assertEqual(baseline, projected)
        self.assertEqual(baseline["distributed"]["fsdp2"]["size"], 2)
        self.assertEqual(baseline["training"]["student"]["lora"]["rank"], 128)
        self.assertEqual(baseline["training"]["student"]["lora"]["alpha"], 8)
        self.assertEqual(baseline["inference"]["infer_every_iters"], 100)
        self.assertEqual(baseline["data"]["val"]["max_samples"], 4)
        self.assertEqual(baseline["seed"], 42)

    def test_wan_score_sigma_matches_h3_video_shift_quantization_and_clamp(self):
        config = self.config()
        scheduler = DMDFlowMatchingScheduler(config)
        context = ScoreSigmaContext(None, None, 1000, torch.device("cpu"), scheduler)
        sampler = build_score_sigma_sampler(config["training"]["dmd"]["score_sampling"], use_rollout_min=False, use_rollout_max=False)
        for value in (0.0, 0.0001, 0.2371, 0.9999):
            with self.subTest(value=value), patch("torch.rand", return_value=torch.tensor([value])):
                actual = sampler.sample(context)
                quantized = torch.ceil(torch.tensor([value]) * 1000) / 1000
                expected = (6 * quantized / (1 + 5 * quantized)).clamp(0.02, 1.0)
                torch.testing.assert_close(actual, expected)

    def test_rollout_and_preview_schedules_are_identical(self):
        scheduler = DMDFlowMatchingScheduler(self.config())
        scheduler.set_timesteps(8, device="cpu")
        base = torch.linspace(1, 0, 9)
        expected = 6 * base / (1 + 5 * base)
        torch.testing.assert_close(scheduler.sigmas, expected)


if __name__ == "__main__":
    unittest.main()
