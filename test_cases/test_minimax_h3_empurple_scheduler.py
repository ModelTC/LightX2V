import math
import unittest
from unittest.mock import patch

import torch

from lightx2v.models.schedulers.minimax_h3 import scheduler as scheduler_module


class MiniMaxH3EmpurpleSchedulerTest(unittest.TestCase):
    def _build_scheduler(self, *, teacher_steps=40, prefix_steps=4, student_grid_points=5):
        config = {
            "infer_steps": student_grid_points,
            "video_flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "h3_step_update": "training_euler",
            "h3_empurple": {
                "enabled": True,
                "teacher_num_inference_steps": teacher_steps,
                "teacher_prefix_steps": prefix_steps,
                "teacher_video_flow_shift": 6.0,
                "teacher_audio_flow_shift": 3.0,
                "student_video_flow_shift": 6.0,
                "student_audio_flow_shift": 3.0,
                "teacher_step_update": "reference_blend",
                "student_step_update": "training_euler",
                "student_lora_scale": 1.0,
            },
        }
        with patch.object(scheduler_module, "AI_DEVICE", "cpu"):
            return scheduler_module.MiniMaxH3Scheduler(config)

    def test_prefix4_plus_full_student4_is_continuous(self):
        scheduler = self._build_scheduler()

        self.assertEqual(scheduler.infer_steps, 8)
        self.assertEqual(scheduler.step_phases, ("teacher",) * 4 + ("student",) * 4)
        self.assertEqual(scheduler.step_lora_scales, (0.0,) * 4 + (1.0,) * 4)
        self.assertEqual(scheduler.step_updates, ("reference_blend",) * 4 + ("training_euler",) * 4)
        self.assertEqual(scheduler.video_sigmas.numel(), 9)
        self.assertEqual(scheduler.audio_sigmas.numel(), 9)
        self.assertTrue(torch.all(scheduler.video_sigmas[:-1] > scheduler.video_sigmas[1:]))
        self.assertTrue(torch.all(scheduler.audio_sigmas[:-1] > scheduler.audio_sigmas[1:]))
        self.assertEqual(float(scheduler.video_sigmas[-1]), 0.0)
        self.assertEqual(float(scheduler.audio_sigmas[-1]), 0.0)

        self.assertTrue(math.isclose(scheduler.empurple_handoff_base_sigma, 0.9))
        expected_video = 6.0 * 0.9 / (1.0 + 5.0 * 0.9)
        expected_audio = 3.0 * 0.9 / (1.0 + 2.0 * 0.9)
        self.assertTrue(math.isclose(float(scheduler.video_sigmas[4]), expected_video, rel_tol=1e-6))
        self.assertTrue(math.isclose(float(scheduler.audio_sigmas[4]), expected_audio, rel_tol=1e-6))

    def test_strict_turbo_boundary_is_teacher10_then_student3(self):
        scheduler = self._build_scheduler(prefix_steps=10, student_grid_points=4)

        self.assertEqual(scheduler.infer_steps, 13)
        expected_raw_grid = torch.tensor([0.75, 0.5, 0.25, 0.0])
        expected_video = scheduler_module._shift_sigma(expected_raw_grid, 6.0)
        expected_audio = scheduler_module._shift_sigma(expected_raw_grid, 3.0)
        torch.testing.assert_close(scheduler.video_sigmas[10:], expected_video)
        torch.testing.assert_close(scheduler.audio_sigmas[10:], expected_audio)

    def test_invalid_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "teacher_prefix_steps"):
            self._build_scheduler(prefix_steps=40)

    def test_standard_four_nfe_schedule_is_unchanged(self):
        config = {
            "infer_steps": 5,
            "video_flow_shift": 6.0,
            "audio_flow_shift": 3.0,
            "h3_step_update": "training_euler",
        }
        with patch.object(scheduler_module, "AI_DEVICE", "cpu"):
            scheduler = scheduler_module.MiniMaxH3Scheduler(config)

        raw_grid = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        torch.testing.assert_close(scheduler.video_sigmas, scheduler_module._shift_sigma(raw_grid, 6.0))
        torch.testing.assert_close(scheduler.audio_sigmas, scheduler_module._shift_sigma(raw_grid, 3.0))
        self.assertEqual(scheduler.infer_steps, 4)
        self.assertEqual(scheduler.step_phases, ("student",) * 4)


if __name__ == "__main__":
    unittest.main()
