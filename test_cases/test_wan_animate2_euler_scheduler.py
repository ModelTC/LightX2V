import unittest

import torch

from lightx2v.models.schedulers.wan.animate2.scheduler import WanAnimate2Scheduler


class WanAnimate2EulerSchedulerTests(unittest.TestCase):
    def test_euler_uses_first_order_flow_update_for_middle_steps(self):
        scheduler = WanAnimate2Scheduler(
            {
                "infer_steps": 4,
                "sample_shift": 5.0,
                "sample_guide_scale": 1.0,
                "flow_solver": "euler",
            }
        )
        scheduler.sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0], dtype=torch.float32)
        scheduler.step_index = 2
        scheduler.lower_order_nums = 1
        scheduler.model_outputs = [torch.tensor([2.0]), torch.tensor([-100.0])]
        sample = torch.tensor([4.0], dtype=torch.float32)
        velocity = torch.tensor([0.5], dtype=torch.float32)

        result = scheduler.step(velocity, sample)

        expected = sample + (scheduler.sigmas[3] - scheduler.sigmas[2]) * velocity
        torch.testing.assert_close(result, expected)

    def test_unknown_flow_solver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Wan-Animate-2 scheduler"):
            WanAnimate2Scheduler(
                {
                    "infer_steps": 10,
                    "sample_shift": 5.0,
                    "sample_guide_scale": 1.0,
                    "flow_solver": "unknown",
                }
            )


if __name__ == "__main__":
    unittest.main()
