"""YuE2's released reverse-time midpoint flow solver."""

import torch


class YuE2Scheduler:
    def __init__(self, steps=32):
        if steps < 1:
            raise ValueError("ode_steps must be positive")
        self.steps = steps
        self.clear()

    def prepare(self, noise, device, dtype):
        self.state = noise.to(device=device, dtype=dtype)
        self.step_index = 0

    @staticmethod
    def raw_time(value):
        # Keep the original CPU FP64 logit, including endpoint clamp.
        return torch.logit(torch.tensor(value, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()

    def solve(self, velocity, check_stop):
        dt = 1.0 / self.steps
        for i in range(self.steps):
            self.step_index = i
            check_stop()
            t = 1.0 - i * dt
            first = velocity(self.state, self.raw_time(t))
            midpoint = self.state - first * (dt / 2)
            check_stop()
            self.state = self.state - velocity(midpoint, self.raw_time(t - dt / 2)) * dt
        result = self.state.float().cpu()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Non-finite YuE2 latent")
        return result

    def clear(self):
        self.state = None
        self.step_index = 0
