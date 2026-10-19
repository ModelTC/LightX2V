# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License; see lightx2v/models/networks/trellis2/LICENSE.

import numpy as np
import torch


class FlowEulerSampler:
    def __init__(self, sigma_min=1e-5):
        self.sigma_min = sigma_min

    def _predict(self, model, x, t, cond, **kwargs):
        timestep = torch.full((x.shape[0],), 1000 * t, device=x.device, dtype=torch.float32)
        return model(x, timestep, cond, **kwargs)

    def _xstart(self, x, t, prediction):
        return (1 - self.sigma_min) * x - (self.sigma_min + (1 - self.sigma_min) * t) * prediction

    def sample_once(self, model, x, t, t_prev, cond, neg_cond, guidance_strength, guidance_interval, guidance_rescale=0.0, **kwargs):
        strength = guidance_strength if guidance_interval[0] <= t <= guidance_interval[1] else 1.0
        if strength == 0:
            return x - (t - t_prev) * self._predict(model, x, t, neg_cond, **kwargs)
        positive = self._predict(model, x, t, cond, **kwargs)
        if strength != 1:
            negative = self._predict(model, x, t, neg_cond, **kwargs)
            prediction = strength * positive + (1 - strength) * negative
            if guidance_rescale > 0:
                positive_x0 = self._xstart(x, t, positive)
                guided_x0 = self._xstart(x, t, prediction)
                dims = list(range(1, guided_x0.ndim))
                ratio = positive_x0.std(dim=dims, keepdim=True) / guided_x0.std(dim=dims, keepdim=True)
                rescaled = guidance_rescale * guided_x0 * ratio + (1 - guidance_rescale) * guided_x0
                prediction = ((1 - self.sigma_min) * x - rescaled) / (self.sigma_min + (1 - self.sigma_min) * t)
        else:
            prediction = positive
        return x - (t - t_prev) * prediction

    def sample(self, model, noise, cond, neg_cond, steps=12, rescale_t=1.0, **kwargs):
        times = np.linspace(1, 0, steps + 1)
        times = (rescale_t * times / (1 + (rescale_t - 1) * times)).tolist()
        sample = noise
        for current, previous in zip(times[:-1], times[1:]):
            sample = self.sample_once(model, sample, current, previous, cond, neg_cond, **kwargs)
        return sample
