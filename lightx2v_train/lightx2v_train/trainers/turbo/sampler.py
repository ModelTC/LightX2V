"""The stochastic TrigFlow x0/re-noise sampler used by Turbo training.

This is deliberately separate from PDMD's deterministic RF Euler rollout.
It can also sample exported Turbo adapters through an H3 capability at inference.
"""

import torch

from .core import prediction_from_cleanward_velocity, rf_to_trig_time, shift_rf_time, trig_corrupt, trig_to_rf_state, trig_to_rf_time


def joint_like(reference, video, audio):
    return type(reference)(video, audio, reference.shape)


def predict_trig(capability, state, theta_pair, condition):
    times = torch.stack([trig_to_rf_time(theta) for theta in theta_pair]).float()
    rf_state = joint_like(state, trig_to_rf_state(state.video, theta_pair[0]), trig_to_rf_state(state.audio, theta_pair[1]))
    velocity = capability.predict_velocity_at_sigmas(rf_state, times, condition)
    return tuple(prediction_from_cleanward_velocity(getattr(state, name), theta_pair[index], getattr(velocity, name)) for index, name in enumerate(("video", "audio")))


def noise_like(reference):
    return joint_like(reference, torch.randn_like(reference.video, dtype=torch.float32), torch.randn_like(reference.audio, dtype=torch.float32))


def sample_turbo_latents(capability, noise, condition, *, num_steps=8, video_shift=12.0, audio_shift=3.0, with_grad=False):
    """Predict clean x0, then independently re-noise before each next call.

    As in the released rCM training code, only the final student call receives
    parameter gradients. Callers own RNG seeding. Both modalities follow the
    same base RF grid with their own shift, applied exactly once.
    """
    if num_steps < 1:
        raise ValueError("Turbo sampling requires at least one step")
    base = torch.linspace(1, 0, num_steps + 1, device=noise.video.device, dtype=torch.float64)
    grids = tuple(rf_to_trig_time(shift_rf_time(base, shift)) for shift in (video_shift, audio_shift))
    state = noise
    for index in range(num_steps):
        theta = tuple(grid[index] for grid in grids)
        with torch.set_grad_enabled(with_grad and index == num_steps - 1):
            prediction = predict_trig(capability, state, theta, condition)
            clean = joint_like(state, prediction[0].x0, prediction[1].x0)
        if index < num_steps - 1:
            epsilon = noise_like(clean)
            state = joint_like(clean, trig_corrupt(clean.video, epsilon.video, grids[0][index + 1]), trig_corrupt(clean.audio, epsilon.audio, grids[1][index + 1]))
    return clean
