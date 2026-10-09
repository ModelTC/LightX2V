"""H3 PDMD trajectory semantics, adapted to LightX2V's clean-ward velocity.

Reference: ZeamoxWang/pdmd, commit 6b6e10635495a2d9b989fee393235d406d3840c4,
src/pdmd/{engine,schedule}.py. This module does not import the external backend.
Unlike that backend's v=noise-clean, our capability returns v=clean-noise.
Every sigma passed to the capability is an explicit physical (video, audio)
pair: training must not accidentally apply the inference shifts a second time.
"""

import hashlib
from dataclasses import dataclass

import torch


def role_at(iteration, critic_steps=5):
    if iteration < 0 or critic_steps < 1:
        raise ValueError("iteration must be non-negative and critic_steps positive")
    return "student" if iteration % (critic_steps + 1) == critic_steps else "fake"


def critic_updates_before(iteration, critic_steps=5):
    role_at(iteration, critic_steps)
    return iteration - iteration // (critic_steps + 1)


def stream_seed(*parts):
    """Stable named streams, matching the reference's seed convention."""
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def rollout_grid(steps, shift=1.0, device="cpu"):
    if steps < 1 or shift <= 0:
        raise ValueError("rollout steps and shift must be positive")
    base = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float64)
    return (shift * base / (1 + (shift - 1) * base)).float()


def critic_sigmas(generator, *, video_threshold=0.95, audio_floor=0.85, device="cpu"):
    if not 0 <= video_threshold <= 1 or not 0 <= audio_floor <= 1:
        raise ValueError("critic sigma threshold/floor must be in [0, 1]")
    # Draw on CPU as in the released training loop; the generator owns the RNG.
    draws = torch.rand(3, generator=generator, dtype=torch.float64)
    audio = torch.where(draws[0] > video_threshold, audio_floor + (1 - audio_floor) * draws[2], draws[1])
    return torch.stack((draws[0], audio)).to(device=device, dtype=torch.float32)


def interval_sigmas(video_grid, audio_grid, index, ratio):
    if not 0 <= index < len(video_grid) - 1 or len(video_grid) != len(audio_grid):
        raise ValueError("Invalid rollout interval")
    if not 0 <= ratio <= 1:
        raise ValueError("query ratio must be in [0, 1]")
    return torch.stack(tuple(grid[index + 1] + ratio * (grid[index] - grid[index + 1]) for grid in (video_grid, audio_grid)))


def joint_map(value, operation):
    return type(value)(operation(value.video, 0), operation(value.audio, 1), value.shape)


def joint_noise_like(value, generator):
    return joint_map(value, lambda tensor, _: torch.randn(tensor.shape, device=tensor.device, dtype=torch.float32, generator=generator))


@dataclass
class H3PdmdTrajectory:
    states: list
    condition: dict
    video_grid: torch.Tensor
    audio_grid: torch.Tensor


@torch.no_grad()
def full_rollout(student, noise, condition, video_grid, audio_grid):
    """All N Euler steps, not the old DMD random early-exit endpoint."""
    student.set_training(False)
    states = [joint_map(noise, lambda tensor, _: tensor.detach().float())]
    for index in range(len(video_grid) - 1):
        sigmas = torch.stack((video_grid[index], audio_grid[index]))
        next_sigmas = torch.stack((video_grid[index + 1], audio_grid[index + 1]))
        velocity = student.predict_velocity(states[-1], sigmas, condition)
        state = states[-1]
        # Clean-ward velocity reverses the sign of the reference's Euler step.
        states.append(
            type(state)(
                state.video.float() + (sigmas[0] - next_sigmas[0]) * velocity.video.float(),
                state.audio.float() + (sigmas[1] - next_sigmas[1]) * velocity.audio.float(),
                state.shape,
            )
        )
    return H3PdmdTrajectory(states, condition, video_grid, audio_grid)


def critic_objective(student, fake, trajectory, sigmas, noise):
    """Fresh Gaussian re-noising of the FULL student rollout endpoint."""
    clean = trajectory.states[-1]
    query = student.add_noise(None, clean, noise, sigmas)
    fake.set_training(False)  # eval mode does not disable gradients
    velocity = fake.predict_velocity(query, sigmas, trajectory.condition)
    target = student.training_target(clean, noise)
    return student.regression_loss(velocity, target)


def student_objective(student, fake, teacher, trajectory, index, ratio):
    """One grad-enabled student call, with a query on its predicted trajectory."""
    start = trajectory.states[index]
    sigmas = torch.stack((trajectory.video_grid[index], trajectory.audio_grid[index]))
    student.set_training(False)
    velocity = student.predict_velocity(start, sigmas, trajectory.condition)
    clean = student.x0_from_velocity(start, velocity, sigmas)
    query_sigmas = interval_sigmas(trajectory.video_grid, trajectory.audio_grid, index, ratio)
    with torch.no_grad():
        noise_hat = type(start)(
            start.video.float() - (1 - sigmas[0]) * velocity.video.float(),
            start.audio.float() - (1 - sigmas[1]) * velocity.audio.float(),
            start.shape,
        )
        query = student.add_noise(None, student.detach(clean), noise_hat, query_sigmas)
        fake.set_training(False)
        teacher.set_training(False)
        fake_velocity = fake.predict_velocity(query, query_sigmas, trajectory.condition)
        teacher_velocity = teacher.predict_velocity(query, query_sigmas, trajectory.condition)
        fake_clean = student.x0_from_velocity(query, fake_velocity, query_sigmas)
        teacher_clean = student.x0_from_velocity(query, teacher_velocity, query_sigmas)
    loss = student.dmd_loss(clean, fake_clean, teacher_clean)
    return loss, {
        **student.dmd_metrics(),
        "student_exit_index": float(index),
        "score_sigma_video": query_sigmas[0].detach(),
        "score_sigma_audio": query_sigmas[1].detach(),
    }
