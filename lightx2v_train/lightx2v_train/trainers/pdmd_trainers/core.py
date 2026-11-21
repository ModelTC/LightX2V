"""Projected DMD rollouts, following ZeamoxWang/pdmd at 6b6e106."""

from dataclasses import dataclass, fields, is_dataclass

import torch

from .utils import map_latents, pdmd_critic_loss, pdmd_loss_with_stats


def is_student_update(iteration, fake_steps):
    return iteration % (fake_steps + 1) == fake_steps


def fake_updates_before(iteration, fake_steps):
    return iteration - iteration // (fake_steps + 1)


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(**{field.name: move_tree(getattr(value, field.name), device) for field in fields(value)})
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    return value


def rollout_grid(steps, video_shift, audio_shift, *, device):
    base = torch.linspace(1, 0, steps + 1, device=device, dtype=torch.float64)[:, None]
    shifts = base.new_tensor((video_shift, audio_shift))
    return (shifts * base / (1 + (shifts - 1) * base)).float()


def critic_sigmas(generator, *, video_threshold, audio_floor, device, joint):
    draws = torch.rand(3 if joint else 1, generator=generator, dtype=torch.float64)
    if joint:
        audio = torch.where(draws[0] > video_threshold, audio_floor + (1 - audio_floor) * draws[2], draws[1])
        draws = torch.stack((draws[0], audio))
    return draws.to(device=device, dtype=torch.float32)


@dataclass
class Trajectory:
    states: list
    conditions: tuple
    sigmas: torch.Tensor


@torch.no_grad()
def full_rollout(student, noise, conditions, sigmas):
    student.set_training(False)
    states = [map_latents(noise, lambda value: value.detach().float())]
    # H3 AV velocities point toward clean, while tensor-only flow uses noise minus clean.
    tensor_only = torch.is_tensor(noise)
    if tensor_only:
        sigmas = sigmas[:, :1]
    for start, end in zip(sigmas[:-1], sigmas[1:]):
        state = states[-1]
        velocity = student.predict_velocity(state, start, conditions[0])
        if tensor_only:
            state = state + (end[0] - start[0]) * velocity.float()
        else:
            state = type(state)(
                state.video + (start[0] - end[0]) * velocity.video.float(),
                state.audio + (start[1] - end[1]) * velocity.audio.float(),
                state.shape,
            )
        states.append(state)
    return Trajectory(states, conditions, sigmas)


def critic_objective(trainer, trajectory, sigmas, noise):
    clean = trajectory.states[-1]
    query = trainer.student.add_noise(trainer.scheduler, clean, noise, sigmas)
    trainer.fake.set_training(False)
    velocity = trainer.fake.predict_velocity(query, sigmas, trajectory.conditions[0])
    target = trainer.student.training_target(clean, noise)
    if torch.is_tensor(clean):
        return trainer.student.regression_loss(velocity, target)
    video_loss = pdmd_critic_loss(velocity.video, target.video)
    audio_loss = pdmd_critic_loss(velocity.audio, target.audio)
    return trainer.student.video_weight * video_loss + trainer.student.audio_weight * audio_loss


def student_prediction(student, trajectory, index):
    start = trajectory.states[index]
    sigmas = trajectory.sigmas[index]
    student.set_training(False)
    velocity = student.predict_velocity(start, sigmas, trajectory.conditions[0])
    return velocity, student.x0_from_velocity(start, velocity, sigmas)


def student_objective(trainer, trajectory, index, ratio, *, recompute=False):
    student = trainer.student
    with torch.set_grad_enabled(not recompute):
        velocity, clean = student_prediction(student, trajectory, index)
    if recompute:
        clean = map_latents(clean, lambda value: value.detach().requires_grad_(True))

    start = trajectory.states[index]
    sigmas, next_sigmas = trajectory.sigmas[index : index + 2]
    query_sigmas = next_sigmas + ratio * (sigmas - next_sigmas)
    condition, negative = trajectory.conditions
    with torch.no_grad():
        if torch.is_tensor(start):
            noise = start.float() + (1 - sigmas[0]) * velocity.float()
        else:
            noise = type(start)(
                start.video.float() - (1 - sigmas[0]) * velocity.video.float(),
                start.audio.float() - (1 - sigmas[1]) * velocity.audio.float(),
                start.shape,
            )
        query = student.add_noise(trainer.scheduler, student.detach(clean), noise, query_sigmas)
        trainer.fake.set_training(False)
        trainer.teacher.set_training(False)
        fake_velocity = trainer.fake.predict_velocity(query, query_sigmas, condition)
        teacher_velocity = trainer._predict_teacher_velocity(query, query_sigmas, condition, negative)
        fake_clean = student.x0_from_velocity(query, fake_velocity, query_sigmas)
        teacher_clean = student.x0_from_velocity(query, teacher_velocity, query_sigmas)

    metrics = {"student_exit_index": index, "score_sigma_video": query_sigmas[0]}
    if torch.is_tensor(clean):
        loss = student.dmd_loss(clean, fake_clean, teacher_clean)
    else:
        video, video_normalizer, video_rms, video_nonfinite = pdmd_loss_with_stats(clean.video, fake_clean.video, teacher_clean.video)
        audio, audio_normalizer, audio_rms, audio_nonfinite = pdmd_loss_with_stats(clean.audio, fake_clean.audio, teacher_clean.audio)
        video, audio = student.video_weight * video, student.audio_dmd_weight * audio
        loss = video + audio
        metrics.update(
            dmd_video=video.detach(),
            dmd_audio=audio.detach(),
            dmd_video_normalizer=video_normalizer,
            dmd_audio_normalizer=audio_normalizer,
            dmd_video_direction_rms=video_rms,
            dmd_audio_direction_rms=audio_rms,
            dmd_nonfinite_update=video_nonfinite + audio_nonfinite,
        )
    metrics["pdmd_loss"] = loss.detach()
    if len(query_sigmas) == 2:
        metrics["score_sigma_audio"] = query_sigmas[1]
    return loss, clean, metrics
