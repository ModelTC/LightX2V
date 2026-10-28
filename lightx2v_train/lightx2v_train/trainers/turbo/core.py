"""Pure tensor math for H3 Turbo consistency and distribution distillation.

The objectives follow ``turbodiffusion/rcm/models/t2v_model_distill_rcm.py``
in thu-ml/TurboDiffusion, revision e3d61360b0eeab60c5cd09ff7d6c1020cccc80b2.
The upstream LTX-2 package currently provides inference, not a training loop.

H3 predicts clean-minus-noise RF velocity, while Turbo uses TrigFlow velocity
``F = d x_theta / d theta``. Conversion here keeps the entire time dependence
differentiable. The JVP helpers use forward autodiff, never finite differences.
All losses sum non-batch dimensions, then average the batch, as upstream does.
"""

from typing import NamedTuple

import torch

UPSTREAM_REVISION = "e3d61360b0eeab60c5cd09ff7d6c1020cccc80b2"


class TrigPrediction(NamedTuple):
    x0: torch.Tensor
    F: torch.Tensor


def _work(value):
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def _time_like(time, value):
    # Turbo's source computes trig coefficients in FP64. Preserve that dtype
    # through noising/conversion; only the model boundary casts to FP32/BF16.
    time = torch.as_tensor(time, device=value.device)
    time = time.to(dtype=torch.promote_types(time.dtype, value.dtype))
    if time.ndim > value.ndim:
        raise ValueError("Time must have no more dimensions than the latent")
    return time.reshape(*time.shape, *((1,) * (value.ndim - time.ndim)))


def _sample_sum(value):
    if value.ndim < 1:
        raise ValueError("Loss tensors require a batch dimension")
    return value.reshape(value.shape[0], -1).sum(dim=1).mean()


def shift_rf_time(time, shift=1.0):
    if shift <= 0:
        raise ValueError("RF shift must be positive")
    return shift * time / (1 + (shift - 1) * time)


def rf_to_trig_time(time):
    """Map RF endpoints [0, 1] exactly to TrigFlow [0, pi/2]."""
    return torch.atan2(time, 1 - time)


def trig_to_rf_time(theta):
    return torch.sin(theta) / (torch.cos(theta) + torch.sin(theta))


def sampled_trig_time(standard_normal, p_mean=-0.8, p_std=1.6):
    """Upstream LogNormal sigma sampler expressed directly in trig time."""
    return torch.atan(torch.exp(standard_normal * p_std + p_mean))


def trig_to_rf_state(state, theta):
    state = _work(state)
    theta = _time_like(theta, state)
    return state / (torch.cos(theta) + torch.sin(theta))


def prediction_from_cleanward_velocity(state, theta, velocity):
    """Convert native H3 ``v = clean - noise`` into the Turbo (x0, F)."""
    state, velocity = _work(state), _work(velocity)
    theta = _time_like(theta, state)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    total = cosine + sine
    return TrigPrediction((state + sine * velocity) / total, ((cosine - sine) * state - velocity) / total)


def preprocess_trig_jvp(state, theta, state_tangent, theta_tangent):
    """Differentiate the RF input conversion before a model's native JVP.

    Returns RF latent, RF time, and their two tangents. The model adapter must
    propagate both tangents through a single joint video/audio model forward.
    """

    def transform(value, time):
        return trig_to_rf_state(value, time), trig_to_rf_time(time)

    primal, tangent = torch.func.jvp(transform, (state, theta), (state_tangent, theta_tangent))
    return primal[0], primal[1], tangent[0], tangent[1]


def postprocess_trig_jvp(state, theta, velocity, state_tangent, theta_tangent, velocity_tangent):
    """Differentiate output conversion after the native velocity JVP."""
    primal, tangent = torch.func.jvp(
        prediction_from_cleanward_velocity,
        (state, theta, velocity),
        (state_tangent, theta_tangent, velocity_tangent),
    )
    return TrigPrediction(*primal), TrigPrediction(*tangent)


def trig_corrupt(clean, noise, theta):
    clean, noise = _work(clean), _work(noise)
    theta = _time_like(theta, clean)
    return torch.cos(theta) * clean + torch.sin(theta) * noise


def dcm_schedule(uniform, *, total_steps=48, skip_steps=1, shift=1.0):
    """Return the upstream random adjacent teacher interval in trig time.

    ``uniform`` is an unscaled U[0, 1) draw. Both H3 modalities must use the
    same draw and total/skip settings, with their respective physical shifts.
    """
    if total_steps < 1 or not 1 <= skip_steps <= total_steps:
        raise ValueError("dCM requires 1 <= skip_steps <= total_steps")
    start = uniform * (1 - skip_steps / total_steps)
    return tuple(rf_to_trig_time(shift_rf_time(1 - (start + k / total_steps), shift)) for k in range(skip_steps + 1))


def trig_euler_step(state, velocity, theta, next_theta):
    state = _work(state)
    return state - _time_like(theta - next_theta, state) * velocity


def dcm_loss(prediction, target, *, scale=100.0):
    return scale * _sample_sum((_work(prediction) - _work(target).detach()).square())


def scm_tangent(state, teacher_velocity, theta):
    """The true joint JVP direction, shared across shifted RF modalities.

    With base RF u, shift a and theta=atan(a*u/(1-u)), the direction du=u(1-u)
    gives dtheta=cos(theta)*sin(theta). Therefore applying each modality's
    pair below simultaneously includes cross-modal derivatives on ONE path.
    """
    theta = torch.as_tensor(theta, device=state.device)
    time_tangent = torch.cos(theta) * torch.sin(theta)
    return _time_like(time_tangent, _work(state)) * _work(teacher_velocity), time_tangent


def scm_direction(state, student_velocity, teacher_velocity, velocity_jvp, theta, *, warmup_ratio=1.0, norm_offset=0.1):
    """Detached, normalized sCM target direction (upstream rearranged JVP)."""
    if not 0 <= warmup_ratio <= 1:
        raise ValueError("sCM warmup ratio must be in [0, 1]")
    if norm_offset <= 0:
        raise ValueError("sCM norm offset must be positive")
    state = _work(state).detach()
    student_velocity = _work(student_velocity).detach()
    teacher_velocity = _work(teacher_velocity).detach()
    velocity_jvp = _work(velocity_jvp).detach()
    theta = _time_like(theta, state)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    direction = -cosine * torch.sqrt((1 - warmup_ratio**2 * sine.square()).clamp_min(0)) * (student_velocity - teacher_velocity)
    direction = direction - (warmup_ratio * cosine * sine * state + velocity_jvp)
    invalid = (torch.isnan(direction) | torch.isnan(student_velocity)).reshape(direction.shape[0], -1).any(dim=1)
    direction = torch.where(_time_like(invalid, direction).bool(), 0.0, direction).double()
    norm = torch.linalg.vector_norm(direction.reshape(direction.shape[0], -1), dim=1)
    return direction / _time_like(norm + norm_offset, direction)


def scm_loss(student_velocity, direction, *, scale=100.0):
    velocity = _work(student_velocity)
    invalid = torch.isnan(velocity).reshape(velocity.shape[0], -1).any(dim=1)
    velocity = torch.where(_time_like(invalid, velocity).bool(), 0.0, velocity)
    return scale * _sample_sum((velocity.double() - velocity.detach().double() - direction.detach()).square())


def dmd_loss(generated_clean, fake_clean, teacher_clean, *, scale=1.0, normalization_floor=1e-5):
    """Ordinary upstream DMD; intentionally no PDMD orthogonal projection."""
    if normalization_floor <= 0:
        raise ValueError("DMD normalization floor must be positive")
    clean = generated_clean.double()
    fake, teacher = fake_clean.detach().double(), teacher_clean.detach().double()
    weight = (clean.detach() - teacher).abs().reshape(clean.shape[0], -1).mean(dim=1).clamp_min(normalization_floor)
    gradient = (fake - teacher) / _time_like(weight, clean)
    loss = (clean - (clean - gradient).detach()).square()
    invalid = torch.isnan(loss).reshape(loss.shape[0], -1).any(dim=1)
    loss = torch.where(_time_like(invalid, loss).bool(), 0.0, loss)
    return scale * _sample_sum(loss)


def critic_loss(generated_clean, fake_clean, theta):
    """Upstream fake score denoising objective, weighted by 1/sin(theta)^2."""
    clean, fake = _work(generated_clean).detach(), _work(fake_clean)
    sine = torch.sin(_time_like(theta, fake))
    return _sample_sum((clean - fake).square() / sine.square())
