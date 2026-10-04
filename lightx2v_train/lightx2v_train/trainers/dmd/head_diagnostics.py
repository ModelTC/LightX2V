"""Read-only, per-example FP32 diagnostics for residual-head DMD.

All reductions are over latent elements within one example, not over the
batch. Callers can pool the resulting sufficient statistics across ranks and
noise bins without mistaking pixels for independent held-out observations.
These functions do not select a training lambda or modify gate state.
"""

import math

import torch

CANDIDATE_LAMBDAS = (("0", 0.0), ("025", 0.25), ("05", 0.5), ("075", 0.75), ("1", 1.0))


def _detached_flattened(*values):
    """Validate metadata only; never synchronize a device tensor to the CPU."""
    reference = values[0]
    if not torch.is_tensor(reference) or reference.ndim < 2:
        raise ValueError("Head diagnostics require tensors with shape [B, ...].")
    if reference.shape[0] < 1 or reference.numel() < reference.shape[0]:
        raise ValueError("Head diagnostics require nonempty examples and batches.")
    for value in values:
        if not torch.is_tensor(value) or value.shape != reference.shape:
            raise ValueError("Head diagnostic tensors must have identical shapes.")
        if value.device != reference.device:
            raise ValueError("Head diagnostic tensors must share a device.")
    return tuple(value.detach().float().flatten(1) for value in values)


def _batch_lambda(applied_lambda, reference):
    value = torch.as_tensor(applied_lambda, device=reference.device, dtype=torch.float32).detach()
    if value.numel() == 1:
        return value.reshape(1).expand(reference.shape[0])
    if value.numel() == reference.shape[0]:
        return value.reshape(reference.shape[0])
    raise ValueError("applied_lambda must be scalar or contain one value per example.")


@torch.no_grad()
def residual_risk_statistics(fake_x0, generated, correction, applied_lambda):
    """Return paired risks for ``r = F - X`` and the subtraction ``F - λC``.

    Every value is a detached FP32 tensor of shape ``[B]`` on the input device.
    ``applied_lambda`` is the caller's already frozen gate choice. In
    particular, ``optimal_lambda`` is a counterfactual diagnostic, never a
    replacement for that choice. Its validity is zero when the correction has
    zero/nonfinite energy or the residual statistics are nonfinite.

    Candidate risks use ``E[r²] - 2λ E[rC] + λ² E[C²]``. Direct full/applied
    risks retain accuracy when cancellation makes that polynomial ill-
    conditioned. Tiny negative candidate values from FP32 roundoff are clipped
    to zero. Validity fields are FP32 zero/one values suitable for aggregation.
    """
    fake, target, head = _detached_flattened(fake_x0, generated, correction)
    lam = _batch_lambda(applied_lambda, fake)
    residual = fake - target
    fake_mse = residual.square().mean(dim=1)
    residual_head_dot = (residual * head).mean(dim=1)
    head_energy = head.square().mean(dim=1)
    full_corrected_mse = (residual - head).square().mean(dim=1)
    selected_lambda_mse = (residual - lam[:, None] * head).square().mean(dim=1)

    optimal_valid = (head_energy > 0) & torch.isfinite(head_energy) & torch.isfinite(residual_head_dot) & torch.isfinite(fake_mse)
    safe_energy = torch.where(optimal_valid, head_energy, torch.ones_like(head_energy))
    optimal_lambda = torch.where(
        optimal_valid,
        (residual_head_dot / safe_energy).clamp(0, 1),
        torch.zeros_like(head_energy),
    )
    result = {
        "lambda_actual": lam,
        "fake_mse": fake_mse,
        "residual_head_dot": residual_head_dot,
        "head_energy": head_energy,
        "residual_rms": fake_mse.sqrt(),
        "correction_rms": head_energy.sqrt(),
        "full_corrected_mse": full_corrected_mse,
        "selected_lambda_mse": selected_lambda_mse,
        "optimal_lambda": optimal_lambda,
        "optimal_lambda_mse": (residual - optimal_lambda[:, None] * head).square().mean(dim=1),
        "optimal_lambda_valid": optimal_valid.float(),
    }
    for suffix, candidate in CANDIDATE_LAMBDAS:
        result[f"candidate_mse_{suffix}"] = (fake_mse - 2 * candidate * residual_head_dot + candidate * candidate * head_energy).clamp_min(0)
    return result


@torch.no_grad()
def student_direction_statistics(fake_x0, teacher_x0, correction, applied_lambda):
    """Compare unnormalized DMD directions ``F-T`` and ``F-λC-T``.

    The correction norm ratio is ``||λC|| / ||F-T||``. Undefined ratios and
    cosine/angle values are zero and have an explicit zero validity mask, so
    logging stays finite for zero directions without reporting an invented
    angle. For a nonzero direction, λ=0 yields ratio=0, cosine=1, angle=0.
    Callers must use the validity masks when averaging these quantities.
    The diagnostics neither normalize nor replace the actual student loss.
    """
    fake, teacher, head = _detached_flattened(fake_x0, teacher_x0, correction)
    lam = _batch_lambda(applied_lambda, fake)
    raw = fake - teacher
    applied = lam[:, None] * head
    corrected = raw - applied
    raw_rms = raw.square().mean(dim=1).sqrt()
    corrected_rms = corrected.square().mean(dim=1).sqrt()
    applied_rms = applied.square().mean(dim=1).sqrt()

    ratio_valid = (raw_rms > 0) & torch.isfinite(raw_rms) & torch.isfinite(applied_rms)
    safe_raw = torch.where(ratio_valid, raw_rms, torch.ones_like(raw_rms))
    norm_ratio = applied_rms / safe_raw
    ratio_valid = ratio_valid & torch.isfinite(norm_ratio)
    norm_ratio = torch.where(ratio_valid, norm_ratio, torch.zeros_like(norm_ratio))

    cosine_denominator = raw_rms * corrected_rms
    direction_dot = (raw * corrected).mean(dim=1)
    cosine_valid = (raw_rms > 0) & (corrected_rms > 0) & torch.isfinite(cosine_denominator) & torch.isfinite(direction_dot) & (cosine_denominator > 0)
    safe_denominator = torch.where(cosine_valid, cosine_denominator, torch.ones_like(cosine_denominator))
    cosine = (direction_dot / safe_denominator).clamp(-1, 1)
    cosine = torch.where(cosine_valid, cosine, torch.zeros_like(cosine))
    # Identical directions must not acquire a spurious angle from the FP32
    # sqrt/product roundoff in their cosine denominator (notably when λ=0).
    identity = cosine_valid & (applied_rms == 0)
    cosine = torch.where(identity, torch.ones_like(cosine), cosine)
    angle = torch.where(cosine_valid, cosine.acos() * (180 / math.pi), torch.zeros_like(cosine))
    return {
        "direction_raw_rms": raw_rms,
        "direction_corrected_rms": corrected_rms,
        "correction_rms": head.square().mean(dim=1).sqrt(),
        "applied_correction_rms": applied_rms,
        "correction_direction_norm_ratio": norm_ratio,
        "correction_direction_norm_ratio_valid": ratio_valid.float(),
        "direction_cosine": cosine,
        "direction_angle_degrees": angle,
        "direction_cosine_valid": cosine_valid.float(),
    }
