"""Full-endpoint DMD rollouts with a one-step surrogate gradient."""

import torch


def full_rollout_with_ste(
    initial_latents,
    num_steps,
    predict_velocity,
    step_by_index,
    *,
    gradient_step_idx=None,
):
    """Run all steps; optionally attach the selected step's x0 Jacobian.

    ``predict_velocity(latents, idx)`` and ``step_by_index(velocity, idx,
    latents)`` use the same active schedule. The latter returns (next_xt, x0).
    Only the selected input is retained, then recomputed with gradients after
    the no-grad rollout. With no gradient index (fake training), the returned
    endpoint is detached and no student computation graph is built.

    The two metadata tensors describe the selected step, independently of
    the full endpoint used for the DMD score query.
    """
    if num_steps < 1:
        raise ValueError("Full DMD rollout requires at least one step.")
    if gradient_step_idx is not None and not 0 <= gradient_step_idx < num_steps:
        raise ValueError(f"gradient_step_idx out of range: {gradient_step_idx}")

    selected_latents = None
    selected_velocity = None
    with torch.no_grad():
        xt = initial_latents.detach()
        for idx in range(num_steps):
            if idx == gradient_step_idx:
                selected_latents = xt.detach()
            velocity = predict_velocity(xt, idx)
            xt, endpoint = step_by_index(velocity, idx, xt)
        endpoint = endpoint.detach()

    if gradient_step_idx is not None:
        with torch.enable_grad():
            velocity = predict_velocity(selected_latents, gradient_step_idx)
            _, selected_x0 = step_by_index(velocity, gradient_step_idx, selected_latents)
            selected_velocity = velocity.detach()
            # Subtract first so the forward value stays exact even in bf16.
            endpoint = endpoint + (selected_x0 - selected_x0.detach())

    return endpoint, selected_latents, selected_velocity
