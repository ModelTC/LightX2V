# Adapted from argsss/DPCache (Apache-2.0), commit
# 5d9dcad55b8092be1e4879c5ac256aefa4ab1fdc. See LICENSE.
# H3 adaptation: standalone helpers and strict pair-state DP, no pickle loading.
import math
from collections import deque

import torch


def compute_derivatives_from_history(history_obj, order):
    if len(history_obj) < 1:
        return {}

    if len(history_obj) < 2:
        return {0: history_obj[-1][0]}

    cache = {}

    def compute_derivative_recursive(idx, order):
        cache_key = (idx, order)
        if cache_key in cache:
            return cache[cache_key]

        cur_tensor, cur_step = history_obj[idx]
        if order == 0:
            result = cur_tensor
        elif idx == 0:
            result = None
        else:
            prev_step = history_obj[idx - 1][1]

            current_deriv = compute_derivative_recursive(idx, order - 1)
            prev_deriv = compute_derivative_recursive(idx - 1, order - 1)

            if current_deriv is None or prev_deriv is None:
                return None

            step_distance = cur_step - prev_step
            result = (current_deriv - prev_deriv) / step_distance

        cache[cache_key] = result
        return result

    try:
        derivatives_dict = {}
        for order in range(order + 1):
            deriv = compute_derivative_recursive(len(history_obj) - 1, order)
            if deriv is None:
                break
            derivatives_dict[order] = deriv

        return derivatives_dict
    finally:
        cache.clear()
        del cache


def taylor_formula(derivative_dict: dict, distance: int, clip_fp16=False) -> torch.Tensor:
    output = 0
    for i in range(len(derivative_dict)):
        if clip_fp16:
            output += (1 / math.factorial(i)) * derivative_dict[i].to(torch.float32) * (distance**i)
        else:
            output += (1 / math.factorial(i)) * derivative_dict[i] * (distance**i)

    return output.to(torch.float16) if clip_fp16 else output


def compute_cost(x_pred: torch.Tensor, x_gt: torch.Tensor, metric: str = "l1", clip_fp16: bool = True) -> float:
    x_pred_clip = x_pred.clip(-65504, 65504) if clip_fp16 else x_pred
    x_gt_clip = x_gt.clip(-65504, 65504) if clip_fp16 else x_gt

    if metric == "l1":
        diff = x_pred_clip / 2 - x_gt_clip / 2
        return diff.abs().mean().item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def calc_cost_matrix_v2(cache_data, alpha_list=[0.8], cost_metric="l1"):
    stream = cache_data["current"]["stream"]
    module = cache_data["cache_module_list"][0]
    mode = cache_data["mode"]
    num_steps = cache_data["num_steps"]
    history_obj = cache_data["history"][stream][module]
    clip_fp16 = cache_data["model_config"].get("clip_fp16", False)
    device = history_obj[0][0].device
    order = cache_data["order"]
    history_steps = [item[1] for item in history_obj]
    history_tensors = [item[0] for item in history_obj]

    reference_step = num_steps - 1
    step_derivatives = compute_derivatives_from_history(deque([history_obj[m] for m in range(num_steps - 1 - order, num_steps)]), order)
    sentinel_pred = taylor_formula(step_derivatives, num_steps - reference_step, clip_fp16)

    smoothed_sentinel_pred_list = [alpha * sentinel_pred + (1 - alpha) * history_obj[num_steps][0] for alpha in alpha_list]
    cost_matrix_list = []

    print(f"Calculating 3d cost tensor using {mode} with {cost_metric} metric.")
    anchor_dist_limit = max(int(num_steps * 0.3), 1)

    for alpha_idx, alpha in enumerate(alpha_list):
        cost_tensor_3d = torch.full((num_steps, num_steps + 1, num_steps + 1), float("inf"), device=device)
        current_sentinel = smoothed_sentinel_pred_list[alpha_idx]

        for i in range(order):
            cost_tensor_3d[: i + 1, i, i + 1] = 0

        for i in range(order, num_steps):
            for anchor in range(max(0, i - anchor_dist_limit), i):
                if anchor < order - 1:
                    continue
                accumulated_error = 0.0

                reference_step = history_steps[i]

                step_i_derivatives = compute_derivatives_from_history(deque([history_obj[m] for m in range(anchor - (order - 1), anchor + 1)] + [history_obj[i]]), order)

                for j in range(i + 1, num_steps + 1):
                    distance = j - reference_step
                    x_pred = taylor_formula(step_i_derivatives, distance, clip_fp16)
                    x_gt = current_sentinel if j == num_steps else history_tensors[j]
                    cost = compute_cost(x_pred, x_gt, metric=cost_metric, clip_fp16=clip_fp16)
                    accumulated_error += cost
                    cost_tensor_3d[anchor, i, j] = accumulated_error

        cost_matrix_list.append(cost_tensor_3d.detach().cpu().numpy())

    del history_tensors, history_steps, step_derivatives, sentinel_pred, smoothed_sentinel_pred_list
    return cost_matrix_list
