"""Optimizer groups and noise coverage for joint AV recovery distillation."""

import math

import torch


def optimizer_groups(model, config):
    groups = {name: [] for name in ("shared", "experts", "router")}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = "router" if ".router." in name else "shared" if ".shared." in name else "experts"
            groups[group].append(parameter)
    return [
        {"name": name, "params": params, "lr": config.get("router_learning_rate", config["learning_rate"]) if name == "router" else config["learning_rate"], "weight_decay": 0.0 if name == "router" else config.get("weight_decay", 0.01)}
        for name, params in groups.items() if params
    ]


def learning_rate_factor(step, total, warmup=0, minimum=1.0):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup - 1, 1)
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def sample_base_sigma(config):
    base = torch.rand((), device="cpu")
    probability = config.get("uniform_video_sigma_probability", 0.0)
    if probability and torch.rand(()) < probability:
        # Invert the video shift. Audio still uses the SAME base time, preserving
        # H3's coupled video/audio schedule rather than sampling them separately.
        shift = config["flow_shift"]["video"]
        base = base / (shift - (shift - 1) * base)
    return base


def parameter_samples(optimizer):
    """Small local FP32 snapshots, used to prove Adam updates every group."""
    result = {}
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        result[group["name"]] = local.detach().flatten()[::97][:1024].float().clone()
    return result
