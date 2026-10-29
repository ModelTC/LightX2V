"""Opt-in tensor capture for native Qwen-Image-2.1 validation debugging.

Version 1 stores CPU tensors and plain Python metadata only, so consumers can
load traces with ``torch.load(path, weights_only=True)``. Latents and velocities
use the native unpatched token layout [1, tokens, channels]. The condition mask
contains only the text/reference prefix; consumers append target image slots.
"""

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch

FORMAT_VERSION = 1


def _cpu_copy(value):
    return value.detach().to(device="cpu").clone()


def _batched(value, ndim):
    value = _cpu_copy(value)
    if value.ndim == ndim - 1:
        value = value.unsqueeze(0)
    if value.ndim != ndim or value.shape[0] != 1:
        raise ValueError(f"Native preview trace expected one sample with {ndim} dimensions, got {tuple(value.shape)}")
    return value


@contextmanager
def capture_validation_trace(runner, job, model_path):
    """Observe one request without changing model inputs, RNG, or offloading.

    All temporarily wrapped methods are restored even if inference raises. CPU
    copies synchronize accelerator work, so this diagnostic mode is slower.
    """
    if runner.config.get("enable_cfg", False):
        raise ValueError("--dump-trace currently requires native conditional inference (enable_cfg=false).")
    request = job["request"]
    height, width = map(int, request["size"])
    trace = {
        "format_version": FORMAT_VERSION,
        "validation_index": int(job["validation_index"]),
        "height": height,
        "width": width,
        "seed": int(request["seed"]),
        "prompt": str(request["prompt"]),
        "model_path": str(model_path),
        "group": str(job["group"]),
        "torch_version": str(torch.__version__),
        "collection_dtype": os.environ.get("DTYPE", "BF16").lower(),
        "condition": {},
        "steps": [],
    }
    scheduler = runner.scheduler
    original_text = runner.run_text_encoder
    original_prepare = scheduler.prepare
    original_infer = runner.model.infer
    original_step = scheduler.step_post

    @wraps(original_text)
    def run_text_encoder(images, shapes):
        outputs = original_text(images, shapes)
        condition = trace["condition"]
        condition["prompt_embed"] = _batched(outputs["cond"]["prompt_embeds"], 3)
        condition["image_pad_mask"] = _batched(outputs["cond"]["image_mask"], 2).bool()
        condition["prompt_embed_mask"] = torch.ones(condition["prompt_embed"].shape[:2], dtype=torch.bool)
        condition["source_img_shapes"] = [tuple(map(int, shape)) for shape in shapes[:-1]]
        return outputs

    @wraps(original_prepare)
    def prepare(input_info):
        result = original_prepare(input_info)
        trace["initial_latents"] = _batched(scheduler.latents, 3)
        trace["sigmas"] = _cpu_copy(scheduler.sigmas)
        trace["timesteps"] = _cpu_copy(scheduler.timesteps)
        # Record the effective canvas after native dimension rounding.
        trace["height"], trace["width"] = map(int, input_info.size)
        return result

    @wraps(original_infer)
    def infer(inputs):
        if not trace["steps"] and inputs.get("image_latents") is not None:
            trace["condition"]["source_latents"] = _batched(inputs["image_latents"], 3)
        step = {"step_index": int(scheduler.step_index), "latents": _batched(scheduler.latents, 3)}
        result = original_infer(inputs)
        step["noise_pred"] = _batched(scheduler.noise_pred, 3)
        trace["steps"].append(step)
        return result

    @wraps(original_step)
    def step_post():
        result = original_step()
        step = trace["steps"][-1]
        if step["step_index"] != int(scheduler.step_index) or "next_latents" in step:
            raise RuntimeError("Native trace observed an unexpected denoising step order.")
        step["next_latents"] = _batched(scheduler.latents, 3)
        trace["final_latents"] = step["next_latents"]
        return result

    replacements = [
        (runner, "run_text_encoder", run_text_encoder),
        (scheduler, "prepare", prepare),
        (runner.model, "infer", infer),
        (scheduler, "step_post", step_post),
    ]
    missing = object()
    originals = []
    try:
        for owner, name, replacement in replacements:
            originals.append((owner, name, vars(owner).get(name, missing)))
            setattr(owner, name, replacement)
        yield trace
    finally:
        for owner, name, previous in reversed(originals):
            if previous is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)


def save_validation_trace(trace, path):
    """Write only a completed trace; no model or scheduler objects are saved."""
    required = ("initial_latents", "sigmas", "timesteps", "final_latents")
    if any(key not in trace for key in required) or len(trace["steps"]) != len(trace["timesteps"]):
        raise RuntimeError("Cannot save an incomplete native validation trace.")
    if any("next_latents" not in step for step in trace["steps"]):
        raise RuntimeError("Cannot save a trace with an unfinished denoising step.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, path)
