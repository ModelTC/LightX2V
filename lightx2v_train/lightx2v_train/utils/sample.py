import torch


def make_sample(inputs=None, conditioning=None, meta=None):
    return {
        "inputs": inputs or {},
        "conditioning": conditioning or {},
        "meta": meta or {},
    }


def sample_inputs(sample):
    return sample.get("inputs", {})


def sample_conditioning(sample):
    return sample.get("conditioning", {})


def sample_meta(sample):
    return sample.get("meta", {})


def sample_prompt(sample, default=""):
    conditioning = sample_conditioning(sample)
    if "prompt" in conditioning:
        return conditioning["prompt"]
    if "prompt" in sample:
        return sample["prompt"]
    return default


def sample_input(sample, *names, default=None):
    inputs = sample_inputs(sample)
    for name in names:
        if name in inputs:
            return inputs[name]
    for name in names:
        if name in sample:
            return sample[name]
    return default


def sample_condition(sample, *names, default=None):
    conditioning = sample_conditioning(sample)
    for name in names:
        if name in conditioning:
            return conditioning[name]
    for name in names:
        if name in sample:
            return sample[name]
    return default


def sample_meta_value(sample, *names, default=None):
    meta = sample_meta(sample)
    for name in names:
        if name in meta:
            return meta[name]
    for name in names:
        if name in sample:
            return sample[name]
    return default


def nested_to_device(value, device, dtype=None):
    if torch.is_tensor(value):
        kwargs = {"device": device}
        if dtype is not None and torch.is_floating_point(value):
            kwargs["dtype"] = dtype
        return value.to(**kwargs)
    if isinstance(value, dict):
        return {key: nested_to_device(item, device, dtype=dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_to_device(item, device, dtype=dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_to_device(item, device, dtype=dtype) for item in value)
    return value


def first_scalar(value, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        return value.flatten()[0].item()
    if isinstance(value, dict):
        for item in value.values():
            return first_scalar(item, default)
        return default
    if isinstance(value, (list, tuple)):
        return first_scalar(value[0], default) if value else default
    return value
