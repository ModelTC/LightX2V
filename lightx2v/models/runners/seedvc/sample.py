"""Load the data contract; feature extraction remains in the application."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_sample(path, config):
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "seedvc-sample-v1" or metadata["variant"] != config["model_variant"]:
        raise ValueError("Sample format/variant does not match the loaded model")
    pre = config["architecture"]["preprocess_params"]
    for key, expected in (("sample_rate", pre["sr"]), ("hop_length", pre["spect_params"]["hop_length"]), ("content_encoder", "openai/whisper-small")):
        if metadata.get(key) != expected:
            raise ValueError(f"Sample {key} must be {expected!r}")
    tensors = {}
    for filename in metadata["tensor_files"]:
        values = load_file(str(path.parent / filename))
        if tensors.keys() & values.keys():
            raise ValueError("Duplicate tensor names in sample")
        tensors.update(values)
    dim = config["architecture"]["model_params"]["length_regulator"]["in_channels"]
    channels = pre["spect_params"]["n_mels"]
    shapes = {"source_content": (1, None, dim), "reference_content": (1, None, dim), "reference_mel": (1, channels, None), "style": (1, 192)}
    if config["model_variant"] == "svc":
        shapes |= {"source_f0": (1, None), "reference_f0": (1, None)}
    for name, shape in shapes.items():
        value = tensors.get(name)
        if value is None or value.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(value.shape, shape, strict=True)):
            raise ValueError(f"Invalid {name}; expected shape {shape}")
        if not value.is_floating_point() or value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point features")
        tensors[name] = value.float()
    if metadata["source_mel_frames"] < 1 or tensors["reference_mel"].shape[-1] > pre["sr"] // pre["spect_params"]["hop_length"] * 25:
        raise ValueError("Source is empty or reference exceeds 25 seconds")
    return metadata, tensors
