"""Checkpoint path discovery without importing the inference runtime."""

import os


def find_torch_model_path(config, ckpt_config_key=None, filename=None, subdir=["original", "fp8", "int8", "distill_models", "distill_fp8", "distill_int8"]):
    if ckpt_config_key and config.get(ckpt_config_key, None) is not None:
        return config.get(ckpt_config_key)

    paths_to_check = [
        os.path.join(config["model_path"], filename),
    ]
    if isinstance(subdir, list):
        for sub in subdir:
            paths_to_check.insert(0, os.path.join(config["model_path"], sub, filename))
    else:
        paths_to_check.insert(0, os.path.join(config["model_path"], subdir, filename))

    for path in paths_to_check:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"PyTorch model file '{filename}' not found.\nPlease download the model from https://huggingface.co/lightx2v/ or specify the model path in the configuration file.")
