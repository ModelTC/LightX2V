import torch


def is_train_cache_dataset(config):
    return config.get("data", {}).get("train", {}).get("name") == "cache_dataset"


def is_cache_build(config):
    return "cache_build" in config


def check_val_is_enabled(config):
    inference_config = config.get("inference", {})
    return inference_config.get("method", "none") != "none" and bool(inference_config.get("infer_every_iters"))


def get_running_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    elif name == "fp16":
        return torch.float16
    elif name == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {name}")
