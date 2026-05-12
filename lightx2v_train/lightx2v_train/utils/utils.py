import torch


def get_running_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    elif name == "fp16":
        return torch.float16
    elif name == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {name}")
