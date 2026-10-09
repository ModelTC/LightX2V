import torch


def configure_precision(config, *, section="training"):
    """Keep FP32 arithmetic at full precision unless TF32 is explicitly requested."""
    allow_tf32 = bool(config.get(section, {}).get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
