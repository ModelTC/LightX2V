"""Configuration admission for single-card native inference."""

import os

import torch


def validate_a100_config(config):
    if os.getenv("PLATFORM", "cuda") != "cuda" or not torch.cuda.is_available():
        raise NotImplementedError("This implementation is validated on NVIDIA A100 only; other platform entries are placeholders")
    if config.get("implementation_status", "implemented") != "implemented":
        raise NotImplementedError("The selected hardware configuration is a placeholder")
    for key in ("parallel", "cpu_offload", "cpu_offload_numa", "seq_parallel", "cfg_parallel", "use_cuda_graph", "enable_compile", "dit_quantized", "enable_reuse"):
        if config.get(key):
            raise NotImplementedError(f"{key} is not implemented in these single-card adapters")
    if config.get("feature_caching", "NoCaching") != "NoCaching":
        raise NotImplementedError("Feature caching is not implemented")
    if config.get("dit_quant_scheme", "Default") != "Default":
        raise NotImplementedError("Only unquantized checkpoints are supported")
    for key, backend in {"attn_type": "torch_sdpa", "ln_norm_type": "torch", "conv_type": "Default"}.items():
        if config.get(key, backend) != backend:
            raise NotImplementedError(f"{key}={config[key]!r} has not been implemented or validated")
    if "infer_steps" in config and (not isinstance(config["infer_steps"], int) or config["infer_steps"] < 1):
        raise ValueError("infer_steps must be a positive integer")
