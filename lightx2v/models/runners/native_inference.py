"""Configuration admission for single-card native inference."""

import os

import torch

from lightx2v.utils.envs import GET_DTYPE


def validate_a100_config(config):
    platform = os.getenv("PLATFORM", "cuda")
    if platform == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform == "metax_cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif platform == "cambricon_mlu" and getattr(torch, "mlu", None) is not None and torch.mlu.is_available():
        device = torch.device("mlu")
    else:
        raise RuntimeError(f"No available device for PLATFORM={platform!r}")
    if config.get("implementation_status", "implemented") != "implemented":
        raise NotImplementedError("The selected hardware configuration is a placeholder")
    if config.get("inference_dtype", "bf16") != "bf16" or GET_DTYPE() != torch.bfloat16:
        raise ValueError("These mixed-precision adapters require inference_dtype=bf16 and DTYPE=BF16")
    if platform == "cuda" and not torch.cuda.is_bf16_supported():
        raise NotImplementedError("This inference implementation requires native BF16 support")
    if platform == "metax_cuda" and not torch.cuda.is_bf16_supported():
        raise NotImplementedError("This inference implementation requires native MetaX BF16 support")
    if platform == "cambricon_mlu":
        is_bf16_supported = getattr(torch.mlu, "is_bf16_supported", None)
        if is_bf16_supported is not None and not is_bf16_supported():
            raise NotImplementedError("This inference implementation requires native MLU BF16 support")
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
    config["device"] = str(device)
    return device
