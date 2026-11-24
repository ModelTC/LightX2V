"""Shared admission and output metadata for the three A100 model adapters."""

import json
import os
import time
from pathlib import Path

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


def begin_request():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def finish_request(start, output, model, info, **metadata):
    torch.cuda.synchronize()
    stats = dict(
        model=model,
        seed=info.seed,
        task=info.task,
        end_to_end_seconds=time.perf_counter() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        latency_scope="input preprocessing through final output serialization; excludes model initialization",
        **metadata,
    )
    if output:
        Path(str(output) + ".json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    return stats
