import os
from functools import lru_cache

import torch

DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
    "FP32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


@lru_cache(maxsize=None)
def CHECK_PROFILING_DEBUG_LEVEL(target_level):
    current_level = int(os.getenv("PROFILING_DEBUG_LEVEL", "0"))
    return current_level >= target_level


@lru_cache(maxsize=None)
def CHECK_ENABLE_GRAPH_MODE():
    ENABLE_GRAPH_MODE = os.getenv("ENABLE_GRAPH_MODE", "false").lower() == "true"
    return ENABLE_GRAPH_MODE


@lru_cache(maxsize=None)
def GET_RUNNING_FLAG():
    RUNNING_FLAG = os.getenv("RUNNING_FLAG", "infer")
    return RUNNING_FLAG


@lru_cache(maxsize=None)
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE", "BF16")
    assert RUNNING_FLAG in ["BF16", "FP16"]
    return DTYPE_MAP[RUNNING_FLAG]


@lru_cache(maxsize=None)
def GET_SENSITIVE_DTYPE():
    RUNNING_FLAG = os.getenv("SENSITIVE_LAYER_DTYPE", "None")
    if RUNNING_FLAG == "None":
        return GET_DTYPE()
    return DTYPE_MAP[RUNNING_FLAG]
