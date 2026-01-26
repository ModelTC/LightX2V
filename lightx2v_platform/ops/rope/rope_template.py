import os
from abc import ABCMeta, abstractmethod
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
def GET_DTYPE():
    RUNNING_FLAG = os.getenv("DTYPE", "BF16")
    assert RUNNING_FLAG in ["BF16", "FP16"]
    return DTYPE_MAP[RUNNING_FLAG]


class RopeTemplate(metaclass=ABCMeta):
    def __init__(self):
        self.infer_dtype = GET_DTYPE()
        self.config = {}

    @abstractmethod
    def apply(self, xq: torch.Tensor, xk: torch.Tensor, cos_sin_cache: torch.Tensor):
        """
        Apply rotary position embedding to query and key tensors.

        Args:
            xq: Query tensor
            xk: Key tensor
            cos_sin_cache: Cosine and sine cache for rotary embedding

        Returns:
            Tuple of (xq, xk) with rotary embedding applied
        """
        pass

    def set_config(self, config=None):
        if config is not None:
            self.config = config
