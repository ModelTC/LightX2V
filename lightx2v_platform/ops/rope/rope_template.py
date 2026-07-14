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
    def __init__(self, layout="interleaved", compute_dtype=torch.float32):
        if layout not in {"interleaved", "split_half"}:
            raise ValueError(f"Unsupported RoPE layout: {layout}")
        self.layout = layout
        self.compute_dtype = compute_dtype
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

    def load(self, weight_dict):
        pass

    def to_cpu(self, non_blocking=False):
        pass

    def to_cuda(self, non_blocking=False):
        pass

    def state_dict(self, destination=None):
        return {} if destination is None else destination

    def load_state_dict(self, destination, block_index, adapter_block_index=None):
        return {} if destination is None else destination

    def load_state_dict_from_disk(self, block_index, adapter_block_index=None):
        pass

    def named_parameters(self, prefix=""):
        return iter(())

    def prepare_freqs(self, freqs):
        return freqs

    def apply_single(self, x: torch.Tensor, freqs, **kwargs):
        output, _ = self.apply(x, x.clone(), freqs, **kwargs)
        return output
