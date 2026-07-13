from typing import Tuple

import torch

from lightx2v.common.magi_custom_op_mode import set_magi_custom_op_mode, use_magi_custom_ops  # noqa: F401
from lightx2v.common.ops.rope import FlashInferRope, TorchComplexRope, TorchRealRope

_QWEN_FLASH_ROPE = FlashInferRope(layout="interleaved")
_QWEN_TORCH_ROPE = TorchComplexRope()
_QWEN_TORCH_NAIVE_ROPE = TorchRealRope(layout="interleaved")


def apply_qwen_rope_with_flashinfer(xq, xk, cos_sin_cache):
    return _QWEN_FLASH_ROPE.apply(xq, xk, cos_sin_cache)


def apply_qwen_rope_with_torch(xq, xk, cos_sin_cache) -> Tuple[torch.Tensor, torch.Tensor]:
    return _QWEN_TORCH_ROPE.apply(xq, xk, cos_sin_cache)


def apply_qwen_rope_with_torch_naive(xq, xk, cos_sin_cache) -> Tuple[torch.Tensor, torch.Tensor]:
    return _QWEN_TORCH_NAIVE_ROPE.apply(xq, xk, cos_sin_cache)
