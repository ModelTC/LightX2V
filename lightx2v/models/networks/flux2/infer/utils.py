from typing import Tuple

import torch

from lightx2v.common.ops.rope import FlashInferRope, TorchRealRope

_FLUX_FLASH_ROPE = FlashInferRope(layout="interleaved")
_FLUX_TORCH_ROPE = TorchRealRope(layout="interleaved")


def apply_rope_with_flashinfer(xq, xk, cos_sin_cache):
    return _FLUX_FLASH_ROPE.apply(xq, xk, cos_sin_cache)


def apply_rope_with_torch(xq, xk, freqs_cis) -> Tuple[torch.Tensor, torch.Tensor]:
    return _FLUX_TORCH_ROPE.apply(xq, xk, freqs_cis)
