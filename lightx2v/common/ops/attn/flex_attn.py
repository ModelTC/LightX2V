import math
from functools import lru_cache, partial

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention
except ImportError:  # Keep unrelated LightX2V models importable on older Torch.
    create_block_mask = None
    torch_flex_attention = None

from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

from .template import AttnWeightTemplate


def _score_mod_impl(score, b_idx, h_idx, q_idx, kv_idx, *, hw: int, log_scale: float):
    scaled_region = (kv_idx >= hw) & (kv_idx < 2 * hw)
    return torch.where(scaled_region, score + log_scale, score)


@lru_cache(maxsize=32)
def _score_mod(hw: int, log_scale: float):
    return partial(_score_mod_impl, hw=hw, log_scale=log_scale)


def _mask_mod(q_limit: int, ref_limit: int, q_total: int, hw: int):
    def attention_mask(b_idx, h_idx, q_idx, kv_idx):
        q_valid = q_idx < q_limit
        is_generation_key = kv_idx < q_limit
        is_first_part = kv_idx < q_total

        generation_frame = kv_idx // hw
        generation_valid = kv_idx < q_limit
        reference_index = kv_idx - q_total
        reference_frame = reference_index // hw + 1
        reference_valid = reference_index < ref_limit

        key_frame = torch.where(is_first_part, generation_frame, reference_frame)
        key_valid = torch.where(is_first_part, generation_valid, reference_valid)
        same_frame_reference = (q_idx // hw == key_frame) & key_valid
        return q_valid & (is_generation_key | same_frame_reference)

    return attention_mask


class _FlexMaskCache:
    def __init__(self):
        self._cache = {}

    def get(self, origin_len: int, origin_area, device):
        width, height = (int(value) for value in origin_area)
        key = (int(origin_len), width, height, str(device))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        latent_frames = int(origin_len) // 4 + 1
        hw = width * height // 256
        q_limit = (latent_frames + 1) * hw
        ref_limit = latent_frames * hw
        q_total = math.ceil(q_limit / 128) * 128
        ref_total = math.ceil(ref_limit / 128) * 128
        block_mask = create_block_mask(
            _mask_mod(q_limit, ref_limit, q_total, hw),
            B=None,
            H=None,
            Q_LEN=q_total,
            KV_LEN=q_total + ref_total,
            device=device,
            _compile=True,
        )
        cached = (block_mask, q_total, ref_total, hw)
        self._cache[key] = cached
        return cached


@ATTN_WEIGHT_REGISTER("flex_attn")
class FlexAttnWeight(AttnWeightTemplate):
    """Weightless torch FlexAttention backend with reusable compiled masks."""

    def __init__(self):
        if create_block_mask is None or torch_flex_attention is None:
            raise ImportError("Wan-Animate-2 requires torch.nn.attention.flex_attention; install a PyTorch build that provides FlexAttention.")
        self.config = {}
        self._masks = _FlexMaskCache()
        self._compiled_attention = torch.compile(
            torch_flex_attention,
            dynamic=False,
            mode="max-autotune",
            fullgraph=True,
            backend="inductor",
        )

    def mask_layout(self, origin_len, origin_area, device):
        return self._masks.get(origin_len, origin_area, device)

    def apply(self, q, k, v, *, origin_len, origin_area, log_scale=0.0, **kwargs):
        del kwargs
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError(f"FlexAttnWeight expects [seq, heads, dim], got q={q.shape}, k={k.shape}, v={v.shape}")
        block_mask, q_total, ref_total, hw = self.mask_layout(origin_len, origin_area, q.device)
        if q.shape[0] != q_total or k.shape[0] != q_total + ref_total or v.shape != k.shape:
            raise ValueError(f"FlexAttention packed shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, expected q_len={q_total}, kv_len={q_total + ref_total}.")

        output_dtype = q.dtype
        compute_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)
        output = self._compiled_attention(
            query=q.unsqueeze(0).transpose(1, 2),
            key=k.unsqueeze(0).transpose(1, 2),
            value=v.unsqueeze(0).transpose(1, 2),
            block_mask=block_mask,
            kernel_options=None,
            score_mod=_score_mod(hw, float(log_scale)),
        )
        return output.transpose(1, 2).squeeze(0).to(output_dtype)
