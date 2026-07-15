import math
from dataclasses import dataclass

import torch


@dataclass
class HunyuanImage3KVCacheLayer:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


class HunyuanImage3StaticKVCache:
    """Per-layer KV cache matching HunyuanImage3 gen_text/gen_image inference."""

    def __init__(self, num_layers, max_cache_len, dynamic=False):
        self.num_layers = int(num_layers)
        self.max_cache_len = int(max_cache_len)
        self.dynamic = bool(dynamic)
        self.layers = [HunyuanImage3KVCacheLayer() for _ in range(self.num_layers)]

    def _ensure_layer(self, layer_idx, key_states, value_states):
        layer = self.layers[layer_idx]
        if layer.key is None:
            key_shape = (*key_states.shape[:2], self.max_cache_len, key_states.shape[-1])
            value_shape = (*value_states.shape[:2], self.max_cache_len, value_states.shape[-1])
            layer.key = torch.zeros(key_shape, device=key_states.device, dtype=key_states.dtype)
            layer.value = torch.zeros(value_shape, device=value_states.device, dtype=value_states.dtype)
        return layer

    def update(self, key_states, value_states, layer_idx, cache_position=None):
        layer = self._ensure_layer(layer_idx, key_states, value_states)
        if cache_position is None:
            layer.key[:, :, : key_states.shape[2]].copy_(key_states)
            layer.value[:, :, : value_states.shape[2]].copy_(value_states)
            return self._slice_dynamic(layer, key_states.shape[2])

        cache_position = cache_position.to(device=key_states.device, dtype=torch.long)
        if cache_position.dim() == 1:
            layer.key.index_copy_(2, cache_position, key_states)
            layer.value.index_copy_(2, cache_position, value_states)
            return self._slice_dynamic(layer, int(cache_position[-1].item()) + 1)

        if cache_position.dim() != 2:
            raise ValueError(f"HunyuanImage3 cache_position must be 1D or 2D, got {cache_position.shape}.")
        if cache_position.shape[0] != key_states.shape[0]:
            raise ValueError(
                f"HunyuanImage3 cache batch mismatch: cache_position={cache_position.shape}, key_states={key_states.shape}."
            )

        for batch_idx in range(cache_position.shape[0]):
            layer.key[batch_idx].index_copy_(1, cache_position[batch_idx], key_states[batch_idx])
            layer.value[batch_idx].index_copy_(1, cache_position[batch_idx], value_states[batch_idx])
        return self._slice_dynamic(layer, int(cache_position.max().item()) + 1)

    def _slice_dynamic(self, layer, end):
        if not self.dynamic:
            return layer.key, layer.value
        end = min(int(end), self.max_cache_len)
        return layer.key[:, :, :end], layer.value[:, :, :end]


def _decompose_freqs(x, cutoff_ratio=0.1):
    original_dtype = x.dtype
    x_fp32 = x.float()
    freq = torch.fft.fft(x_fp32, dim=1)
    freqs = torch.fft.fftfreq(x_fp32.shape[1], d=1.0, device=x.device)
    cutoff = cutoff_ratio * freqs.abs().max()
    low_mask = (freqs.abs() <= cutoff)[None, :, None]
    high_mask = ~low_mask
    low = torch.fft.ifft(freq * low_mask, dim=1).real.to(dtype=original_dtype)
    high = torch.fft.ifft(freq * high_mask, dim=1).real.to(dtype=original_dtype)
    return low, high


class HunyuanImage3TaylorCache:
    """Frequency-split Taylor hidden-state cache used by HunyuanImage3 sampling."""

    def __init__(self, max_order):
        self.max_order = int(max_order)
        self.low_derivatives = [None for _ in range(self.max_order + 1)]
        self.high_derivatives = [None for _ in range(self.max_order + 1)]
        self.last_past_key_values = None

    def taylor_formula(self, distance):
        low_output = None
        high_output = None
        for order, derivative in enumerate(self.low_derivatives):
            if derivative is None:
                break
            term = (distance**order / math.factorial(order)) * derivative
            low_output = term if low_output is None else low_output + term
        for order, derivative in enumerate(self.high_derivatives):
            if derivative is None:
                break
            term = (distance**order / math.factorial(order)) * derivative
            high_output = term if high_output is None else high_output + term
        if low_output is None and high_output is None:
            raise RuntimeError("HunyuanImage3 Taylor cache has no derivatives to extrapolate from.")
        if low_output is None:
            return high_output
        if high_output is None:
            return low_output
        return low_output + high_output

    def derivatives_computation(self, hidden_states, distance, low_freqs_order, high_freqs_order):
        low, high = _decompose_freqs(hidden_states)
        new_low = [None for _ in range(self.max_order + 1)]
        new_high = [None for _ in range(self.max_order + 1)]
        new_low[0] = low
        new_high[0] = high
        safe_distance = max(int(distance), 1)

        for order in range(min(int(low_freqs_order), self.max_order)):
            if self.low_derivatives[order] is None:
                break
            new_low[order + 1] = (new_low[order] - self.low_derivatives[order]) / safe_distance

        for order in range(min(int(high_freqs_order), self.max_order)):
            if self.high_derivatives[order] is None:
                break
            new_high[order + 1] = (new_high[order] - self.high_derivatives[order]) / safe_distance

        self.low_derivatives = new_low
        self.high_derivatives = new_high

    def clear_derivatives(self):
        self.low_derivatives = [None for _ in range(self.max_order + 1)]
        self.high_derivatives = [None for _ in range(self.max_order + 1)]
        self.last_past_key_values = None
