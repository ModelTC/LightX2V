import math
from dataclasses import dataclass

import torch


@dataclass
class HunyuanImage3KVCacheLayer:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


class HunyuanImage3StaticKVCache:
    """Per-layer KV cache matching HunyuanImage3 gen_text/gen_image inference."""

    def __init__(self, num_layers, max_cache_len, dynamic=False, paged=False, page_size=16):
        self.num_layers = int(num_layers)
        requested_cache_len = int(max_cache_len)
        self.dynamic = bool(dynamic)
        self.paged = bool(paged)
        self.page_size = int(page_size)
        if self.page_size < 1:
            raise ValueError(f"HunyuanImage3 KV page_size must be positive, got {self.page_size}.")
        self.num_pages = math.ceil(requested_cache_len / self.page_size) if self.paged else 0
        self.max_cache_len = self.num_pages * self.page_size if self.paged else requested_cache_len
        self.layers = [HunyuanImage3KVCacheLayer() for _ in range(self.num_layers)]
        self.page_table = None
        self.cache_seqlens = None
        self.scheduler_metadata = None
        self.decode_ready = False
        self.num_key_value_heads = None
        self.head_dim = None

    def begin_request(self):
        self.decode_ready = False
        if self.cache_seqlens is not None:
            self.cache_seqlens.zero_()

    def _ensure_layer(self, layer_idx, key_states, value_states):
        layer = self.layers[layer_idx]
        if layer.key is None:
            if self.paged:
                if key_states.shape[0] != 1 or value_states.shape[0] != 1:
                    raise ValueError("HunyuanImage3 paged KV cache currently requires batch size 1.")
                self._ensure_paged_metadata(key_states.device)
                self.num_key_value_heads = int(key_states.shape[1])
                self.head_dim = int(key_states.shape[-1])
                key_shape = (self.num_pages, self.page_size, key_states.shape[1], key_states.shape[-1])
                value_shape = (self.num_pages, self.page_size, value_states.shape[1], value_states.shape[-1])
            else:
                key_shape = (*key_states.shape[:2], self.max_cache_len, key_states.shape[-1])
                value_shape = (*value_states.shape[:2], self.max_cache_len, value_states.shape[-1])
            layer.key = torch.zeros(key_shape, device=key_states.device, dtype=key_states.dtype)
            layer.value = torch.zeros(value_shape, device=value_states.device, dtype=value_states.dtype)
        return layer

    def _ensure_paged_metadata(self, device):
        if self.page_table is None:
            self.page_table = torch.arange(self.num_pages, device=device, dtype=torch.int32).reshape(1, self.num_pages)
            self.cache_seqlens = torch.zeros(1, device=device, dtype=torch.int32)

    def set_paged_decode_length(self, length):
        length = int(length)
        if length < 1 or length > self.max_cache_len:
            raise ValueError(f"HunyuanImage3 paged decode length must be in [1, {self.max_cache_len}], got {length}.")
        self.cache_seqlens.fill_(length)

    def prepare_paged_decode_scheduler(self, *, valid_length, num_query_heads, max_num_splits):
        """Refresh FA3 scheduler data without changing its persistent address."""

        self.set_paged_decode_length(valid_length)
        from lightx2v.common.ops.attn.paged_flash_attn import build_flash_attn3_decode_scheduler_metadata

        runtime_metadata = build_flash_attn3_decode_scheduler_metadata(
            cache_seqlens=self.cache_seqlens,
            max_seqlen_k=self.max_cache_len,
            num_query_heads=int(num_query_heads),
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            qkv_dtype=self.layers[0].key.dtype,
            max_num_splits=int(max_num_splits),
        )
        if self.scheduler_metadata is None:
            fixed_size = max(17, int(runtime_metadata.numel()))
            self.scheduler_metadata = torch.zeros(fixed_size, device=runtime_metadata.device, dtype=torch.int32)
        if runtime_metadata.numel() > self.scheduler_metadata.numel():
            raise RuntimeError(f"FA3 scheduler metadata grew from {self.scheduler_metadata.numel()} to {runtime_metadata.numel()} entries after allocation.")
        self.scheduler_metadata.zero_()
        self.scheduler_metadata[: runtime_metadata.numel()].copy_(runtime_metadata)
        self.decode_ready = True
        return self.scheduler_metadata[: runtime_metadata.numel()]

    def get_paged_layer(self, layer_idx):
        layer = self.layers[int(layer_idx)]
        return layer.key, layer.value

    def update(self, key_states, value_states, layer_idx, cache_position=None):
        layer = self._ensure_layer(layer_idx, key_states, value_states)
        if self.paged:
            return self._update_paged(layer, key_states, value_states, cache_position)
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
            raise ValueError(f"HunyuanImage3 cache batch mismatch: cache_position={cache_position.shape}, key_states={key_states.shape}.")

        for batch_idx in range(cache_position.shape[0]):
            layer.key[batch_idx].index_copy_(1, cache_position[batch_idx], key_states[batch_idx])
            layer.value[batch_idx].index_copy_(1, cache_position[batch_idx], value_states[batch_idx])
        return self._slice_dynamic(layer, int(cache_position.max().item()) + 1)

    def _update_paged(self, layer, key_states, value_states, cache_position):
        positions = cache_position.to(device=key_states.device, dtype=torch.long)[0]

        flat_key = layer.key.reshape(self.max_cache_len, layer.key.shape[2], layer.key.shape[3])
        flat_value = layer.value.reshape(self.max_cache_len, layer.value.shape[2], layer.value.shape[3])
        new_key = key_states.transpose(1, 2).reshape(-1, key_states.shape[1], key_states.shape[-1])
        new_value = value_states.transpose(1, 2).reshape(-1, value_states.shape[1], value_states.shape[-1])
        flat_key.index_copy_(0, positions, new_key)
        flat_value.index_copy_(0, positions, new_value)

        end = int(positions[-1].item()) + 1
        end = min(int(end), self.max_cache_len)
        dense_key = flat_key[:end].unsqueeze(0).transpose(1, 2)
        dense_value = flat_value[:end].unsqueeze(0).transpose(1, 2)
        return dense_key, dense_value

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
