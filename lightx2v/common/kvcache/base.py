import torch


class BaseKVCachePool:
    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._num_layers = num_layers
        self._cache_size = cache_size
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._device = device
        self._dtype = dtype

    def _init_kv_buffer(self):
        self._k_buffer = torch.zeros(
            (self._num_layers, self._cache_size, self._num_heads, self._head_dim),
            dtype=self._dtype,
            device=self._device,
        )
        self._v_buffer = torch.zeros(
            (self._num_layers, self._cache_size, self._num_heads, self._head_dim),
            dtype=self._dtype,
            device=self._device,
        )

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        return self._k_buffer[layer_id][attn_start:local_end]

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        return self._v_buffer[layer_id][attn_start:local_end]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        self._k_buffer[layer_id, : k.shape[0]] = k
        self._v_buffer[layer_id, : v.shape[0]] = v

    def reset(self) -> None:
        self._k_buffer.zero_()
        self._v_buffer.zero_()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def cache_size(self) -> int:
        return self._cache_size
