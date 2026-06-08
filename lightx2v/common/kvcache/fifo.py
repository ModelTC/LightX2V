import torch

from .base import BaseKVCachePool


class FIFOKVCachePool(BaseKVCachePool):
    """FIFO KV cache with temporary/predicted slot metadata.

    New K/V tokens are written into free slots. When the cache is full, the
    oldest live slots are evicted first according to their insertion id.
    """

    def __init__(
        self,
        num_layers: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            cache_size=cache_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self._init_kv_buffer()
        self._id = torch.full((num_layers, cache_size), -1, dtype=torch.long, device=device)
        self._mask = torch.zeros((num_layers, cache_size), dtype=torch.bool, device=device)
        self._is_pred = torch.zeros((num_layers, cache_size), dtype=torch.bool, device=device)

    def _valid_slots(self, layer_id: int) -> torch.Tensor:
        valid = self._mask[layer_id].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            return valid
        order = torch.argsort(self._id[layer_id, valid])
        return valid[order]

    def reset(self) -> None:
        self._id.fill_(-1)
        self._mask.zero_()
        self._is_pred.zero_()

    def clear_pred(self) -> None:
        pred = self._is_pred
        self._mask[pred] = False
        self._id[pred] = -1
        self._is_pred[pred] = False

    def _allocate_slots(self, layer_id: int, key_size: int) -> torch.Tensor:
        mask = self._mask[layer_id]
        ids = self._id[layer_id]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]
            mask[to_free] = False
            ids[to_free] = -1
            self._is_pred[layer_id, to_free] = False
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            raise RuntimeError(f"FIFOKVCachePool cannot allocate {key_size} slots")
        return free[:key_size]

    def _next_cache_id(self, layer_id: int) -> torch.Tensor:
        mask = self._mask[layer_id]
        ids = self._id[layer_id]
        if mask.any():
            return ids[mask].max() + 1
        return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_idx: int | None = None,
        end_idx: int | None = None,
        layer_id: int = 0,
        *,
        is_pred: bool = False,
    ) -> torch.Tensor:
        if end_idx is not None and start_idx is not None and end_idx <= start_idx:
            return torch.empty(0, dtype=torch.long, device=self._device)
        key_size = k.shape[0]
        if start_idx is not None and end_idx is not None and end_idx - start_idx != key_size:
            raise ValueError(f"FIFO KV range size mismatch: range={end_idx - start_idx}, key_size={key_size}")
        slots = self._allocate_slots(layer_id, key_size)
        new_id = self._next_cache_id(layer_id)
        self._k_buffer[layer_id, slots] = k
        self._v_buffer[layer_id, slots] = v
        self._mask[layer_id, slots] = True
        self._id[layer_id, slots] = new_id
        self._is_pred[layer_id, slots] = is_pred
        return slots

    def k_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        valid = self._valid_slots(layer_id)
        if attn_start is not None or local_end is not None:
            attn_start = 0 if attn_start is None else int(attn_start)
            local_end = valid.numel() if local_end is None else int(local_end)
            valid = valid[attn_start:local_end]
        return self._k_buffer[layer_id, valid]

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        valid = self._valid_slots(layer_id)
        if attn_start is not None or local_end is not None:
            attn_start = 0 if attn_start is None else int(attn_start)
            local_end = valid.numel() if local_end is None else int(local_end)
            valid = valid[attn_start:local_end]
        return self._v_buffer[layer_id, valid]

    def restore(self, layer_id: int, slots: torch.Tensor) -> None:
        self._mask[layer_id, slots] = False
        self._id[layer_id, slots] = -1
        self._is_pred[layer_id, slots] = False
