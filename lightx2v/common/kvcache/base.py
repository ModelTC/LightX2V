import torch
import torch.distributed as dist

from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head


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
        if attn_start is None and local_end is None:
            return self._k_buffer[layer_id]
        return self._k_buffer[layer_id][attn_start:local_end]

    def v_cache(self, layer_id: int, attn_start: int | None = None, local_end: int | None = None) -> torch.Tensor:
        if attn_start is None and local_end is None:
            return self._v_buffer[layer_id]
        return self._v_buffer[layer_id][attn_start:local_end]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, layer_id: int) -> None:
        self._k_buffer[layer_id, : k.shape[0]] = k
        self._v_buffer[layer_id, : v.shape[0]] = v

    def reset(self) -> None:
        self._k_buffer.zero_()
        self._v_buffer.zero_()

    def sp_kvcache_attn(
        self,
        q: torch.Tensor,
        k_cache,
        v_cache,
        attention_module,
        seq_p_group,
        num_heads: int,
        head_dim: int,
        *,
        attn_start: int | None = None,
        local_end: int | None = None,
    ) -> torch.Tensor:
        world_size = dist.get_world_size(seq_p_group)
        shard_heads = num_heads // world_size

        full_q = all2all_seq2head(q, group=seq_p_group)
        if isinstance(k_cache, tuple) or isinstance(v_cache, tuple):
            full_k, full_v, full_kv_len = self._sp_quant_kv_to_head_shard(
                k_cache=k_cache,
                v_cache=v_cache,
                shard_heads=shard_heads,
                seq_p_group=seq_p_group,
                attn_start=attn_start,
                local_end=local_end,
            )
        else:
            full_k = all2all_seq2head(k_cache, group=seq_p_group)
            full_v = all2all_seq2head(v_cache, group=seq_p_group)
            full_kv_len = int(full_k.size(0))

        q_lens = torch.tensor([full_q.size(0)], dtype=torch.int32)
        k_lens = torch.tensor([full_kv_len], dtype=torch.int32)
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)

        attn_out = attention_module.apply(
            q=full_q,
            k=full_k,
            v=full_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=full_q.size(0),
            max_seqlen_kv=full_kv_len,
        )
        attn_out = attn_out.view(full_q.size(0), shard_heads, head_dim)
        attn_out = all2all_head2seq(attn_out, group=seq_p_group)
        return attn_out.reshape(q.size(0), num_heads * head_dim)

    def _sp_quant_kv_to_head_shard(
        self,
        k_cache,
        v_cache,
        shard_heads: int,
        seq_p_group,
        *,
        attn_start: int | None = None,
        local_end: int | None = None,
    ):
        raise TypeError(
            f"{self.__class__.__name__} does not support tuple K/V in SP path. "
            "Please use a cache class that implements _sp_quant_kv_to_head_shard."
        )

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
