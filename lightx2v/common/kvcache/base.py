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

    def sp_kvcache_attn_head_shard(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attention_module,
        seq_p_group,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """SP attention for KV cache stored as [global_seq, local_heads, head_dim].

        The caller keeps Q in the normal sequence-sharded layout
        [local_seq, global_heads, head_dim]. We convert Q to the head-sharded
        Ulysses layout, attend against the already head-sharded KV cache, then
        convert the output back to sequence-sharded layout.
        """
        if isinstance(k_cache, tuple) or isinstance(v_cache, tuple):
            raise TypeError(f"{self.__class__.__name__} does not support tuple K/V in head-shard SP path.")

        world_size = dist.get_world_size(seq_p_group)
        shard_heads = num_heads // world_size
        q_heads = all2all_seq2head(q, group=seq_p_group)
        kv_len = int(k_cache.size(0))

        q_lens = torch.tensor([q_heads.size(0)], dtype=torch.int32)
        k_lens = torch.tensor([kv_len], dtype=torch.int32)
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)

        attn_out = attention_module.apply(
            q=q_heads,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=q_heads.size(0),
            max_seqlen_kv=kv_len,
        )
        attn_out = attn_out.view(q_heads.size(0), shard_heads, head_dim)
        attn_out = all2all_head2seq(attn_out, group=seq_p_group)
        return attn_out.reshape(q.size(0), num_heads * head_dim)

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
