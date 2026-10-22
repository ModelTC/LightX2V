"""Shared model-owned state for autoregressive H3 inference.

H3 variants have different attention windows and conditioning. This class owns
the request/cache lifecycle; each model keeps its own KV and RoPE semantics.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.minimax_h3.infer.module_io import MiniMaxH3SequenceParallelState
from lightx2v_platform.base.global_var import AI_DEVICE


class H3AutoregressiveModel:
    kv_cache_name = "media"

    def to_cuda(self):
        # Resident weights may retain pinned CPU copies. Uploading again would
        # replace the live GPU tensors referenced by captured CUDA Graphs.
        resident = not (self.config.get("cpu_offload", False) or self.config.get("lazy_load", False))
        if resident and getattr(self, "_resident_weights_on_device", False):
            return
        super().to_cuda()
        if resident:
            self._resident_weights_on_device = True

    def to_cpu(self):
        self.reset_cuda_graph()
        self._resident_weights_on_device = False
        return super().to_cpu()

    def reset_cuda_graph(self):
        infer = getattr(self, "transformer_infer", None)
        if infer is not None:
            infer.reset_cuda_graph()

    def reset_autoregressive_state(self):
        """Complete cache transfers while retaining storage and captured graphs."""
        manager = getattr(self, "kv_cache_manager", None)
        if manager is not None:
            for cache in manager.self_attn_kv_caches.values():
                cache.sync_all()

    def _bind_rolling_kv_cache(self, capacity, *, step_cache: bool):
        """Create or reset a native H3 rolling cache without moving its storage.

        The model supplies the token capacity. Heads follow TP and Ulysses;
        raw-key RoPE remains the attention implementation's responsibility.
        Diffusion-step caches keep one history per step; clean AV history needs
        only one copy.
        """
        config = self.config
        total_heads = int(config["num_attention_heads"])
        head_dim = int(config["attention_head_dim"])
        sequence_ranks = dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1
        local_heads = total_heads // self.tp_size // sequence_ranks
        signature = (capacity, local_heads, step_cache)
        manager = getattr(self, "kv_cache_manager", None)

        if manager is None or getattr(self, "_kv_cache_signature", None) != signature:
            if manager is not None:
                self.reset_autoregressive_state()
                self.reset_cuda_graph()
            cache_config = dict(config, num_heads=total_heads, dim=total_heads * head_dim, ar_config=dict(config["ar_config"]))
            manager = KVCacheManager(cache_config, device=torch.device(AI_DEVICE), sp_group=self.seq_p_group)
            cache = manager.create_self_attn_kv_cache(
                self.kv_cache_name,
                capacity,
                kv_cache_scheme="rolling",
                step_kv_cache=step_cache,
                num_heads=local_heads,
            )
            cache._init_kv_buffer()
            self.kv_cache_manager = manager
            self._kv_cache_signature = signature
        else:
            self.reset_autoregressive_state()
            manager.reset()

        cache = manager.get_self_attn_kv_cache(self.kv_cache_name)
        self.scheduler.cache = cache
        return cache

    def _seq_parallel_pre_process(self, pre):
        """Pad and shard all H3 rows before causal Ulysses attention."""
        size, rank = dist.get_world_size(self.seq_p_group), dist.get_rank(self.seq_p_group)
        padding = (-pre.hidden_states.shape[0]) % size
        hidden = F.pad(pre.hidden_states, (0, 0, 0, padding))
        pre.sequence_parallel_state = MiniMaxH3SequenceParallelState(0, hidden.shape[0] // size, pre.timestep_indices, pre.adaln_indices, pre.rotary_emb)
        pre.hidden_states = hidden.chunk(size)[rank].contiguous()
        pre.timestep_indices = F.pad(pre.timestep_indices, (0, padding)).chunk(size)[rank].contiguous()
        pre.adaln_indices = F.pad(pre.adaln_indices, (0, padding)).chunk(size)[rank].contiguous()
        return pre

    def _seq_parallel_post_process(self, output, pre):
        return super()._seq_parallel_post_process(output, pre)[: pre.valid_sequence_length]
