from dataclasses import replace
from functools import partial
from math import prod

import torch

from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.common.ops.mm.mm_weight import unwrap_tp_weight
from lightx2v.common.ops.norm.rms_norm_weight import apply_projection_norm
from lightx2v.common.transformer_infer.autoregressive import AutoregressiveTransformerInferMixin
from lightx2v.common.transformer_infer.autoregressive_cuda_graph import AutoregressiveCUDAGraphStack, CUDAGraphPool
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer


class WanAutoregressiveTransformerInfer(AutoregressiveTransformerInferMixin, WanTransformerInfer):
    """Share cached block execution, compilation and graphs for Wan AR models."""

    cache_commit_enabled = False

    def __init__(self, config):
        super().__init__(config)
        ar = config.get("ar_config", {})
        self.num_frame_per_chunk = ar.get("num_frame_per_chunk", 3)
        self._ar_kv_offload = ar.get("kv_offload", False)
        self._model_weight_offload = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "model"
        self.cuda_graph_enabled = config.get("use_cuda_graph", False)
        self.cache_update_only = False
        self.cache_update_done = False
        self.block_graph_pool = None
        self.cuda_graph_stack = AutoregressiveCUDAGraphStack() if config.get("use_cuda_graph", False) else None
        if self.cuda_graph_stack is not None:
            self.compile_options["triton.cudagraphs"] = False

    def get_cuda_graph_pool(self, device):
        if self.block_graph_pool is None:
            self.block_graph_pool = CUDAGraphPool(device)
        return self.block_graph_pool

    def get_graph_weight_bindings(self, blocks):
        if not self._model_weight_offload:
            return ()
        return tuple(tensor for block in blocks for _, leaf in block.named_weight_leaves() for tensor in vars(unwrap_tp_weight(leaf)).values() if isinstance(tensor, torch.Tensor) and tensor.is_cuda)

    def can_use_kv_only(self, blocks, pre_infer_out):
        return self.cache_commit_enabled and len(blocks) == self.blocks_num and not self.has_post_adapter

    def clear_request_state(self):
        self.reset_cross_attn_cache()
        self.cache_update_only = False
        self.cache_update_done = False

    def reset_cuda_graph(self):
        if self.cuda_graph_stack is not None:
            self.cuda_graph_stack.clear()
        self.compiled_kv_blocks.clear()

    def project_cross_attn_kv(self, phase, context, kind="text"):
        context = context.to(self.infer_dtype)
        if kind == "image":
            group = getattr(phase, "cross_img_kv_projection_group", None)
            if group is None:
                key = phase.cross_attn_k_img.apply(context)
                value = phase.cross_attn_v_img.apply(context)
            else:
                key, value = group.apply(context)
            key = apply_projection_norm(phase.cross_attn_norm_k_img, key)
        else:
            group = getattr(phase, "cross_kv_projection_group", None)
            if group is None:
                key = phase.cross_attn_k.apply(context)
                value = phase.cross_attn_v.apply(context)
            else:
                key, value = group.apply(context)
            key = apply_projection_norm(phase.cross_attn_norm_k, key)
        return key.view(-1, self.num_heads, self.head_dim), value.view(-1, self.num_heads, self.head_dim)

    def store_cross_attn_kv(self, name, layer_id, kv):
        return self.cross_attn_kv_cache.store(name, layer_id, kv)

    def get_cross_attn_kv(self, phase, context, kind="text", layer_id=None):
        layer_id = self.block_idx if layer_id is None else layer_id
        name = self._cross_attn_branch, kind
        kv = self.cross_attn_kv_cache.get(name, layer_id)
        if kv is None:
            kv = self.store_cross_attn_kv(name, layer_id, self.project_cross_attn_kv(phase, context, kind))
        return kv

    def set_block_cache_limits(self, block_idx: int) -> None:
        self.kv_cache_size = self.kv_cache_manager.kv_size_for_layer(block_idx)
        self.max_attention_size = self.kv_cache_manager.max_attention_size_for_layer(block_idx)

    def get_kv_cache_update(self, token_count: int, is_ref_prefill: bool = False, *, global_token_count: int | None = None) -> KVCacheUpdate:
        manager = self.kv_cache_manager
        cache = manager.self_attn_kv_cache
        world_size = torch.distributed.get_world_size(self.seq_p_group) if self.seq_parallel else 1
        cached_tokens = global_token_count
        if cached_tokens is None:
            cached_tokens = getattr(self, "_cache_token_count", token_count * world_size)
        seg_index = int(self.scheduler.seg_index)
        current_start = 0 if is_ref_prefill else manager.ref_tokens + seg_index * cached_tokens
        current_end = current_start + cached_tokens
        global_end = cache.get_seq_length(self.block_idx)
        local_end = cache.get_valid_length(self.block_idx)
        frame_tokens = getattr(manager, "frame_seq_length", cached_tokens // self.num_frame_per_chunk)
        sink_tokens = manager.sink_size * frame_tokens
        evicted_tokens = 0
        if manager.uses_sliding_window(self.block_idx) and current_end > global_end:
            evicted_tokens = max(0, local_end + cached_tokens - self.kv_cache_size)
        write_end = local_end - evicted_tokens + current_end - global_end
        return KVCacheUpdate(
            write_start=write_end - cached_tokens,
            write_end=write_end,
            attention_start=max(0, write_end - self.max_attention_size),
            global_end=current_end,
            sink_tokens=sink_tokens,
            evicted_tokens=evicted_tokens,
        )

    def infer_with_kvcache(self, blocks, x, pre_infer_out):
        if self.cache_update_only:
            if len(blocks) > 1:
                x = self.infer_cache_blocks(blocks[:-1], x, pre_infer_out)
            return self.update_self_attn_cache(len(blocks) - 1, blocks[-1], x, pre_infer_out)
        return self.infer_cache_blocks(blocks, x, pre_infer_out)

    def infer_cache_blocks(self, blocks, x, pre_infer_out):
        manager = self.kv_cache_manager
        self.kv_cache_size = manager.kv_size
        self.max_attention_size = manager.max_attention_size
        self._kv_offload = self._ar_kv_offload
        self._cache_token_count = self.get_cache_token_count(x, pre_infer_out)
        if self._kv_offload or not x.is_cuda:
            return self.infer_cached_layers(blocks, x, pre_infer_out)
        if self.cuda_graph_stack is not None and self.can_use_cuda_graph(blocks, x, pre_infer_out):
            return self.infer_with_cuda_graph(blocks, x, pre_infer_out)
        return self.infer_with_ring_attention(blocks, x, pre_infer_out)

    def get_cache_token_count(self, x, pre_infer_out):
        if self.seq_parallel:
            return prod(pre_infer_out.grid_sizes.tuple)
        return x.shape[0]

    def infer_cached_layers(self, blocks, x, pre_infer_out):
        cache = self.kv_cache_manager.self_attn_kv_cache
        for block_idx, block in enumerate(blocks):
            self.block_idx = block_idx
            self.set_block_cache_limits(block_idx)
            x = self.infer_block_func(block, x, pre_infer_out)
        if self._kv_offload:
            cache.sync_all()
        return x

    def can_use_cuda_graph(self, blocks, x, pre_infer_out=None):
        return self.cuda_graph_enabled and x.is_cuda

    def prepare_block_caches(self, blocks, x, pre_infer_out=None):
        cache = self.kv_cache_manager.self_attn_kv_cache
        self_attn_caches, bindings, write_views = [], [], []
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            self.set_block_cache_limits(block_idx)
            cache_update = self.get_kv_cache_update(x.shape[0], getattr(pre_infer_out, "adapter_args", {}).get("is_ref_prefill", False))
            prepared = self.prepare_self_attn_cache(cache, block_idx, cache_update)
            self_attn_caches.append(prepared)
            bindings.extend(prepared.bindings)
            write_views.extend(cache.write_views(block_idx, cache_update.write_start, cache_update.write_end))
        cache.upload_write_metadata()
        return tuple(self_attn_caches), bindings, tuple(write_views)

    def infer_with_ring_attention(self, blocks, x, pre_infer_out):
        self_attn_caches, _, _ = self.prepare_block_caches(blocks, x, pre_infer_out)
        cross_kv = self.prepare_cross_attn_cache(blocks, pre_infer_out.context)
        conditions = self.get_block_conditions(pre_infer_out)
        conditional_names = tuple(conditions)
        start_frame_tensor = self.frame_position(pre_infer_out, x.device)
        return self.infer_blocks(
            x,
            pre_infer_out.embed0,
            pre_infer_out.context,
            pre_infer_out.grid_sizes.tensor,
            start_frame_tensor,
            *(conditions[name] for name in conditional_names),
            blocks=blocks,
            pre_infer_out=pre_infer_out,
            self_attn_caches=self_attn_caches,
            cross_kv=cross_kv,
            conditional_names=conditional_names,
        )

    def infer_with_cuda_graph(self, blocks, x, pre_infer_out):
        self_attn_caches, bindings, write_views = self.prepare_block_caches(blocks, x, pre_infer_out)
        cross_kv = self.prepare_cross_attn_cache(blocks, pre_infer_out.context)
        conditions = self.get_block_conditions(pre_infer_out)
        conditional_names = tuple(conditions)
        for kv in cross_kv:
            bindings.extend(kv)
        bindings.append(pre_infer_out.freqs)
        bindings.extend(self.get_graph_weight_bindings(blocks))
        start_frame_tensor = self.frame_position(pre_infer_out, x.device)
        infer_blocks = partial(self.infer_blocks, blocks=tuple(blocks), pre_infer_out=pre_infer_out, self_attn_caches=self_attn_caches, cross_kv=cross_kv, conditional_names=conditional_names)
        inputs = (x, pre_infer_out.embed0, pre_infer_out.context, pre_infer_out.grid_sizes.tensor, start_frame_tensor, *(conditions[name] for name in conditional_names))
        return self.cuda_graph_stack.run(
            (tuple(id(block) for block in blocks), self.use_compile, conditional_names),
            infer_blocks,
            inputs,
            bindings=tuple(bindings),
            mutated_buffers=write_views,
            variant=(
                "cache_commit" if self.cache_update_only else "denoise",
                self.kv_cache_manager.current_step,
                getattr(pre_infer_out, "adapter_args", {}).get("is_ref_prefill", False),
                tuple(self.kv_cache_manager.self_attn_kv_cache.get_valid_length(layer) for layer in range(len(blocks))),
                tuple(tuple(tensor.shape) for tensor in inputs),
            ),
            memory_pool=self.get_cuda_graph_pool(x.device),
        )

    def get_block_conditions(self, pre_infer_out):
        return pre_infer_out.conditional_dict

    def make_block_pre(self, pre_infer_out, x, embed0, context, grid_sizes, conditions):
        return replace(pre_infer_out, x=x, embed0=embed0, context=context, grid_sizes=replace(pre_infer_out.grid_sizes, tensor=grid_sizes), conditional_dict=conditions)

    def infer_blocks(self, x, embed0, context, grid_sizes, start_frame, *conditional_inputs, blocks, pre_infer_out, self_attn_caches, cross_kv, conditional_names=()):
        graph_pre = self.make_block_pre(pre_infer_out, x, embed0, context, grid_sizes, dict(zip(conditional_names, conditional_inputs)))
        for block_idx, block in enumerate(blocks):
            self.block_idx = block_idx
            x = self.run_block(block_idx, block, x, graph_pre, self_attn_caches[block_idx], cross_kv[block_idx], start_frame)
        return x
