import math
from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.kvcache.attn.rope import PhysicalRoPERingAttention
from lightx2v.common.kvcache.attn.sage_rope import SageRoPERingAttention
from lightx2v.common.kvcache.layout import KVCacheUpdate
from lightx2v.common.ops.norm.rms_norm_weight import apply_projection_norm
from lightx2v.models.networks.wan.infer.autoregressive import WanAutoregressiveTransformerInfer
from lightx2v.models.networks.wan.infer.autoregressive.causal_transformer_infer import WanCausalTransformerInfer
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class EdenLiveTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.compile_backend == "default":
            self.compile_options.setdefault("emulate_precision_casts", True)
        self.text_len = config["text_len"]
        self.freqs_cache = {}
        self.block_offload = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block"
        self.infer_func = self.infer_with_blocks_offload if self.block_offload else self.infer_without_offload
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "phase":
            raise NotImplementedError("EdenLive supports block or model offload")

    def infer(self, weights, pre_infer_out):
        self.current_pre = pre_infer_out
        frames, height, width = pre_infer_out.grid_sizes.tuple
        freq_key = (self.get_rope_frames(frames), height, width, pre_infer_out.x.device)
        if freq_key not in self.freqs_cache:
            freqs = self.prepare_rope_freqs(*freq_key)
            if self.config["rope_type"] == "flashinfer_rope" and not self.config.get("rope_chunk", False) and not self.seq_parallel:
                freqs = weights.blocks[0].compute_phases[0].rope.prepare_freqs(freqs)
            self.freqs_cache[freq_key] = freqs
        self.key_freqs = self.freqs_cache[freq_key]
        visible_tokens = self.get_attention_frames(frames) * height * width
        self.query_freqs = self.key_freqs[visible_tokens - frames * height * width : visible_tokens]
        if self.seq_parallel:
            tokens = frames * height * width
            start = pre_infer_out.adapter_args["seq_start"]
            padding = (-tokens) % dist.get_world_size(self.seq_p_group)
            query_key = "query", freq_key, frames, visible_tokens, start, pre_infer_out.x.shape[0]
            if query_key not in self.freqs_cache:
                self.freqs_cache[query_key] = F.pad(self.query_freqs, (0, 0) * (self.query_freqs.ndim - 1) + (0, padding))[start : start + pre_infer_out.x.shape[0]]
            self.query_freqs = self.freqs_cache[query_key]
        return super().infer(weights, pre_infer_out)

    def reset_infer_states(self, x, context):
        super().reset_infer_states(x, context)
        if self.has_image_context:
            segments = self.current_pre.adapter_args.get("text_segments")
            text_tokens = self.text_len * (len(segments) if segments is not None else 1)
            self.image_context_len = context.shape[0] - text_tokens
            self.cross_attn_cu_seqlens_kv_img = torch.tensor([0, self.image_context_len], dtype=torch.int32)
            self.cross_attn_cu_seqlens_kv = torch.tensor([0, text_tokens], dtype=torch.int32)

    def get_attention_frames(self, frames):
        return frames

    def get_rope_frames(self, frames):
        return self.get_attention_frames(frames)

    def get_compile_block_key(self, block_idx, block):
        return id(block) if self.block_offload else block_idx

    def clear_request_state(self):
        self.current_pre = None
        self.key_freqs = None
        self.query_freqs = None
        self.freqs_cache = {}

    def infer_block(self, block, x, pre_infer_out, cross_kv=None, self_attn_cache=None):
        # The timestep embedding is shared by every block.
        shift, scale, gate, ff_shift, ff_scale, ff_gate = pre_infer_out.embed0.add(block.compute_phases[0].modulation.tensor).chunk(6, dim=1)
        y = self.infer_self_attn(block.compute_phases[0], x, shift, scale, pre_infer_out.grid_sizes.tuple, self_attn_cache=self_attn_cache)
        x.add_(y.mul_(gate.squeeze(1)))
        attention = self.infer_cross_attn_output(block.compute_phases[1], x, pre_infer_out.context, cross_kv)
        x.add_(attention)
        phase = block.compute_phases[2]
        normalized = self.modulate_func(phase.norm2.apply(x), scale=ff_scale, shift=ff_shift).reshape_as(x)
        hidden = phase.ffn_0.apply(normalized)
        ffn = getattr(phase, "ffn_gelu", None)
        y = ffn.apply(hidden) if ffn is not None else phase.ffn_2.apply(F.gelu(hidden, approximate="tanh"))
        x.add_(y.mul_(ff_gate.squeeze(1)))
        return x

    def infer_non_blocks(self, weights, x, embed):
        shift, scale = embed.unsqueeze(1).add(weights.head_modulation.tensor).unbind(1)
        x = weights.norm.apply(x).mul_(scale.add_(1)).add_(shift)
        return weights.head.apply(x)

    def infer_cross_attn_output(self, phase, x, context, cross_kv=None):
        segments = self.current_pre.adapter_args["text_segments"]
        if self.has_image_context:
            context_img = context[: self.image_context_len]
            context = context[self.image_context_len :]
        normalized = phase.norm3.apply(x)
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(normalized)).view(-1, self.num_heads, self.head_dim)
        k, v = self.get_cross_kv(phase, context) if cross_kv is None else cross_kv[:2]
        if segments is None:
            attention = phase.cross_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
                max_seqlen_q=q.shape[0],
                max_seqlen_kv=k.shape[0],
            )
        else:
            tokens_per_frame = math.prod(self.current_pre.grid_sizes.tuple[1:])
            local_start = self.current_pre.adapter_args.get("seq_start", 0)
            attention = torch.zeros_like(q)
            cursor = 0
            for index, frames in enumerate(segments):
                end = cursor + frames * tokens_per_frame
                left, right = max(cursor - local_start, 0), min(end - local_start, q.shape[0])
                if right > left:
                    text_start = index * self.text_len
                    text_end = text_start + self.text_len
                    output = phase.cross_attn_1.apply(
                        q=q[left:right],
                        k=k[text_start:text_end],
                        v=v[text_start:text_end],
                        cu_seqlens_q=torch.tensor([0, right - left], dtype=torch.int32),
                        cu_seqlens_kv=torch.tensor([0, self.text_len], dtype=torch.int32),
                        max_seqlen_q=right - left,
                        max_seqlen_kv=self.text_len,
                    )
                    attention[left:right] = output.view(right - left, self.num_heads, self.head_dim)
                cursor = end
            attention = attention.flatten(1)
        if self.has_image_context:
            k_img, v_img = self.get_image_cross_kv(phase, context_img) if cross_kv is None else cross_kv[2:]
            image_attention = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
                max_seqlen_q=q.shape[0],
                max_seqlen_kv=k_img.shape[0],
            )
            attention.add_(image_attention)
        return phase.cross_attn_o.apply(attention)

    def get_image_cross_kv(self, phase, context):
        key = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context)).view(-1, self.num_heads, self.head_dim)
        return key, phase.cross_attn_v_img.apply(context).view_as(key)

    def get_cross_kv(self, phase, context):
        group = getattr(phase, "cross_kv_projection_group", None)
        if group is None:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, self.num_heads, self.head_dim)
            return k, phase.cross_attn_v.apply(context).view_as(k)
        k, v = group.apply(context)
        k = apply_projection_norm(phase.cross_attn_norm_k, k).view(-1, self.num_heads, self.head_dim)
        v = v.view_as(k)
        return k, v

    def prepare_rope_freqs(self, frames, height, width, device):
        half = self.head_dim // 2
        widths = (half - 2 * (half // 3), half // 3, half // 3)
        axes = []
        for count, size in zip((frames, height, width), widths):
            angle = torch.outer(torch.arange(count, device=device, dtype=torch.float64), 10000 ** (-torch.arange(size, device=device, dtype=torch.float64) / size))
            axes.append(torch.polar(torch.ones_like(angle), angle).to(torch.complex64))
        return torch.cat(
            (axes[0][:, None, None].expand(-1, height, width, -1), axes[1][None, :, None].expand(frames, -1, width, -1), axes[2][None, None, :].expand(frames, height, -1, -1)), dim=-1
        ).reshape(-1, 1, half)

    def infer_self_attn(self, phase, x, shift_msa, scale_msa, grid_sizes=None, self_attn_cache=None, **kwargs):
        normalized = self.modulate_func(phase.norm1.apply(x), scale=scale_msa, shift=shift_msa).reshape_as(x)
        group = getattr(phase, "self_attn_projection_group", None)
        if group is None:
            q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(normalized)).view(-1, self.num_heads, self.head_dim)
            k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(normalized)).view_as(q)
            v = phase.self_attn_v.apply(normalized).view_as(q)
        else:
            q, k, v = group.apply(normalized)
            q = apply_projection_norm(phase.self_attn_norm_q, q).view(-1, self.num_heads, self.head_dim)
            k = apply_projection_norm(phase.self_attn_norm_k, k).view_as(q)
            v = v.view_as(q)
        k, v = self.gather_self_attn_kv(k, v, math.prod(grid_sizes))
        attention = self.infer_self_attn_output(phase, q, k, v, grid_sizes, self_attn_cache)
        return phase.self_attn_o.apply(attention)

    def gather_self_attn_kv(self, key, value, token_count):
        if not self.seq_parallel:
            return key, value
        world_size = dist.get_world_size(self.seq_p_group)
        shape = (world_size * key.shape[0], *key.shape[1:])
        gathered_key, gathered_value = key.new_empty(shape), value.new_empty(shape)
        dist.all_gather_into_tensor(gathered_key, key.contiguous(), group=self.seq_p_group)
        dist.all_gather_into_tensor(gathered_value, value.contiguous(), group=self.seq_p_group)
        return gathered_key[:token_count], gathered_value[:token_count]

    def infer_self_attn_output(self, phase, q, k, v, grid_sizes, self_attn_cache=None):
        if q.shape[0] == k.shape[0]:
            q, k = phase.rope.apply(q, k, self.key_freqs)
        else:
            q = phase.rope.apply_single(q, self.query_freqs)
            k = phase.rope.apply_single(k, self.key_freqs)
        return self.apply_self_attention(phase, q, k, v)

    def apply_self_attention(self, phase, q, k, v):
        return phase.self_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, k.shape[0]], dtype=torch.int32),
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )


class EdenLiveARTransformerInfer(EdenLiveTransformerInfer, WanCausalTransformerInfer):
    """Cache raw K in ring slots and re-anchor RoPE for each visible window."""

    def __init__(self, config):
        super().__init__(config)
        if config["self_attn_1_type"] in ("flash_attn2", "flash_attn3"):
            self.self_attn_backend = PhysicalRoPERingAttention()
        elif config["self_attn_1_type"] == "sage_attn2":
            self.self_attn_backend = SageRoPERingAttention(profile="edenlive")
        else:
            raise ValueError("EdenLive AR requires SageAttention2 or FlashAttention2/3")
        self.cache_pools = {}
        self.kv_cache_manager.key_only = isinstance(self.self_attn_backend, SageRoPERingAttention)
        self._cache_signature = None
        self._graph_cross_generation = 0
        self.cache_commit_enabled &= not config.get("enable_cfg", False)
        self.cross_kv_compute = super().get_cross_kv
        self.image_cross_kv_compute = super().get_image_cross_kv
        if self.use_compile and self.compile_backend == "default":
            self.cross_kv_compute = torch.compile(self.cross_kv_compute, dynamic=None, options=self.compile_options)
            self.image_cross_kv_compute = torch.compile(self.image_cross_kv_compute, dynamic=None, options=self.compile_options)

    def get_rope_frames(self, frames):
        # Growing windows use an unchanged prefix of this table. Keeping its
        # storage/revision stable avoids invalidating historical quantized K
        # solely because the visible window gained another chunk.
        return self.config["ar_config"]["local_attn_size"]

    def initialize_kv_cache(self, latent_shape, ref_num_frames=None):
        ar = self.config["ar_config"]
        chunk, window, sink = ar["num_frame_per_chunk"], ar["local_attn_size"], ar["sink_size"]
        if not 0 <= sink <= window - chunk or window % chunk or sink % chunk:
            raise ValueError("EdenLive requires whole-chunk window and sink sizes with room for the current chunk")
        frame_tokens = math.prod(size // patch for size, patch in zip(latent_shape[2:], self.config["patch_size"][1:]))
        manager = self.kv_cache_manager
        signature = window * frame_tokens, manager.tensor_num_heads, chunk * frame_tokens, sink * frame_tokens, manager.key_only, self.head_dim, manager.dtype, str(manager.device)
        if signature != self._cache_signature:
            self.reset_cuda_graph()
            manager.self_attn_kv_caches.clear()
            for branch in (True, False) if self.config["enable_cfg"] else (True,):
                pool = manager.create_self_attn_kv_cache(branch, signature[0], num_heads=signature[1])
                if manager.key_only:
                    pool.configure_key_only(write_alignment=(chunk * frame_tokens,))
                pool._init_kv_buffer()
            self._cache_signature = signature
        else:
            manager.reset()
        self.cache_pools = dict(manager.self_attn_kv_caches)
        return latent_shape[1]

    @torch.inference_mode(False)
    def prepare_rope_freqs(self, frames, height, width, device):
        freqs = super().prepare_rope_freqs(frames, height, width, device).squeeze(1)
        return torch.cat((freqs.real, freqs.imag), dim=-1).float().contiguous()

    def infer(self, weights, pre_infer_out):
        self.cache_update_done = False
        self.current_pre = pre_infer_out
        self.reset_infer_states(pre_infer_out.x, pre_infer_out.context)
        self.prepare_condition_cache(weights, pre_infer_out)
        return super().infer(weights, pre_infer_out)

    def infer_non_blocks(self, weights, x, embed):
        if self.cache_update_done and self.scheduler.cache_update:
            return None
        return super().infer_non_blocks(weights, x, embed)

    def can_use_kv_only(self, blocks, pre_infer_out):
        return self.scheduler.cache_update and WanAutoregressiveTransformerInfer.can_use_kv_only(self, blocks, pre_infer_out)

    def infer_kv_block(self, block, x, pre_infer_out):
        phase = block.compute_phases[0]
        shift, scale, _, _, _, _ = pre_infer_out.embed0.add(phase.modulation.tensor).chunk(6, dim=1)
        normalized = self.modulate_func(phase.norm1.apply(x), scale=scale, shift=shift).reshape_as(x)
        group = getattr(phase, "self_attn_cache_kv_projection_group", None)
        if group is None:
            k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(normalized))
            v = phase.self_attn_v.apply(normalized)
        else:
            k, v = group.apply(normalized)
            k = apply_projection_norm(phase.self_attn_norm_k, k)
        key, value = k.view(-1, self.num_heads, self.head_dim), v.view(-1, self.num_heads, self.head_dim)
        return self.gather_self_attn_kv(key, value, math.prod(pre_infer_out.grid_sizes.tuple))

    def prepare_condition_cache(self, weights, pre_infer_out):
        branch = self.scheduler.infer_condition
        name = branch, "text"
        source = pre_infer_out.adapter_args["context_source"]
        changed = self.cross_attn_kv_cache.set_context(name, source, cache_key=pre_infer_out.adapter_args["context_id"])
        if self.has_image_context:
            image_source = pre_infer_out.adapter_args["image_context_source"]
            changed = self.cross_attn_kv_cache.set_context((branch, "image"), image_source) or changed
        if changed:
            self._graph_cross_generation += 1
        if self.block_offload or not self.cross_attn_kv_cache.enabled or source.is_inference():
            return
        context = pre_infer_out.context[self.image_context_len :] if self.has_image_context else pre_infer_out.context
        for layer_id, block in enumerate(weights.blocks):
            if self.cross_attn_kv_cache.get(name, layer_id) is None:
                self.cross_attn_kv_cache.store(name, layer_id, self.cross_kv_compute(block.compute_phases[1], context))
            if self.has_image_context and self.cross_attn_kv_cache.get((branch, "image"), layer_id) is None:
                image = pre_infer_out.context[: self.image_context_len]
                kv = self.image_cross_kv_compute(block.compute_phases[1], image)
                self.cross_attn_kv_cache.store((branch, "image"), layer_id, kv)

    def get_cross_kv(self, phase, context, layer_id=None):
        layer_id = self.block_idx if layer_id is None else layer_id
        name = self.scheduler.infer_condition, "text"
        kv = self.cross_attn_kv_cache.get(name, layer_id)
        if kv is None:
            kv = self.cross_attn_kv_cache.store(name, layer_id, self.cross_kv_compute(phase, context))
        return kv

    def get_image_cross_kv(self, phase, context):
        name = self.scheduler.infer_condition, "image"
        kv = self.cross_attn_kv_cache.get(name, self.block_idx)
        if kv is None:
            kv = self.cross_attn_kv_cache.store(name, self.block_idx, self.image_cross_kv_compute(phase, context))
        return kv

    def prepare_rope_cache(self, phase, x, grid_sizes, layer):
        ar = self.config["ar_config"]
        frames, height, width = grid_sizes
        frame_tokens = height * width
        pool = self.cache_pools[self.scheduler.infer_condition]
        state = self.self_attn_backend.prepare_rope_ring(
            pool,
            layer,
            cache_key=(self.scheduler.infer_condition, layer),
            current_tokens=frames * frame_tokens,
            visible_tokens=self.get_attention_frames(frames) * frame_tokens,
            sink_tokens=ar["sink_size"] * frame_tokens,
            freqs=self.key_freqs,
            generation=self.scheduler.chunk_start,
        )
        return state

    def finish_rope_cache(self, pool, layer, grid_sizes):
        frames, height, width = grid_sizes
        frame_tokens = height * width
        end = pool.get_valid_length(layer)
        visible_end = self.get_attention_frames(frames) * frame_tokens
        evicted = max(0, end + frames * frame_tokens - visible_end)
        update = KVCacheUpdate(visible_end - frames * frame_tokens, visible_end, 0, (self.scheduler.chunk_start + frames) * frame_tokens, self.config["ar_config"]["sink_size"] * frame_tokens, evicted)
        if self.scheduler.cache_update:
            pool.finish_attention(layer, update)
            pool.roll_window(layer, update.sink_tokens, evicted)
            pool.ring_layout(layer).record_write(visible_end)
            pool.set_ends(layer, update.global_end, visible_end)
        else:
            pool.end_layer(layer)

    def infer_self_attn_output(self, phase, q, k, v, grid_sizes, self_attn_cache):
        return self.self_attn_backend.apply_rope_ring(
            q,
            k,
            v,
            self_attn_cache,
            attention=phase.self_attn_1,
            rope=phase.rope,
            query_freqs=self.query_freqs,
        )

    def get_block_cross_kv(self, block, context, layer_id=None):
        text = context[self.image_context_len :] if self.has_image_context else context
        kv = self.get_cross_kv(block.compute_phases[1], text, layer_id)
        if self.has_image_context:
            self.block_idx = layer_id
            kv += self.get_image_cross_kv(block.compute_phases[1], context[: self.image_context_len])
        return kv

    def infer_blocks(self, x, embed0, *cross_inputs, pre_infer_out, functions, caches):
        pre = replace(pre_infer_out, x=x, embed0=embed0)
        previous = self.current_pre
        self.current_pre = pre
        cross_count = 4 if self.has_image_context else 2
        try:
            for index, (function, cache) in enumerate(zip(functions, caches)):
                x = function(x, pre, cross_inputs[cross_count * index : cross_count * (index + 1)], cache)
            return x
        finally:
            self.current_pre = previous

    def can_use_cuda_graph(self, blocks, x, pre_infer_out):
        return self.cuda_graph_stack is not None and x.is_cuda

    def infer_without_offload(self, blocks, x, pre_infer_out):
        if self._ar_kv_offload:
            for index, block in enumerate(blocks):
                x = self.run_block(index, block, x, pre_infer_out)
            return x
        graph_blocks = blocks[:-1] if self.cache_update_only else blocks
        if graph_blocks:
            grid = pre_infer_out.grid_sizes.tuple
            pool = self.cache_pools[self.scheduler.infer_condition]
            caches = tuple(self.prepare_rope_cache(block.compute_phases[0], x, grid, index) for index, block in enumerate(graph_blocks))
            pool.upload_write_metadata()
            cross_kv = tuple(self.get_block_cross_kv(block, pre_infer_out.context, index) for index, block in enumerate(graph_blocks))
            x = self.run_prepared_blocks(graph_blocks, x, pre_infer_out, caches, cross_kv, tuple(range(len(graph_blocks))), variant="cache_commit" if self.cache_update_only else "denoise")
            for index in range(len(graph_blocks)):
                self.finish_rope_cache(pool, index, grid)
        if self.cache_update_only:
            last = len(blocks) - 1
            x = self.run_block(last, blocks[last], x, pre_infer_out)
        return x

    def run_prepared_blocks(self, blocks, x, pre_infer_out, caches, cross_kv, layer_ids, *, variant):
        functions = tuple(self.get_compiled_block(layer, block) if self.use_compile else partial(self.infer_block, block) for layer, block in zip(layer_ids, blocks))
        runner = partial(self.infer_blocks, pre_infer_out=pre_infer_out, functions=functions, caches=caches)
        inputs = x, pre_infer_out.embed0, *(tensor for cross in cross_kv for tensor in cross)
        if not self.can_use_cuda_graph(blocks, x, pre_infer_out):
            return runner(*inputs)
        branch = self.scheduler.infer_condition
        bindings = (*(tensor for cache in caches for tensor in cache.bindings), self.query_freqs, *self.get_graph_weight_bindings(blocks))
        mutated = tuple(tensor for layer in layer_ids for tensor in self.self_attn_backend.write_views((branch, layer)))
        cross_version = branch, self.cross_attn_kv_cache.generation, self._graph_cross_generation
        kinds = ("text", "image") if self.has_image_context else ("text",)
        cross_versions = []
        for layer in layer_ids:
            for kind in kinds:
                cached = kind == "text" and self.cross_attn_kv_cache.get((branch, kind), layer) is not None
                for projection in range(2):
                    # Offloaded layers can reuse a source buffer; image context can change in place.
                    cross_versions.append((*cross_version, layer, kind, projection) if cached else None)
        segments = pre_infer_out.adapter_args.get("text_segments")
        layout = (
            pre_infer_out.grid_sizes.tuple,
            self.get_attention_frames(pre_infer_out.grid_sizes.tuple[0]),
            tuple(pre_infer_out.embed0.shape),
            None if segments is None else tuple(segments),
            pre_infer_out.adapter_args.get("seq_start", 0),
            tuple(getattr(cache, "graph_key", None) for cache in caches),
        )
        return self.cuda_graph_stack.run(
            (tuple(id(block) for block in blocks), layout, branch),
            runner,
            inputs,
            bindings=bindings,
            mutated_buffers=mutated,
            input_versions=(None, None, *cross_versions),
            variant=(branch, variant, layout),
            memory_pool=self.get_cuda_graph_pool(x.device),
        )

    def infer_cache_commit(self, x, embed0, *, block_idx, block, pre_infer_out, cache):
        pre = replace(pre_infer_out, x=x, embed0=embed0)
        key, value = self.run_kv_block(block_idx, block, x, pre)
        self.self_attn_backend.update_rope_ring(key, value, cache)
        return x

    def run_block(self, block_idx, block, x, pre_infer_out):
        self.block_idx = block_idx
        branch = self.scheduler.infer_condition
        pool = self.cache_pools[branch]
        grid = pre_infer_out.grid_sizes.tuple
        cache = self.prepare_rope_cache(block.compute_phases[0], x, grid, block_idx)
        pool.upload_write_metadata()
        if self.cache_update_only and block_idx == self.blocks_num - 1:
            runner = partial(self.infer_cache_commit, block_idx=block_idx, block=block, pre_infer_out=pre_infer_out, cache=cache)
            if self.can_use_cuda_graph((block,), x, pre_infer_out):
                x = self.cuda_graph_stack.run(
                    (id(block), grid, getattr(cache, "graph_key", None)),
                    runner,
                    (x, pre_infer_out.embed0),
                    bindings=(*cache.bindings, *self.get_graph_weight_bindings((block,))),
                    mutated_buffers=self.self_attn_backend.write_views((branch, block_idx)),
                    variant=(branch, "cache_kv", grid, self.get_attention_frames(grid[0]), getattr(cache, "graph_key", None)),
                    memory_pool=self.get_cuda_graph_pool(x.device),
                )
            else:
                x = runner(x, pre_infer_out.embed0)
            self.cache_update_done = True
        else:
            cross_kv = self.get_block_cross_kv(block, pre_infer_out.context, block_idx)
            x = self.run_prepared_blocks((block,), x, pre_infer_out, (cache,), (cross_kv,), (block_idx,), variant=("block", block_idx))
        self.finish_rope_cache(pool, block_idx, grid)
        return x

    def get_attention_frames(self, frames):
        ar = self.config["ar_config"]
        return min(self.scheduler.chunk_start, ar["local_attn_size"] - ar["num_frame_per_chunk"]) + frames

    def clear_request_state(self):
        freqs = self.freqs_cache
        EdenLiveTransformerInfer.clear_request_state(self)
        WanCausalTransformerInfer.clear_request_state(self)
        self.freqs_cache = freqs
        self.cache_pools = {}

    def reset_cuda_graph(self):
        super().reset_cuda_graph()
        self.cache_update_done = False
