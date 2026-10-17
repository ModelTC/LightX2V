import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.attn.sage_sm90_fused import make_transposed_v, update_transposed_v
from lightx2v.common.ops.norm.rms_norm_weight import apply_projection_norm
from lightx2v.models.networks.wan.infer.edenlive.cuda_graph import EdenLiveARGraphMixin, EdenLiveCUDAGraphStack
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


class EdenLiveTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if self.compile_backend == "default":
            self.compile_options.setdefault("emulate_precision_casts", True)
        self.text_len = config["text_len"]
        self.freqs_cache = {}
        self.block_offload = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block"
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "phase":
            raise NotImplementedError("EdenLive supports block or model offload")

    def infer(self, weights, pre_infer_out):
        self.current_pre = pre_infer_out
        frames, height, width = pre_infer_out.grid_sizes.tuple
        freq_key = (self._visible_frames(frames), height, width, pre_infer_out.x.device)
        if freq_key not in self.freqs_cache:
            freqs = self._freqs(*freq_key)
            if self.config["rope_type"] == "flashinfer_rope" and not self.config.get("rope_chunk", False) and not self.seq_parallel:
                freqs = weights.blocks[0].compute_phases[0].rope.prepare_freqs(freqs)
            self.freqs_cache[freq_key] = freqs
        self.key_freqs = self.freqs_cache[freq_key]
        self.query_freqs = self.key_freqs[-frames * height * width :]
        if self.seq_parallel:
            tokens = frames * height * width
            start = pre_infer_out.adapter_args["seq_start"]
            self.query_freqs = F.pad(self.query_freqs, (0, 0, 0, 0, 0, (-tokens) % dist.get_world_size(self.seq_p_group)))[start : start + pre_infer_out.x.shape[0]]
        return super().infer(weights, pre_infer_out)

    def _visible_frames(self, frames):
        return frames

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
        if self_attn_cache is not None:
            y, raw_k, raw_v = y
        x.add_(y.mul_(gate.squeeze(1)))
        attention = self.infer_cross_attn_output(block.compute_phases[1], x, pre_infer_out.context, cross_kv)
        x.add_(attention)
        phase = block.compute_phases[2]
        normalized = self.modulate_func(phase.norm2.apply(x), scale=ff_scale, shift=ff_shift).reshape_as(x)
        hidden = phase.ffn_0.apply(normalized)
        ffn = getattr(phase, "ffn_gelu", None)
        y = ffn.apply(hidden) if ffn is not None else phase.ffn_2.apply(F.gelu(hidden, approximate="tanh"))
        x.add_(y.mul_(ff_gate.squeeze(1)))
        return x if self_attn_cache is None else (x, raw_k, raw_v)

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
        prepared_kv = cross_kv if cross_kv is not None and len(cross_kv) == 4 else None
        if prepared_kv is None:
            k, v = self.get_cross_kv(phase, context) if cross_kv is None else cross_kv
        if segments is None:
            if prepared_kv is not None:
                attention = phase.cross_attn_1.apply_prepared(q, prepared_kv)
            else:
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
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, self.num_heads, self.head_dim)
            v_img = phase.cross_attn_v_img.apply(context_img).view_as(k_img)
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

    def get_cross_kv(self, phase, context):
        group = getattr(phase, "cross_kv_projection_group", None)
        if group is None:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, self.num_heads, self.head_dim)
            return k, phase.cross_attn_v.apply(context).view_as(k)
        k, v = group.apply(context)
        k = apply_projection_norm(phase.cross_attn_norm_k, k).view(-1, self.num_heads, self.head_dim)
        v = v.view_as(k)
        return k, v

    def _freqs(self, frames, height, width, device):
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
        current_tokens = math.prod(grid_sizes)
        if self.seq_parallel:
            gathered_k = [torch.empty_like(k) for _ in range(dist.get_world_size(self.seq_p_group))]
            gathered_v = [torch.empty_like(v) for _ in gathered_k]
            dist.all_gather(gathered_k, k, self.seq_p_group)
            dist.all_gather(gathered_v, v, self.seq_p_group)
            k, v = torch.cat(gathered_k)[:current_tokens], torch.cat(gathered_v)[:current_tokens]
        attention = self._self_attention(phase, q, k, v, grid_sizes, self_attn_cache)
        output = phase.self_attn_o.apply(attention)
        return output if self_attn_cache is None else (output, k, v)

    def _self_attention(self, phase, q, k, v, grid_sizes, self_attn_cache=None):
        if q.shape[0] == k.shape[0]:
            q, k = phase.rope.apply(q, k, self.key_freqs)
        else:
            q = phase.rope.apply_single(q, self.query_freqs)
            k = phase.rope.apply_single(k, self.key_freqs)
        return self._attention(phase, q, k, v)

    def _attention(self, phase, q, k, v):
        return phase.self_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, k.shape[0]], dtype=torch.int32),
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )


class EdenLiveARTransformerInfer(EdenLiveARGraphMixin, EdenLiveTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.cache_pools = {}
        self.history_cache = {}
        ar = config["ar_config"]
        self.reuse_history = not config.get("cpu_offload", False) and not ar.get("kv_offload", False) and not ar.get("kv_quant") and ar.get("kv_cache_scheme", "fp") == "fp"
        self.transposed_v_cache = hasattr(ATTN_WEIGHT_REGISTER[config["self_attn_1_type"]], "apply_pretransposed")
        self.cross_kv_cache = {}
        self.cuda_graph_stack = EdenLiveCUDAGraphStack() if config.get("use_cuda_graph", False) else None
        self._graph_weight_key = None
        self._graph_cross_generation = 0
        self._commit_weight_key = None
        self._commit_kv_only_active = False
        self._commit_kv_only_done = False
        self._cache_update_kv_functions = {}
        self.reuse_cross_kv = self.reuse_history and self.compile_backend == "default"
        self.reuse_quantized_cross_kv = self.reuse_cross_kv and not self.seq_parallel and hasattr(ATTN_WEIGHT_REGISTER[config["cross_attn_1_type"]], "prepare_kv")
        self.cross_kv_compute = super().get_cross_kv
        if self.use_compile and self.reuse_cross_kv:
            self.cross_kv_compute = torch.compile(self.cross_kv_compute, dynamic=None, options=self.compile_options)

    def infer(self, weights, pre_infer_out):
        self._commit_kv_only_active = False
        self._commit_kv_only_done = False
        if self.reuse_cross_kv:
            self.prepare_cross_kv(weights, pre_infer_out)
        return super().infer(weights, pre_infer_out)

    def infer_non_blocks(self, weights, x, embed):
        if getattr(self, "_commit_kv_only_done", False) and self.scheduler.cache_update:
            # The AR model's commit-only path consumes only the updated cache.
            return None
        return super().infer_non_blocks(weights, x, embed)

    def _can_commit_kv_only(self, blocks, pre):
        if (
            not blocks
            or len(blocks) != self.blocks_num
            or self.config.get("cpu_offload", False)
            or self.config.get("lazy_load", False)
            or self.seq_parallel
            or self.config.get("tensor_parallel", False)
            or self.config.get("enable_cfg", False)
            or self.config.get("cfg_parallel", False)
            or self.config.get("lora_configs")
            or self.config.get("lora_dynamic_apply", False)
            or getattr(self, "has_post_adapter", False)
            or self.has_image_context
            or not self.reuse_history
            or self.compile_backend != "default"
            or pre.adapter_args.get("text_segments") is not None
        ):
            return False
        key = tuple(id(block) for block in blocks)
        if getattr(self, "_commit_weight_key", None) != key:
            leaves = [leaf for block in blocks for _, leaf in block.named_weight_leaves()]
            if any(getattr(leaf, "has_lora_branch", False) or getattr(leaf, "has_diff", False) or hasattr(leaf, "weight_diff") or hasattr(leaf, "bias_diff") for leaf in leaves):
                return False
            self._commit_weight_key = key
        return True

    def infer_cache_update_kv(self, block, x, pre_infer_out):
        """Produce the final layer's raw commit K/V without its unused suffix."""
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
        return k.view(-1, self.num_heads, self.head_dim), v.view(-1, self.num_heads, self.head_dim)

    def get_compiled_cache_update_kv(self, block_idx, block):
        functions = getattr(self, "_cache_update_kv_functions", None)
        if functions is None:
            functions = self._cache_update_kv_functions = {}
        cached = functions.get(block_idx)
        if cached is not None and cached[0] is block:
            return cached[1]

        def compute(x, pre):
            return self.infer_cache_update_kv(block, x, pre)

        options = {"options": self.compile_options} if self.compile_options else {}
        compiled = torch.compile(compute, dynamic=None, **options)
        functions[block_idx] = (block, compiled)
        return compiled

    def prepare_cross_kv(self, weights, pre_infer_out):
        branch = self.scheduler.infer_condition
        context_id = pre_infer_out.adapter_args["context_id"]
        # Segmented prompts must retain independent K means and V scales.
        prepare_quantized = self.reuse_quantized_cross_kv and pre_infer_out.adapter_args.get("text_segments") is None
        cache_key = (context_id, prepare_quantized)
        cached = self.cross_kv_cache.get(branch)
        if cached is None or cached[0] != cache_key:
            context = pre_infer_out.context[-self.text_len :] if self.has_image_context else pre_infer_out.context
            values = []
            for block in weights.blocks:
                phase = block.compute_phases[1]
                kv = self.cross_kv_compute(phase, context)
                values.append(phase.cross_attn_1.prepare_kv(*kv) if prepare_quantized else kv)
            self.cross_kv_cache[branch] = (cache_key, values)
            self._graph_cross_generation = getattr(self, "_graph_cross_generation", 0) + 1

    def run_block(self, block_idx, block, x, pre_infer_out):
        pool = self.cache_pools[self.scheduler.infer_condition]
        if getattr(self, "_commit_kv_only_active", False) and self.scheduler.cache_update and block_idx == self.blocks_num - 1:
            compute = self.get_compiled_cache_update_kv(block_idx, block) if self.use_compile else lambda value, pre: self.infer_cache_update_kv(block, value, pre)
            raw_k, raw_v = compute(x, pre_infer_out)
            self.commit_kv(pool, block_idx, raw_k, raw_v, pre_infer_out.grid_sizes.tuple)
            self._commit_kv_only_done = True
            return x
        if self.config["ar_config"].get("kv_offload", False):
            pool.begin_layer(block_idx)
        grid_sizes = pre_infer_out.grid_sizes.tuple
        cache = self.prepare_self_attn_cache(block.compute_phases[0], x, grid_sizes, block_idx)
        cross_kv = self.cross_kv_cache[self.scheduler.infer_condition][1][block_idx] if self.reuse_cross_kv else None
        x, raw_k, raw_v = super().run_block(block_idx, block, x, pre_infer_out, cross_kv, cache)
        self.commit_kv(pool, block_idx, raw_k, raw_v, grid_sizes)
        if self.config["ar_config"].get("kv_offload", False):
            pool.end_layer(block_idx, next_prefetch=None)
        return x

    def _visible_frames(self, frames):
        ar = self.config["ar_config"]
        return min(self.scheduler.chunk_start, ar["local_attn_size"] - ar["num_frame_per_chunk"]) + frames

    def get_compile_block_key(self, block_idx, block):
        return block_idx

    def clear_request_state(self):
        # A graph binds history and RoPE storage. Reset their validity, while
        # retaining the allocations across requests; text KV is staged by the
        # graph helper and refreshed from the next request's cache.
        graph = getattr(self, "cuda_graph_stack", None)
        history = self.history_cache if graph is not None else {}
        freqs = self.freqs_cache if graph is not None else {}
        super().clear_request_state()
        self.cache_pools = {}
        self.history_cache = {key: (-1, keys, values) for key, (_, keys, values) in history.items()}
        self.freqs_cache = freqs
        self.cross_kv_cache = {}
        self._graph_weight_key = None
        self._commit_weight_key = None
        self._commit_kv_only_active = False
        self._commit_kv_only_done = False

    def invalidate_cuda_graph(self):
        if self.cuda_graph_stack is not None:
            self.cuda_graph_stack.clear()
        self._graph_weight_key = None
        self._commit_weight_key = None
        self._commit_kv_only_done = False
        if hasattr(self, "_cache_update_kv_functions"):
            self._cache_update_kv_functions.clear()

    def prepare_self_attn_cache(self, phase, x, grid_sizes, layer):
        frames, height, width = grid_sizes
        current_tokens = math.prod(grid_sizes)
        pool = self.cache_pools[self.scheduler.infer_condition]
        end = pool.get_local_end(layer)
        sink = self.config["ar_config"]["sink_size"] * height * width
        capacity = self.config["ar_config"]["local_attn_size"] * height * width
        query_start = min(self.scheduler.chunk_start, self.config["ar_config"]["local_attn_size"] - self.config["ar_config"]["num_frame_per_chunk"])
        visible_capacity = (query_start + frames) * height * width
        visible_evict = max(0, end + current_tokens - visible_capacity)
        history_key = (self.scheduler.infer_condition, layer)
        cached = self.history_cache.get(history_key) if self.reuse_history else None
        if self.reuse_history:
            if cached is not None and cached[1].shape[0] != capacity:
                # Warmup may visit a different resolution before the target.
                self.invalidate_cuda_graph()
                cached = None
            if cached is None:
                key_storage = x.new_empty(capacity, self.num_heads, self.head_dim)
                value_storage = make_transposed_v(key_storage[:0].unsqueeze(0), capacity) if self.transposed_v_cache else torch.empty_like(key_storage)
                cached = (-1, key_storage, value_storage)
            _, key_storage, value_storage = cached
            keys = key_storage[:visible_capacity]
            if self.transposed_v_cache:
                padded = (visible_capacity + 127) // 128 * 128
                values = value_storage.view(-1)[: self.num_heads * self.head_dim * padded].view(1, self.head_dim, self.num_heads, padded)
            else:
                values = value_storage[:visible_capacity]
            if cached[0] != self.scheduler.chunk_start:
                if self.transposed_v_cache:
                    # Clear the partial block before writing its valid permuted rows.
                    values[..., visible_capacity // 128 * 128 :].zero_()
                ranges = ((0, sink), (sink + visible_evict, end)) if visible_evict else ((0, end),)
                offset = 0
                for left, right in ranges:
                    target_k = keys[offset : offset + right - left]
                    target_v = target_k if self.transposed_v_cache else values[offset : offset + right - left]
                    pool.v_cache(layer, left, right, out=target_v)
                    if self.transposed_v_cache:
                        # K storage is scratch for V until the ordered copy below.
                        update_transposed_v(target_v.unsqueeze(0), values, offset)
                    pool.k_cache(layer, left, right, out=target_k)
                    offset += right - left
                if offset:
                    keys[:offset].copy_(phase.rope.apply_single(keys[:offset], self.key_freqs[:offset]))
                self.history_cache[history_key] = (self.scheduler.chunk_start, key_storage, value_storage)
        else:
            history_k = pool.k_cache(layer, 0, end)
            history_v = pool.v_cache(layer, 0, end)
            if visible_evict:
                history_k = torch.cat((history_k[:sink], history_k[sink + visible_evict :]))
                history_v = torch.cat((history_v[:sink], history_v[sink + visible_evict :]))
            keys = x.new_empty(visible_capacity, self.num_heads, self.head_dim)
            values = torch.empty_like(keys)
            keys[: history_k.shape[0]].copy_(history_k)
            values[: history_v.shape[0]].copy_(history_v)
        return keys, values

    def commit_kv(self, pool, layer, k, v, grid_sizes):
        if self.scheduler.cache_update:
            frames, height, width = grid_sizes
            current_tokens = math.prod(grid_sizes)
            end = pool.get_local_end(layer)
            capacity = self.config["ar_config"]["local_attn_size"] * height * width
            evict = max(0, end + current_tokens - capacity)
            if pool.get_global_end(layer) != self.scheduler.chunk_start * height * width:
                raise RuntimeError("EdenLive cache commit cursor does not match the chunk")
            if evict:
                sink = self.config["ar_config"]["sink_size"] * height * width
                pool.roll_window(layer, sink, evict)
            end -= evict
            pool.store_kv(k, v, end, end + current_tokens, layer)
            pool.set_ends(layer, (self.scheduler.chunk_start + frames) * height * width, end + current_tokens)

    def _self_attention(self, phase, q, k, v, grid_sizes, self_attn_cache):
        keys, values = self_attn_cache
        current_tokens = k.shape[0]
        if self.reuse_history:
            # FlashInfer rotates in place; retain raw K for the graph-external commit.
            rotated_k = k.clone()
            if q.shape[0] == k.shape[0]:
                q, rotated_k = phase.rope.apply(q, rotated_k, self.query_freqs)
            else:
                q = phase.rope.apply_single(q, self.query_freqs)
                rotated_k = phase.rope.apply_single(rotated_k, self.key_freqs[-current_tokens:])
            keys[-current_tokens:].copy_(rotated_k)
            if self.transposed_v_cache:
                update_transposed_v(v.unsqueeze(0), values, keys.shape[0] - current_tokens)
                return phase.self_attn_1.apply_pretransposed(q, keys, values)
            else:
                values[-current_tokens:].copy_(v)
                return self._attention(phase, q, keys, values)
        else:
            keys[-current_tokens:].copy_(k)
            values[-current_tokens:].copy_(v)
            return super()._self_attention(phase, q, keys, values, grid_sizes)
