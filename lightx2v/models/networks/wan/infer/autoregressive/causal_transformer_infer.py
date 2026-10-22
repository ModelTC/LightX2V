from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.kvcache.attn import RingAttention, SageRingAttention
from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.models.networks.wan.infer.autoregressive import WanAutoregressiveTransformerInfer
from lightx2v.models.networks.wan.infer.autoregressive.compile_ops import layer_norm
from lightx2v.models.networks.wan.infer.autoregressive.rope import apply_dynamic_causal_rope, apply_dynamic_causal_rope_single
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.registry_factory import LN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanCausalTransformerInfer(WanAutoregressiveTransformerInfer, WanOffloadTransformerInfer):
    """Run chunked Wan blocks with pluggable rolling attention and causal RoPE."""

    cache_commit_enabled = True

    def __init__(self, config):
        super().__init__(config)
        self.kv_cache_manager = KVCacheManager(config, device=torch.device(AI_DEVICE), sp_group=self.seq_p_group)
        self.self_attn_backend = SageRingAttention() if config["self_attn_1_type"] == "sage_attn2" else RingAttention()

        self.cleanup_after_infer = self.clean_cuda_cache
        self.clean_cuda_cache = False
        if self.use_compile and self.compile_backend == "default":
            self.compile_options.setdefault("emulate_precision_casts", True)

        if self._ar_kv_offload:
            self.infer_block_func = self.infer_block_with_kvoffload
        else:
            self.infer_block_func = self.infer_block_with_kvcache

        self._weight_offload_block_compute = config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "block"
        self.infer_func = self.infer_with_kvcache_blocks_offload if self._weight_offload_block_compute else self.infer_with_kvcache

    def initialize_kv_cache(self, latent_shape, ref_num_frames=None):
        if latent_shape[1] < self.num_frame_per_chunk:
            raise ValueError(f"Autoregressive inference requires at least {self.num_frame_per_chunk} latent frames")
        self.kv_cache_manager.initialize(latent_shape, ref_num_frames=ref_num_frames, before_allocate=self.reset_cuda_graph)
        if dist.is_initialized() and (self.seq_p_group is not None or dist.get_world_size() > 1):
            dist.barrier(group=self.seq_p_group)
        return self.kv_cache_manager.num_output_frames

    def reset_kv_cache(self):
        self.kv_cache_manager.reset()

    def prepare_self_attn_cache(self, pool, layer_id, update):
        return self.self_attn_backend.prepare(pool, layer_id, update)

    def infer(self, weights, pre_infer_out):
        self.kv_cache_manager.current_step = self.scheduler.step_index
        self.cache_update_only = self.can_use_kv_only(weights.blocks, pre_infer_out)
        output = super().infer(weights, pre_infer_out)
        if self.cleanup_after_infer:
            torch_device_module.empty_cache()
        return output

    def clear_request_state(self):
        super().clear_request_state()
        self.self_attn_backend.reset(keep_storage=True)

    def reset_cuda_graph(self):
        self.block_graph_pool = None
        self.reset_cross_attn_cache()
        super().reset_cuda_graph()
        self.self_attn_backend.reset()

    def can_use_kv_only(self, blocks, pre):
        return super().can_use_kv_only(blocks, pre) and getattr(self.scheduler, "is_rerun", False) and not getattr(pre, "adapter_args", {}).get("is_ref_prefill", False)

    def prepare_block_input(self, phase, x, pre):
        return x

    def apply_self_attn_norm(self, phase, x, shift, scale):
        weight = 1 + scale.squeeze()
        bias = shift.squeeze()
        if hasattr(phase, "smooth_norm1_weight"):
            weight = weight * phase.smooth_norm1_weight.tensor
            bias = bias * phase.smooth_norm1_bias.tensor
        normalized = self.apply_layer_norm(phase.norm1, x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            normalized = normalized.to(self.sensitive_layer_dtype)
        normalized.mul_(weight.reshape(-1, x.shape[-1])[:1]).add_(bias.reshape(-1, x.shape[-1])[:1])
        return normalized.to(self.infer_dtype) if self.sensitive_layer_dtype != self.infer_dtype else normalized

    def project_self_attn_qkv(self, phase, normalized):
        group = getattr(phase, "self_attn_projection_group", None)
        if group is None:
            query = phase.self_attn_q.apply(normalized)
            key = phase.self_attn_k.apply(normalized)
            value = phase.self_attn_v.apply(normalized)
        else:
            query, key, value = group.apply(normalized)
        shape = (-1, self.num_heads, self.head_dim)
        query = self.apply_qk_norm(phase.self_attn_norm_q, query).view(shape)
        key = self.apply_qk_norm(phase.self_attn_norm_k, key).view(shape)
        return query, key, value.view(shape)

    def project_self_attn_kv(self, phase, normalized):
        group = getattr(phase, "self_attn_cache_kv_projection_group", None)
        if group is None:
            key = phase.self_attn_k.apply(normalized)
            value = phase.self_attn_v.apply(normalized)
        else:
            key, value = group.apply(normalized)
        shape = (-1, self.num_heads, self.head_dim)
        key = self.apply_qk_norm(phase.self_attn_norm_k, key).view(shape)
        return key, value.view(shape)

    def infer_kv_block(self, block, x, pre_infer_out, start_frame=None):
        phase = block.compute_phases[0]
        x = self.prepare_block_input(phase, x, pre_infer_out)
        shift, scale, _, _, _, _ = self.pre_process(phase.modulation, pre_infer_out.embed0)
        normalized = self.apply_self_attn_norm(phase, x, shift, scale)
        key, value = self.project_self_attn_kv(phase, normalized)
        if self.seq_parallel:
            key = all2all_seq2head(key, group=self.seq_p_group)[: self._cache_token_count]
            value = all2all_seq2head(value, group=self.seq_p_group)[: self._cache_token_count]
        if start_frame is not None:
            key = apply_dynamic_causal_rope_single(
                key, pre_infer_out.grid_sizes.tensor, pre_infer_out.freqs, start_frame, complex128_math=self.seq_parallel and phase.causal_rope.compute_dtype == torch.float64
            )
        else:
            first_frame = self.kv_cache_manager.ref_num_frames + self.scheduler.seg_index * self.num_frame_per_chunk
            key = phase.causal_rope.apply_single(key.unsqueeze(0), pre_infer_out.freqs, grid_sizes=pre_infer_out.grid_sizes.tensor, start_frame=first_frame).type_as(value)[0]
        return key, value

    def infer_cache_commit(self, x, embed0, context, grid_sizes, start_frame, *conditional_inputs, block_idx, block, pre, cache, conditional_names):
        prepared = self.make_block_pre(pre, x, embed0, context, grid_sizes, dict(zip(conditional_names, conditional_inputs)))
        key, value = self.run_kv_block(block_idx, block, x, prepared, start_frame)
        self.self_attn_backend.update(key, value, cache)
        return x

    def update_self_attn_cache(self, block_idx, block, x, pre_infer_out):
        self.block_idx = block_idx
        self.set_block_cache_limits(block_idx)
        pool = self.kv_cache_manager.self_attn_kv_cache
        update = self.get_kv_cache_update(x.shape[0])
        if not x.is_cuda:
            key, value = self.run_kv_block(block_idx, block, x, pre_infer_out)
            pool.update_kv(key, value, block_idx, update)
            return x
        cache = self.prepare_self_attn_cache(pool, block_idx, update)
        pool.upload_write_metadata()
        conditions = self.get_block_conditions(pre_infer_out)
        names = tuple(conditions)
        inputs = (x, pre_infer_out.embed0, pre_infer_out.context, pre_infer_out.grid_sizes.tensor, self.frame_position(pre_infer_out, x.device), *(conditions[name] for name in names))
        runner = partial(self.infer_cache_commit, block_idx=block_idx, block=block, pre=pre_infer_out, cache=cache, conditional_names=names)
        if self.cuda_graph_stack is None:
            output = runner(*inputs)
        else:
            output = self.cuda_graph_stack.run(
                (id(block), self.use_compile, names),
                runner,
                inputs,
                bindings=(*cache.bindings, pre_infer_out.freqs, *self.get_graph_weight_bindings((block,))),
                mutated_buffers=self.get_block_write_views(pool, block_idx, update, cache),
                variant=("kv_commit", block_idx, self.kv_cache_manager.current_step, update.attention_start, update.write_end, tuple(tuple(tensor.shape) for tensor in inputs)),
                memory_pool=self.get_cuda_graph_pool(x.device),
            )
        self.finish_self_attn_cache(pool, block_idx, update)
        return output

    def _calculate_q_k_len(self, q, k_lens):
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    def infer_with_kvcache_blocks_offload(self, blocks, x, pre_infer_out):
        """Run transformer blocks with both weight offload and KV cache support."""
        mgr = self.kv_cache_manager
        self.kv_cache_size = mgr.kv_size
        self.max_attention_size = mgr.max_attention_size
        self._kv_offload = self._ar_kv_offload
        kv_cache = mgr.self_attn_kv_cache
        num_blocks = len(blocks)
        self._cache_token_count = self.get_cache_token_count(x, pre_infer_out)
        self.offload_manager.compute_stream.wait_stream(torch_device_module.current_stream())

        for block_idx in range(num_blocks):
            self.block_idx = block_idx
            self.set_block_cache_limits(block_idx)
            if self._kv_offload:
                self._next_prefetch = None

            if self.lazy_load:
                self.offload_manager.start_prefetch_block((block_idx + 1) % num_blocks)
            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)
            if self.lazy_load:
                self.offload_manager.swap_cpu_buffers()

            self.offload_manager.prefetch_weights((block_idx + 1) % num_blocks, blocks)
            gpu_block = self.offload_manager.cuda_buffers[0]
            if AI_DEVICE == "xpu":
                x = self.infer_block_func(gpu_block, x, pre_infer_out)
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    if self.cache_update_only and block_idx == num_blocks - 1:
                        x = self.update_self_attn_cache(block_idx, gpu_block, x, pre_infer_out)
                    else:
                        x = self.infer_block_func(gpu_block, x, pre_infer_out)

            self.offload_manager.swap_blocks()

        if self._kv_offload:
            if self._weight_offload_block_compute and AI_DEVICE == "cuda":
                self.offload_manager.compute_stream.synchronize()
            else:
                comp = getattr(kv_cache, "compute_stream", None)
                if comp is not None:
                    comp.synchronize()
            kv_cache.sync_all()
        return x

    def frame_position(self, pre, device):
        first_frame = 0 if getattr(pre, "adapter_args", {}).get("is_ref_prefill", False) else self.kv_cache_manager.ref_num_frames + self.scheduler.seg_index * self.num_frame_per_chunk
        return torch.full((), first_frame, dtype=torch.int64, device=device)

    def finish_self_attn_cache(self, pool, layer_id, update):
        pool.finish_attention(layer_id, update)

    def get_block_write_views(self, pool, layer_id, update, cache):
        return pool.write_views(layer_id, update.write_start, update.write_end)

    def infer_prepared_block(self, x, embed0, context, grid_sizes, start_frame, *conditional_inputs, block_idx, block, pre, cache, cross_kv, conditional_names):
        prepared = self.make_block_pre(pre, x, embed0, context, grid_sizes, dict(zip(conditional_names, conditional_inputs)))
        return self.run_block(block_idx, block, x, prepared, cache, cross_kv, start_frame)

    def run_cached_block(self, block_idx, block, x, pre):
        """Prepare cache state outside the compiled or captured transformer block."""
        pool = self.kv_cache_manager.self_attn_kv_cache
        reference = getattr(pre, "adapter_args", {}).get("is_ref_prefill", False)
        update = self.get_kv_cache_update(x.shape[0], reference)
        cache = self.prepare_self_attn_cache(pool, block_idx, update)
        pool.upload_write_metadata()
        cross_kv = self.get_block_cross_kv(block, pre.context, block_idx)
        start_frame = self.frame_position(pre, x.device)
        conditions = self.get_block_conditions(pre)
        names = tuple(conditions)
        inputs = (x, pre.embed0, pre.context, pre.grid_sizes.tensor, start_frame, *(conditions[name] for name in names))
        runner = partial(self.infer_prepared_block, block_idx=block_idx, block=block, pre=pre, cache=cache, cross_kv=cross_kv, conditional_names=names)
        if self.cuda_graph_stack is not None and x.is_cuda:
            output = self.cuda_graph_stack.run(
                (id(block), self.use_compile, names),
                runner,
                inputs,
                bindings=(*cache.bindings, *cross_kv, pre.freqs, *self.get_graph_weight_bindings((block,))),
                mutated_buffers=self.get_block_write_views(pool, block_idx, update, cache),
                variant=("cached_block", block_idx, self.kv_cache_manager.current_step, reference, update.attention_start, update.write_end, tuple(tuple(tensor.shape) for tensor in inputs)),
                memory_pool=self.get_cuda_graph_pool(x.device),
            )
        else:
            output = runner(*inputs)
        self.finish_self_attn_cache(pool, block_idx, update)
        return output

    def infer_block_with_kvoffload(self, block, x, pre_infer_out):
        """Run a transformer block with KV cache offload.

        ``RollingKVCachePool`` uses OffloadedStaticCache-style whole-layer
        prefetch inside ``infer_self_attn_with_kvcache`` (after ring roll).
        """
        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        if self._weight_offload_block_compute:
            return self.infer_block_with_kvcache(block, x, pre_infer_out)
        comp = getattr(kv_cache, "compute_stream", None)
        if comp is not None:
            with torch_device_module.stream(comp):
                return self.infer_block_with_kvcache(block, x, pre_infer_out)
        return self.infer_block_with_kvcache(block, x, pre_infer_out)

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
        if not x.is_cuda:
            return self.run_block(self.block_idx, block, x, pre_infer_out)
        return self.run_cached_block(self.block_idx, block, x, pre_infer_out)

    def apply_qk_norm(self, norm, x):
        return norm.apply(x)

    def apply_layer_norm(self, norm, x):
        if torch.compiler.is_compiling() and isinstance(norm, LN_WEIGHT_REGISTER["torch"]):
            return layer_norm(x, norm._get_actual_weight(), norm._get_actual_bias(), norm.eps, norm.sensitive_layer_dtype != norm.infer_dtype)
        return norm.apply(x)

    def infer_block(self, block, x, pre_infer_out, self_attn_cache=None, cross_kv=None, start_frame=None):
        """Run one cached block; prepared tensor inputs keep cache planning outside compile."""
        x = self.prepare_block_input(block.compute_phases[0], x, pre_infer_out)

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )

        y_out = self.infer_self_attn_with_kvcache(
            block.compute_phases[0],
            pre_infer_out.grid_sizes.tensor,
            x,
            pre_infer_out.seq_lens,
            pre_infer_out.freqs,
            shift_msa,
            scale_msa,
            getattr(pre_infer_out, "adapter_args", {}).get("is_ref_prefill", False),
            self_attn_cache,
            start_frame,
        )

        x, attn_out = self.infer_cross_attn_with_kvcache(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
            cross_kv,
            block=block,
            conditional_dict=pre_infer_out.conditional_dict,
        )

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)

        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)
        return x

    def infer_self_attn_with_kvcache(
        self,
        phase,
        grid_sizes,
        x,
        seq_lens,
        freqs,
        shift_msa,
        scale_msa,
        is_ref_prefill=False,
        self_attn_cache=None,
        start_frame=None,
    ):
        norm1_out = self.apply_self_attn_norm(phase, x, shift_msa, scale_msa)

        q, k, v = self.project_self_attn_qkv(phase, norm1_out)

        if self_attn_cache is not None:
            padded_tokens = q.shape[0]
            if self.seq_parallel:
                q = all2all_seq2head(q, group=self.seq_p_group)
                k = all2all_seq2head(k, group=self.seq_p_group)
                v = all2all_seq2head(v, group=self.seq_p_group)
                padded_tokens = q.shape[0]
                q, k, v = q[: self._cache_token_count], k[: self._cache_token_count], v[: self._cache_token_count]
            q, k = apply_dynamic_causal_rope(q, k, grid_sizes, freqs, start_frame, complex128_math=self.seq_parallel and phase.causal_rope.compute_dtype == torch.float64)
            output = self.self_attn_backend.apply(q, k, v, self_attn_cache, phase.self_attn_1)
            if self.seq_parallel:
                output = output.view(q.shape[0], -1, self.head_dim)
                output = F.pad(output, (0, 0, 0, 0, 0, padded_tokens - output.shape[0]))
                output = all2all_head2seq(output, group=self.seq_p_group).flatten(1)
            return phase.self_attn_o.apply(output)

        seg_index = int(self.scheduler.seg_index)
        is_ref_prefill = bool(is_ref_prefill)
        ref_num_frames = int(self.kv_cache_manager.ref_num_frames)
        current_start_frame = 0 if is_ref_prefill else ref_num_frames + seg_index * self.num_frame_per_chunk

        q, k = phase.causal_rope.apply(q.unsqueeze(0), k.unsqueeze(0), freqs, grid_sizes=grid_sizes, start_frame=current_start_frame)
        q, k = q.type_as(v)[0], k.type_as(v)[0]

        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        seq_parallel = self.config.get("seq_parallel", False)
        cache_update = self.get_kv_cache_update(q.shape[0], is_ref_prefill)
        if seq_parallel:
            k = all2all_seq2head(k, group=self.seq_p_group)
            v = all2all_seq2head(v, group=self.seq_p_group)
        attn_k, attn_v = kv_cache.update(k, v, self.block_idx, cache_update)

        if seq_parallel:
            attn_out = kv_cache.sp_kvcache_attn_head_shard(
                q=q,
                k_cache=attn_k,
                v_cache=attn_v,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
            )
        else:
            k_lens = torch.empty_like(seq_lens).fill_(attn_k.size(0))
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=attn_k.size(0),
            )

        y = phase.self_attn_o.apply(attn_out)

        if self._kv_offload:
            self.kv_cache_manager.self_attn_kv_cache.end_layer(
                self.block_idx,
                next_prefetch=None,
            )
        return y

    def get_block_cross_kv(self, block, context, layer_id=None):
        phase = block.compute_phases[1]
        text = context[257:] if self.has_image_context else context
        kv = self.get_cross_attn_kv(phase, text, layer_id=layer_id)
        if self.has_image_context:
            kv += self.get_cross_attn_kv(phase, context[:257], kind="image", layer_id=layer_id)
        return kv

    def prepare_cross_attn_cache(self, blocks, context):
        return tuple(self.get_block_cross_kv(block, context, layer_id) for layer_id, block in enumerate(blocks))

    def store_cross_attn_kv(self, name, layer_id, kv):
        if self.cuda_graph_stack is not None:
            # Reuse text and image projection storage across requests.
            pool = self.kv_cache_manager.get_cross_attn_kv_cache(name[0], kind=name[1])
            pool.store_kv(*kv, layer_id)
            length = kv[0].shape[0]
            kv = pool.k_cache(layer_id)[:length], pool.v_cache(layer_id)[:length]
        return super().store_cross_attn_kv(name, layer_id, kv)

    def prepare_cross_attn_input(self, block, x, conditional_dict):
        return x

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa, cross_kv=None, block=None, conditional_dict=None):
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // num_frames

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))
        x = self.prepare_cross_attn_input(block, x, conditional_dict)

        norm3_out = self.apply_layer_norm(phase.norm3, x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = self.apply_qk_norm(phase.cross_attn_norm_q, phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        if cross_kv is not None:
            k, v = cross_kv[:2]
            attn_out = phase.cross_attn_1.apply(q=q, k=k, v=v)
        else:
            k, v = self.get_cross_attn_kv(phase, context)
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=torch.tensor([k.size(0)], dtype=torch.int32))
            attn_out = phase.cross_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
            )

        if context_img is not None:
            k_img, v_img = self.get_cross_attn_kv(phase, context_img, kind="image") if cross_kv is None else cross_kv[2:]
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32),
            )
            attn_out.add_(
                phase.cross_attn_2.apply(
                    q=q,
                    k=k_img,
                    v=v_img,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=q.size(0),
                    max_seqlen_kv=k_img.size(0),
                )
            )

        attn_out = phase.cross_attn_o.apply(attn_out)

        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        x.add_(attn_out)

        num_frames = c_shift_msa.shape[0]
        frame_seqlen = x.shape[0] // c_shift_msa.shape[0]

        if hasattr(phase, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze()) * phase.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze() * phase.smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa
            norm2_bias = c_shift_msa

        norm2_out = self.apply_layer_norm(phase.norm2, x)
        norm2_out = norm2_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        norm2_out.mul_(norm2_weight).add_(norm2_bias)
        norm2_out = norm2_out.flatten(0, 1)

        hidden = phase.ffn_0.apply(norm2_out)
        ffn = getattr(phase, "ffn_gelu", None)
        y = ffn.apply(hidden) if ffn is not None else phase.ffn_2.apply(torch.nn.functional.gelu(hidden, approximate="tanh"))

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        num_frames = c_gate_msa.shape[0]
        frame_seqlen = x.shape[0] // c_gate_msa.shape[0]
        y = y.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x.add_(y * c_gate_msa)
        x = x.flatten(0, 1)

        return x

    def infer_non_blocks(self, weights, x, e):
        if self.cache_update_only:
            return None
        num_frames = e.shape[0]
        frame_seqlen = x.shape[0] // e.shape[0]

        x = weights.norm.apply(x)
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))

        t = self.scheduler.timestep_input
        e = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        modulation = weights.head_modulation.tensor
        e = (modulation.unsqueeze(1) + e).chunk(2, dim=2)

        x.mul_(1 + e[1][0]).add_(e[0][0])
        x = x.flatten(0, 1)
        x = weights.head.apply(x)

        return x
