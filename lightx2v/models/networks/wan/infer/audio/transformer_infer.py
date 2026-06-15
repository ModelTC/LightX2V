import torch
import torch.distributed as dist
from loguru import logger

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

from lightx2v.common.ops.attn.utils.all2all import all2all_seq2head
from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import align_hidden_states_and_mask, calculate_n_query_tokens, get_qk_lens_audio_range
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import apply_audio_cache_rope
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanAudioPostAdapterMixin:
    def _setup_audio_post_adapter(self, config):
        self.has_post_adapter = True
        self.phases_num = 4
        self.audio_num_tokens = int(config.get("audio_num_tokens", config.get("num_audio_tokens", 128)))

    @torch.no_grad()
    def reset_post_adapter_states(self):
        self.post_adapter_states_ready = False

    @torch.no_grad()
    def infer_post_adapter(self, phase, x, pre_infer_out):
        grid_sizes = pre_infer_out.grid_sizes.tensor
        audio_encoder_output = pre_infer_out.adapter_args["audio_encoder_output"]
        person_mask_latens = pre_infer_out.adapter_args["person_mask_latens"]
        pre_frame_tokens = grid_sizes[0][1:].prod()
        n_tokens = pre_infer_out.valid_token_len

        ori_dtype = x.dtype
        device = x.device

        if self.seq_p_group is not None:
            sp_size = dist.get_world_size(self.seq_p_group)
            sp_rank = dist.get_rank(self.seq_p_group)
        else:
            sp_size = 1
            sp_rank = 0

        if not self.post_adapter_states_ready:
            n_tokens_per_rank = torch.tensor(x.size(0), dtype=torch.int32)
            self.n_query_tokens = calculate_n_query_tokens(sp_rank, sp_size, n_tokens_per_rank, n_tokens)
            self.q_lens, self.k_lens, self.max_seqlen_q, self.max_seqlen_k, self.t0, self.t1 = get_qk_lens_audio_range(
                n_tokens_per_rank=n_tokens_per_rank,
                n_query_tokens=self.n_query_tokens,
                n_tokens_per_frame=pre_frame_tokens,
                sp_rank=sp_rank,
                num_tokens_x4=self.audio_num_tokens,
            )
            self.perceiver_attn_cu_seqlens_q = torch.cat([self.q_lens.new_zeros([1]), self.q_lens]).cumsum(0, dtype=torch.int32).to(device, non_blocking=True)
            self.perceiver_attn_cu_seqlens_k = torch.cat([self.k_lens.new_zeros([1]), self.k_lens]).cumsum(0, dtype=torch.int32).to(device, non_blocking=True)
            self.post_adapter_states_ready = True

        hidden_states_aligned, hidden_states_tail, person_mask_aligned = align_hidden_states_and_mask(self.n_query_tokens, x, person_mask_latens)
        total_residual = None
        for i in range(audio_encoder_output.shape[0]):
            audio_encoder = audio_encoder_output[i]
            audio_encoder = audio_encoder[self.t0 : self.t1].reshape(-1, audio_encoder.size(-1))
            residual = self.perceiver_attention_ca(phase, audio_encoder, hidden_states_aligned, self.scheduler.audio_adapter_t_emb)

            residual = residual.to(ori_dtype)
            if self.n_query_tokens == 0:
                residual = residual * 0.0
            if person_mask_aligned is not None:
                residual = residual * person_mask_aligned[i].unsqueeze(-1)

            if total_residual is None:
                total_residual = residual
            else:
                total_residual += residual

        x = torch.cat([hidden_states_aligned + total_residual, hidden_states_tail], dim=0)
        return x

    @torch.no_grad()
    def perceiver_attention_ca(self, phase, audio_encoder_output, latents, t_emb):
        audio_encoder_output = phase.norm_kv.apply(audio_encoder_output)
        shift, scale, gate = (t_emb + phase.shift_scale_gate.tensor)[0].chunk(3, dim=0)
        norm_q = phase.norm_q.apply(latents)
        latents = norm_q * (1 + scale) + shift
        q = phase.to_q.apply(latents)
        k, v = phase.to_kv.apply(audio_encoder_output).chunk(2, dim=-1)

        q = q.view(q.size(0), self.num_heads, self.head_dim)
        k = k.view(k.size(0), self.num_heads, self.head_dim)
        v = v.view(v.size(0), self.num_heads, self.head_dim)

        if "npu" in AI_DEVICE:
            out = ATTN_WEIGHT_REGISTER.get("npu_flash_attn")().apply(
                q=q, k=k, v=v, cu_seqlens_q=self.perceiver_attn_cu_seqlens_q, cu_seqlens_kv=self.perceiver_attn_cu_seqlens_k, max_seqlen_q=self.max_seqlen_q, max_seqlen_kv=self.max_seqlen_k
            )
        else:
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.perceiver_attn_cu_seqlens_q,
                cu_seqlens_k=self.perceiver_attn_cu_seqlens_k,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size=(-1, -1),
                deterministic=False,
            )
        out = out.view(-1, self.num_heads * self.head_dim)
        return phase.to_out.apply(out) * gate


class WanAudioTransformerInfer(WanAudioPostAdapterMixin, WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self._setup_audio_post_adapter(config)


class WanAudioARTransformerInfer(WanAudioPostAdapterMixin, WanSFTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self._setup_audio_post_adapter(config)
        self._audio_grid_meta_cache = {}

    def _audio_grid_meta(self, grid_sizes):
        key = (grid_sizes.data_ptr(), int(self.scheduler.seg_index), int(self.scheduler.step_index))
        meta = self._audio_grid_meta_cache.get(key)
        if meta is not None:
            return meta

        frames, h, w = [int(v) for v in grid_sizes[0].tolist()]
        if self.config.get("seq_parallel", False):
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
        else:
            world_size = 1
            rank = 0
        meta = (frames, h, w, world_size, rank)
        self._audio_grid_meta_cache.clear()
        self._audio_grid_meta_cache[key] = meta
        return meta

    def _spatial_freqs_for_rank(self, freqs, h, w, local_per_frame, world_size=1, rank=0):
        c = self.head_dim // 2
        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        spatial_freqs = torch.cat(
            [
                freqs_split[1][:h].view(h, 1, -1).expand(h, w, -1),
                freqs_split[2][:w].view(1, w, -1).expand(h, w, -1),
            ],
            dim=-1,
        ).reshape(h * w, -1)
        if not self.config.get("seq_parallel", False):
            return spatial_freqs

        padding_size = (world_size - (spatial_freqs.size(0) % world_size)) % world_size
        if padding_size > 0:
            pad = torch.ones(padding_size, spatial_freqs.size(1), dtype=spatial_freqs.dtype, device=spatial_freqs.device)
            spatial_freqs = torch.cat([spatial_freqs, pad], dim=0)
        return torch.chunk(spatial_freqs, world_size, dim=0)[rank][:local_per_frame]

    def _rope_freqs_for_cache_range(self, freqs, h, w, world_size, rank, token_start, token_end, ref_tokens, local_per_frame):
        c = self.head_dim // 2
        temporal_dim = c - 2 * (c // 3)
        freqs_split = freqs.split([temporal_dim, c // 3, c // 3], dim=1)
        spatial_freqs = self._spatial_freqs_for_rank(freqs, h, w, local_per_frame, world_size, rank)

        token_idx = torch.arange(token_start, token_end, device=freqs.device, dtype=torch.long)
        is_ref = token_idx < ref_tokens
        gen_idx = torch.clamp(token_idx - ref_tokens, min=0)
        frame_idx = gen_idx // local_per_frame
        ref_spatial_idx = token_idx % local_per_frame
        gen_spatial_idx = gen_idx % local_per_frame
        spatial_idx = torch.where(is_ref, ref_spatial_idx, gen_spatial_idx)

        temporal_freqs = freqs_split[0][frame_idx]
        temporal_freqs = torch.where(is_ref.unsqueeze(-1), torch.ones_like(temporal_freqs), temporal_freqs)
        return torch.cat([temporal_freqs, spatial_freqs[spatial_idx]], dim=-1).unsqueeze(1)

    def _apply_rope_with_cache_range(self, x, freqs, h, w, world_size, rank, token_start, token_end, ref_tokens, local_per_frame):
        orig_dtype = x.dtype
        if self.config.get("causal_rope_type", "triton") == "triton":
            return apply_audio_cache_rope(
                x,
                freqs,
                h=h,
                w=w,
                token_start=token_start,
                ref_tokens=ref_tokens,
                local_per_frame=local_per_frame,
                world_size=world_size,
                rank=rank,
            ).to(orig_dtype)
        pos_freqs = self._rope_freqs_for_cache_range(freqs, h, w, world_size, rank, token_start, token_end, ref_tokens, local_per_frame)
        n = x.size(1)
        x_c = torch.view_as_complex(x.float().reshape(x.size(0), n, -1, 2))
        out = torch.view_as_real(x_c * pos_freqs.to(torch.complex64)).flatten(2)
        return out.to(orig_dtype)

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj"):
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn_with_kvcache(
            block.compute_phases[0], pre_infer_out.grid_sizes.tensor, x, pre_infer_out.seq_lens, pre_infer_out.freqs, shift_msa, scale_msa, pre_infer_out.adapter_args.get("is_ref_prefill", False)
        )
        x, attn_out = self.infer_cross_attn_with_kvcache(block.compute_phases[1], x, pre_infer_out.context, y_out, gate_msa)
        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

        if pre_infer_out.adapter_args.get("audio_encoder_output") is None:
            return x
        return self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)

    def infer_self_attn_with_kvcache(self, phase, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa, is_ref_prefill):
        norm1_weight = 1 + scale_msa.squeeze()
        norm1_bias = shift_msa.squeeze()
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = norm1_weight * phase.smooth_norm1_weight.tensor
            norm1_bias = norm1_bias * phase.smooth_norm1_bias.tensor
        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        if norm1_weight.dim() == 2:
            norm1_weight = norm1_weight[0:1, :]
            norm1_bias = norm1_bias[0:1, :]
        norm1_out.mul_(norm1_weight).add_(norm1_bias)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        seq_parallel = self.config.get("seq_parallel", False)
        num_new = int(q.size(0))
        frames, h, w, sp_world_size, sp_rank = self._audio_grid_meta(grid_sizes)
        replicated_ref_prefill = bool(seq_parallel and is_ref_prefill)
        local_ref_tokens = (
            self.kv_cache_manager.ref_tokens_global if replicated_ref_prefill else self.kv_cache_manager.ref_tokens_global // sp_world_size if seq_parallel else self.kv_cache_manager.ref_tokens
        )
        cache_ref_tokens = self.kv_cache_manager.ref_tokens
        segment_idx = self.scheduler.seg_index
        local_current_start = 0 if is_ref_prefill else local_ref_tokens + segment_idx * num_new
        local_current_end = local_current_start + num_new
        cache_num_new = num_new if replicated_ref_prefill else num_new * sp_world_size if seq_parallel else num_new
        current_start = 0 if is_ref_prefill else cache_ref_tokens + segment_idx * cache_num_new
        current_end = current_start + cache_num_new
        global_end = kv_cache.get_global_end(self.block_idx)
        local_end = kv_cache.get_local_end(self.block_idx)
        local_per_frame = num_new // frames if frames > 0 else 0
        cache_per_frame = local_per_frame if replicated_ref_prefill else local_per_frame * sp_world_size if seq_parallel else local_per_frame
        sink_tokens = self.kv_cache_manager.sink_size * cache_per_frame

        need_roll = self.kv_cache_manager.local_attn_size != -1 and current_end > global_end and cache_num_new + local_end > self.kv_cache_size
        if need_roll:
            num_evicted = cache_num_new + local_end - self.kv_cache_size
            local_end_after_roll = local_end - num_evicted
        else:
            num_evicted = 0
            local_end_after_roll = local_end

        local_end_idx = local_end_after_roll + current_end - global_end
        local_start_idx = local_end_idx - cache_num_new
        attn_start = max(0, local_end_idx - self.max_attention_size)

        # Ring-buffer KV caches roll by metadata only. Do this before the
        # offload H2D materialization so the CPU physical ring is interpreted
        # in the post-roll logical order.
        if need_roll:
            kv_cache.roll_window(self.block_idx, sink_tokens, num_evicted)

        if self._kv_offload:
            kv_cache.begin_layer(self.block_idx)

        if seq_parallel:
            if replicated_ref_prefill:
                q_rope = self._apply_rope_with_cache_range(q, freqs, h, w, 1, 0, local_current_start, local_current_end, local_ref_tokens, local_per_frame)
                k_rope = self._apply_rope_with_cache_range(k, freqs, h, w, 1, 0, local_current_start, local_current_end, local_ref_tokens, local_per_frame)
                shard_heads = self.num_heads // sp_world_size
                h0 = sp_rank * shard_heads
                h1 = h0 + shard_heads
                kv_cache.store_kv(k_rope[:, h0:h1], v[:, h0:h1], local_start_idx, local_end_idx, self.block_idx)
            else:
                start_frame = segment_idx * frames
                q_rope, k_rope = self._apply_rope_sp(q, k, grid_sizes, freqs, start_frame)
                use_fp8_comm = self.config["parallel"].get("seq_p_fp8_comm", False)
                use_fp4_comm = self.config["parallel"].get("seq_p_fp4_comm", False)
                k_to_store = all2all_seq2head(
                    k_rope,
                    group=self.seq_p_group,
                    use_fp8_comm=use_fp8_comm,
                    use_fp4_comm=use_fp4_comm,
                )
                v_to_store = all2all_seq2head(
                    v,
                    group=self.seq_p_group,
                    use_fp8_comm=use_fp8_comm,
                    use_fp4_comm=use_fp4_comm,
                )
                kv_cache.store_kv(k_to_store, v_to_store, local_start_idx, local_end_idx, self.block_idx)
        else:
            kv_cache.store_kv(k, v, local_start_idx, local_end_idx, self.block_idx)
        kv_cache.set_ends(self.block_idx, current_end, local_end_idx)

        if self.clean_cuda_cache:
            del norm1_out, norm1_weight, norm1_bias
            torch_device_module.empty_cache()

        if seq_parallel:
            if replicated_ref_prefill:
                cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q_rope, k_lens=torch.empty_like(seq_lens).fill_(k_rope.size(0)))
                attn_out = phase.self_attn_1.apply(
                    q=q_rope,
                    k=k_rope,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=q_rope.size(0),
                    max_seqlen_kv=k_rope.size(0),
                )
            else:
                attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
                attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)
                attn_out = kv_cache.sp_kvcache_attn_head_shard(
                    q=q_rope,
                    k_cache=attn_k,
                    v_cache=attn_v,
                    attention_module=phase.self_attn_1,
                    seq_p_group=self.seq_p_group,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )
        else:
            attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
            attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)
            if self.config.get("ar_config", {}).get("kv_quant", {}).get("calibrate", False):
                kv_cache.capture_attn(self.block_idx, attn_start, local_end_idx)
            q = self._apply_rope_with_cache_range(q, freqs, h, w, sp_world_size, sp_rank, local_start_idx, local_end_idx, local_ref_tokens, local_per_frame)
            attn_k = self._apply_rope_with_cache_range(attn_k, freqs, h, w, sp_world_size, sp_rank, attn_start, local_end_idx, local_ref_tokens, local_per_frame)
            if isinstance(attn_k, tuple):
                k_lens = torch.empty_like(seq_lens).fill_(attn_k[0].size(0))
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
                max_seqlen_kv=attn_k.size(0) if not isinstance(attn_k, tuple) else attn_k[0].size(0),
            )

        y = phase.self_attn_o.apply(attn_out)
        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch_device_module.empty_cache()
        if self._kv_offload:
            self.kv_cache_manager.self_attn_kv_cache.end_layer(
                self.block_idx,
                next_prefetch=None,
            )
        return y
