import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    logger.info("flash_attn_varlen_func not found, please install flash_attn2 first")
    flash_attn_varlen_func = None

from lightx2v.common.kvcache.attn.rope import PhysicalRoPERingAttention
from lightx2v.common.kvcache.attn.sage_rope import SageRoPERingAttention
from lightx2v.common.ops.attn.ulysses_attn import UlyssesAttnWeight
from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import align_hidden_states_and_mask, calculate_n_query_tokens, get_qk_lens_audio_range
from lightx2v.models.networks.wan.infer.autoregressive.causal_transformer_infer import WanCausalTransformerInfer
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


class _CachedAudioAttention:
    """Adapt the physical cache to the shared Ulysses exchange interface."""

    def __init__(self, backend, cache, attention, valid_tokens):
        self.backend = backend
        self.cache = cache
        self.attention = attention
        self.valid_tokens = valid_tokens

    def apply(self, q, k, v, **kwargs):
        tokens = self.valid_tokens
        output = self.backend.apply(q[:tokens], k[:tokens], v[:tokens], self.cache, self.attention)
        output = output.view(tokens, q.shape[1], q.shape[2])
        return F.pad(output, (0, 0, 0, 0, 0, q.shape[0] - tokens))


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
        if audio_encoder_output is None:
            return x
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
            residual = self.perceiver_attention_ca(
                phase,
                audio_encoder,
                hidden_states_aligned,
                pre_infer_out.adapter_args["audio_adapter_t_emb"] if "audio_adapter_t_emb" in pre_infer_out.adapter_args else self.scheduler.audio_adapter_t_emb,
            )

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

        # Audio adapter projections are replicated across tensor-parallel ranks.
        num_heads = q.shape[-1] // self.head_dim
        q = q.view(q.size(0), num_heads, self.head_dim)
        k = k.view(k.size(0), num_heads, self.head_dim)
        v = v.view(v.size(0), num_heads, self.head_dim)

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
        out = out.view(-1, num_heads * self.head_dim)
        return phase.to_out.apply(out) * gate


class WanAudioTransformerInfer(WanAudioPostAdapterMixin, WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self._setup_audio_post_adapter(config)


class WanAudioARTransformerInfer(WanAudioPostAdapterMixin, WanCausalTransformerInfer):
    cache_commit_enabled = False

    def __init__(self, config):
        super().__init__(config)
        self._setup_audio_post_adapter(config)
        ar_config = config.get("ar_config", {})
        if config["self_attn_1_type"] == "sage_attn2":
            self.self_attn_backend = SageRoPERingAttention(share_step_keys=ar_config.get("step_kv_cache", False))
        else:
            self.self_attn_backend = PhysicalRoPERingAttention()
        self.kv_cache_manager.key_only = isinstance(self.self_attn_backend, SageRoPERingAttention)
        if self.seq_parallel and self.seq_p_attn_type != "ulysses":
            raise ValueError("Seko AR cached attention requires seq_p_attn_type='ulysses'.")
        if self.seq_parallel and self.seq_p_head_parallel:
            raise NotImplementedError(
                "Seko AR cached attention does not support seq_p_head_parallel=True: "
                "the physical post-RoPE cache requires all local heads together. "
                "Use seq_p_head_parallel=False; Triton layout, tensor fusion and FP8 communication remain supported."
            )
        self.parallel_attention = UlyssesAttnWeight()
        self._rope_freq_buffers = {}
        self._current_pre = None
        self._post_adapter_signature = None
        self.rope_position_mode = ar_config.get("rope_position_mode", "local")
        if self.rope_position_mode not in {"local", "global"}:
            raise ValueError("ar_config.rope_position_mode must be 'local' or 'global'.")

        self.rope_max_frames = ar_config.get("rope_max_frames")
        if self.rope_max_frames is not None:
            self.rope_max_frames = int(self.rope_max_frames)
            if self.rope_max_frames <= 0:
                raise ValueError("ar_config.rope_max_frames must be positive.")
            if self.rope_position_mode != "global":
                raise ValueError("ar_config.rope_max_frames requires ar_config.rope_position_mode='global'.")
            if self.rope_max_frames < self.num_frame_per_chunk:
                raise ValueError("ar_config.rope_max_frames must be at least ar_config.num_frame_per_chunk.")

    def infer(self, weights, pre_infer_out):
        self._current_pre = pre_infer_out
        pre_infer_out.adapter_args["audio_adapter_t_emb"] = self.scheduler.audio_adapter_t_emb
        self.prepare_post_adapter(pre_infer_out)
        if not self._weight_offload_block_compute:
            self.prepare_cross_attn_cache(weights.blocks, pre_infer_out.context)
        return super().infer(weights, pre_infer_out)

    def initialize_kv_cache(self, latent_shape, ref_num_frames=None):
        self.self_attn_backend.reset(keep_storage=True)
        return super().initialize_kv_cache(latent_shape, ref_num_frames)

    def prepare_block_input(self, phase, x, pre):
        return phase.before_proj.apply(x) + pre.x if hasattr(phase, "before_proj") else x

    def get_compile_block_key(self, block_idx, block):
        return id(block) if self._weight_offload_block_compute else block_idx

    def get_block_conditions(self, pre):
        conditions = dict(super().get_block_conditions(pre))
        for name in ("audio_encoder_output", "person_mask_latens", "audio_adapter_t_emb"):
            value = pre.adapter_args.get(name)
            if value is not None:
                # Ignore strides of singleton axes inherited from the full audio clip.
                conditions[name] = value.view(value.shape)
        return conditions

    def make_block_pre(self, pre, x, embed0, context, grid_sizes, conditions):
        prepared = super().make_block_pre(pre, x, embed0, context, grid_sizes, conditions)
        prepared.adapter_args = dict(pre.adapter_args)
        for name in ("audio_encoder_output", "person_mask_latens", "audio_adapter_t_emb"):
            prepared.adapter_args[name] = conditions.get(name)
        return prepared

    def prepare_post_adapter(self, pre):
        if pre.adapter_args.get("audio_encoder_output") is None:
            return
        device = pre.x.device
        world_size = dist.get_world_size(self.seq_p_group) if self.seq_parallel else 1
        rank = dist.get_rank(self.seq_p_group) if self.seq_parallel else 0
        signature = pre.x.shape[0], pre.valid_token_len, pre.grid_sizes.tuple[1:], world_size, rank, device
        if self._post_adapter_signature == signature:
            return
        tokens_per_rank = torch.tensor(pre.x.shape[0], dtype=torch.int32)
        self.n_query_tokens = calculate_n_query_tokens(rank, world_size, tokens_per_rank, pre.valid_token_len)
        self.q_lens, self.k_lens, self.max_seqlen_q, self.max_seqlen_k, self.t0, self.t1 = get_qk_lens_audio_range(
            n_tokens_per_rank=tokens_per_rank,
            n_query_tokens=self.n_query_tokens,
            n_tokens_per_frame=torch.tensor(pre.grid_sizes.tuple[1] * pre.grid_sizes.tuple[2], dtype=torch.int32),
            sp_rank=rank,
            num_tokens_x4=self.audio_num_tokens,
        )
        self.perceiver_attn_cu_seqlens_q = torch.cat((self.q_lens.new_zeros(1), self.q_lens)).cumsum(0, dtype=torch.int32).to(device)
        self.perceiver_attn_cu_seqlens_k = torch.cat((self.k_lens.new_zeros(1), self.k_lens)).cumsum(0, dtype=torch.int32).to(device)
        self.n_query_tokens = int(self.n_query_tokens)
        self._post_adapter_signature = signature

    def reset_post_adapter_states(self):
        self.post_adapter_states_ready = True

    def prepare_attention_freqs(self, update, capacity):
        pre = self._current_pre
        _, height, width = pre.grid_sizes.tuple
        frame_tokens = height * width
        generation = self.scheduler.seg_index, update.write_end, update.global_end, height, width
        cached = self._rope_freq_buffers.get(capacity)
        if cached is None or cached[1].device != pre.x.device:
            with torch.inference_mode(False):
                cached = None, torch.empty(capacity, self.head_dim, device=pre.x.device, dtype=torch.float32)
        previous_generation, storage = cached
        frequencies = storage[: update.write_end]
        if generation != previous_generation:
            positions = torch.arange(update.write_end, device=pre.x.device)
            if self.rope_position_mode == "global":
                global_end = update.global_end
                if self.rope_max_frames is not None and not self.seq_parallel:
                    global_end = min(global_end, self.kv_cache_manager.ref_tokens + self.rope_max_frames * frame_tokens)
                positions = torch.where(positions < update.sink_tokens, positions, global_end - update.write_end + positions)
                if self.rope_max_frames is not None and self.seq_parallel:
                    # SP historically stores rotated K: retain each chunk's clamped write position.
                    ref_tokens = self.kv_cache_manager.ref_tokens
                    chunk_tokens = self.num_frame_per_chunk * frame_tokens
                    generated = positions - ref_tokens
                    chunk_start = generated.div(chunk_tokens, rounding_mode="floor") * chunk_tokens
                    max_start = (self.rope_max_frames - self.num_frame_per_chunk) * frame_tokens
                    clamped = chunk_start.clamp(max=max_start) + generated.remainder(chunk_tokens) + ref_tokens
                    positions = torch.where(positions < ref_tokens, positions, clamped)
            half = self.head_dim // 2
            temporal, spatial_h, spatial_w = pre.freqs.split((half - 2 * (half // 3), half // 3, half // 3), dim=1)
            spatial = positions % frame_tokens
            selected = torch.cat((temporal[positions // frame_tokens], spatial_h[spatial // width], spatial_w[spatial % width]), dim=-1)
            frequencies.copy_(torch.cat((selected.real, selected.imag), dim=-1))
            self._rope_freq_buffers[capacity] = generation, storage
        return frequencies

    def prepare_self_attn_cache(self, pool, layer_id, update):
        capacity = pool.get_max_cache_shape(layer_id)
        frequencies = self.prepare_attention_freqs(update, capacity)
        return self.self_attn_backend.prepare(pool, layer_id, update, freqs=frequencies)

    def prepare_block_caches(self, blocks, x, pre_infer_out=None):
        caches, bindings, writes = super().prepare_block_caches(blocks, x, pre_infer_out)
        scratch = {cache.key.data_ptr(): cache.key for cache in caches if getattr(cache, "key", None) is not None}
        return caches, bindings, (*writes, *scratch.values())

    def get_block_write_views(self, pool, layer_id, update, cache):
        scratch = getattr(cache, "key", None)
        return (*super().get_block_write_views(pool, layer_id, update, cache), *((scratch,) if scratch is not None else ()))

    def get_kv_cache_update(self, token_count, is_ref_prefill=False, *, global_token_count=None):
        if global_token_count is None:
            global_token_count = self._current_pre.valid_token_len
        return super().get_kv_cache_update(token_count, is_ref_prefill, global_token_count=global_token_count)

    def infer_self_attn_with_kvcache(self, phase, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa, is_ref_prefill=False, self_attn_cache=None, start_frame=None):
        normalized = self.apply_self_attn_norm(phase, x, shift_msa, scale_msa)
        query, key, value = self.project_self_attn_qkv(phase, normalized)
        if self.seq_parallel and not is_ref_prefill:
            # Grouped QKV projections return strided views; the shared Triton
            # layout kernels consume contiguous current-chunk inputs.
            if self.seq_p_prepost_backend == "triton":
                query, key, value = (tensor.contiguous() for tensor in (query, key, value))
            output, auxiliary = self.parallel_attention.apply(
                q=query,
                k=key,
                v=value,
                attention_module=_CachedAudioAttention(self.self_attn_backend, self_attn_cache, phase.self_attn_1, self._current_pre.valid_token_len),
                seq_p_group=self.seq_p_group,
                prepost_backend=self.seq_p_prepost_backend,
                a2a_backend=self.seq_p_a2a_backend,
                quant_scheme=self.seq_p_quant_scheme,
                tensor_fusion=self.seq_p_tensor_fusion,
                head_parallel=False,
            )
            if auxiliary is not None:
                raise RuntimeError("Seko AR self-attention has no auxiliary token output.")
            return phase.self_attn_o.apply(output)

        if self.seq_parallel:
            world_size = dist.get_world_size(self.seq_p_group)
            rank = dist.get_rank(self.seq_p_group)
            query, key, value = (tensor.chunk(world_size, dim=1)[rank].contiguous() for tensor in (query, key, value))
        output = self.self_attn_backend.apply(query, key, value, self_attn_cache, phase.self_attn_1)
        if self.seq_parallel:
            output = output.view(query.shape)
            gathered = torch.empty((world_size * output.shape[0], *output.shape[1:]), device=output.device, dtype=output.dtype)
            dist.all_gather_into_tensor(gathered, output.contiguous(), group=self.seq_p_group)
            output = gathered.view(world_size, *output.shape).permute(1, 0, 2, 3).flatten(1)
        return phase.self_attn_o.apply(output)
