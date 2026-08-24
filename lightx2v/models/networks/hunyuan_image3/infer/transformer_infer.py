import weakref

import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.ops.attn import *  # noqa: F403,F401 - registers LightX2V attention kernels
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.ops.norm.triton_ops import fused_qk_rms_norm, rms_norm_kernel
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_mlp, apply_rotary_pos_emb, first_weight_device, repeat_kv, to_device
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER


class HunyuanImage3TransformerInfer(BaseTransformerInfer):
    ATTENTION_IMPL_ALIASES = {
        "eager": "torch_sdpa",
        "sdpa": "torch_sdpa",
        "torch_sdpa": "torch_sdpa",
        "flash_attention_2": "flash_attn2",
        "flash_attn2": "flash_attn2",
        "flash_attention_3": "flash_attn3",
        "flash_attn3": "flash_attn3",
        "sage_attn2": "sage_attn2",
        "sage_attn3": "sage_attn3",
    }

    def __init__(self, config):
        self.config = config
        self.parallel_context = config.get("parallel_context")
        self.num_layers = int(config.get("num_layers") or config["num_hidden_layers"])
        self.hidden_size = config["hidden_size"]
        self.global_num_heads = int(config.get("num_attention_heads") or config["num_heads"])
        self.global_num_key_value_heads = int(config.get("num_key_value_heads") or self.global_num_heads)
        self.num_key_value_groups = self.global_num_heads // self.global_num_key_value_heads
        self.head_dim = config.get("attention_head_dim", self.hidden_size // self.global_num_heads)
        if config.get("tensor_parallel", False):
            self.tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
            self.tp_rank = dist.get_rank(self.tp_group)
            self.tp_size = dist.get_world_size(self.tp_group)
        else:
            self.tp_group = None
            self.tp_rank = 0
            self.tp_size = 1
        self.num_heads = self.global_num_heads // self.tp_size
        self.num_key_value_heads = self.global_num_key_value_heads // self.tp_size
        self.hidden_act = config.get("hidden_act", "silu")
        self.attn_impl = self._normalize_attention_impl(config.get("attn_impl", "torch_sdpa"))
        self.denoise_attn_impl = self._normalize_attention_impl(config.get("denoise_attn_impl", self.attn_impl))
        self._attn_kernels = {impl: None if impl == "torch_sdpa" else self._build_attention_kernel(impl) for impl in {self.attn_impl, self.denoise_attn_impl}}
        decode_impl = config.get("ar_decode_attn_impl")
        self.ar_decode_attn_kernel = self._build_attention_kernel(decode_impl) if decode_impl else None
        self._attn_cu_seqlens_cache = {}
        self._attn_segment_specs_cache = {}
        self._attn_fallback_warnings = set()
        self._sp_gather_buffers = {}
        self._pre_infer_device_cache = {}
        self.ar_decode_use_triton_rms_norm = bool(config.get("ar_decode_use_triton_rms_norm", False))
        self.ar_decode_use_fused_qk_rms_norm = bool(config.get("ar_decode_use_fused_qk_rms_norm", False))
        self.ar_decode_use_compact_moe_router = bool(config.get("ar_decode_use_compact_moe_router", False))
        self.ar_decode_overlap_shared_expert = bool(config.get("ar_decode_overlap_shared_expert", False))
        self._ar_shared_expert_streams = {}
        if config.get("seq_parallel", False):
            self.seq_p_group = config.get("device_mesh").get_group(mesh_dim="seq_p")
            self.sequence_parallel_attn_type = str(config["parallel"].get("seq_p_attn_type", "kv_all_gather")).strip().lower().replace("-", "_")
            if self.sequence_parallel_attn_type in ("kv_allgather", "kv_gather"):
                self.sequence_parallel_attn_type = "kv_all_gather"
            elif self.sequence_parallel_attn_type == "ulysses_sp":
                self.sequence_parallel_attn_type = "ulysses"
        else:
            self.seq_p_group = None
            self.sequence_parallel_attn_type = None

    def _active_phase(self):
        return self.parallel_context.phase if self.parallel_context is not None else "legacy"

    def _active_attention_impl(self):
        if self._active_phase() == "denoise":
            return self.denoise_attn_impl
        return self.attn_impl

    def _active_attention_kernel(self, attn_impl):
        return self._attn_kernels[attn_impl]

    def _active_tp_state(self):
        if self.parallel_context is not None:
            return (
                self.parallel_context.active_tp_group,
                self.parallel_context.active_tp_rank,
                self.parallel_context.active_tp_size,
                self.parallel_context.logical_tp_rank,
            )
        return self.tp_group, self.tp_rank, self.tp_size, self.tp_rank

    def _active_seq_group(self):
        if self.parallel_context is not None:
            return self.parallel_context.active_seq_group
        return self.seq_p_group

    def _active_seq_size(self):
        if self.parallel_context is not None:
            return self.parallel_context.active_seq_size
        return dist.get_world_size(self.seq_p_group) if self.seq_p_group is not None else 1

    def _active_seq_parallel(self):
        if self.parallel_context is not None:
            return self.parallel_context.active_seq_parallel
        return self.seq_p_group is not None

    def _is_single_token_ar_decode(self, tensor):
        return self._active_phase() == "ar" and tensor.ndim >= 2 and tensor.shape[-2] == 1

    @staticmethod
    def _rms_weight(norm):
        if not getattr(norm, "has_diff", False) and not getattr(norm, "has_lora_branch", False):
            return norm.weight
        return norm._get_actual_weight()

    def _apply_block_rms_norm(self, norm, hidden_states):
        if self.ar_decode_use_triton_rms_norm and self._is_single_token_ar_decode(hidden_states) and hidden_states.is_cuda:
            weight = self._rms_weight(norm)
            if weight is not None:
                return rms_norm_kernel(
                    hidden_states,
                    weight,
                    norm.eps,
                    match_torch_rms_cast=True,
                )
        return norm.apply(hidden_states)

    def _apply_attention_qk_norm(self, phase, query_states, key_states):
        norm_q = getattr(phase, "query_layernorm", None)
        norm_k = getattr(phase, "key_layernorm", None)
        if norm_q is None:
            return query_states, key_states
        if (
            norm_k is not None
            and self.ar_decode_use_fused_qk_rms_norm
            and self._is_single_token_ar_decode(query_states)
            and query_states.is_cuda
            and key_states.is_cuda
            and query_states.shape[-1] == key_states.shape[-1]
            and norm_q.eps == norm_k.eps
        ):
            q_shape = query_states.shape
            k_shape = key_states.shape
            head_dim = q_shape[-1]
            query_states, key_states = fused_qk_rms_norm(
                query_states.reshape(-1, head_dim),
                key_states.reshape(-1, head_dim),
                self._rms_weight(norm_q),
                self._rms_weight(norm_k),
                norm_q.eps,
                match_torch_rms_cast=True,
            )
            return query_states.reshape(q_shape), key_states.reshape(k_shape)
        query_states = norm_q.apply(query_states)
        if norm_k is not None:
            key_states = norm_k.apply(key_states)
        return query_states, key_states

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    @classmethod
    def _normalize_attention_impl(cls, attn_impl):
        attn_impl = str(attn_impl or "torch_sdpa")
        if attn_impl not in cls.ATTENTION_IMPL_ALIASES:
            supported = ", ".join(sorted(cls.ATTENTION_IMPL_ALIASES))
            raise ValueError(f"Unsupported HunyuanImage3 attn_impl={attn_impl!r}. Supported values: {supported}.")
        return cls.ATTENTION_IMPL_ALIASES[attn_impl]

    def _build_attention_kernel(self, attn_impl):
        if attn_impl == "flash_attn2":
            from lightx2v.common.ops.attn.flash_attn import flash_attn_func_v2, flash_attn_varlen_func_v2

            if flash_attn_func_v2 is None or flash_attn_varlen_func_v2 is None:
                raise ImportError("HunyuanImage3 attn_impl='flash_attn2' requires flash-attn v2.")
        elif attn_impl == "flash_attn3":
            from lightx2v.common.ops.attn.flash_attn import flash_attn_func_v3, flash_attn_varlen_func_v3

            if flash_attn_func_v3 is None or flash_attn_varlen_func_v3 is None:
                raise ImportError("HunyuanImage3 attn_impl='flash_attn3' requires flash-attn v3 / flash_attn_interface.")
        elif attn_impl == "flash_attn3_paged":
            from lightx2v.common.ops.attn.paged_flash_attn import require_paged_flash_attn3

            require_paged_flash_attn3()
        elif attn_impl == "sage_attn2":
            from lightx2v.common.ops.attn.sage_attn import sageattn

            if sageattn is None:
                raise ImportError("HunyuanImage3 attn_impl='sage_attn2' requires sageattention.")
        elif attn_impl == "sage_attn3":
            from lightx2v.common.ops.attn.sage_attn import sageattn3_blackwell

            if sageattn3_blackwell is None:
                raise ImportError("HunyuanImage3 attn_impl='sage_attn3' requires sageattention3.")
        if attn_impl not in ATTN_WEIGHT_REGISTER:
            raise ValueError(f"HunyuanImage3 attn_impl={attn_impl!r} is not registered in LightX2V ATTN_WEIGHT_REGISTER.")
        return ATTN_WEIGHT_REGISTER[attn_impl]()

    def _normalize_attention_dtype(self, tensor):
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor
        if tensor.device.type == "cuda":
            return tensor.to(torch.bfloat16)
        return tensor.to(torch.float32)

    def _get_cu_seqlens(self, name, batch, seq_len, device, attn_impl):
        key = (attn_impl, name, batch, seq_len, device.type, device.index)
        cu_seqlens = self._attn_cu_seqlens_cache.get(key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, batch * seq_len + 1, seq_len, dtype=torch.int32)
            if attn_impl in ("flash_attn2", "flash_attn3"):
                cu_seqlens = cu_seqlens.to(device, non_blocking=True)
            self._attn_cu_seqlens_cache[key] = cu_seqlens
        return cu_seqlens

    def _attention_mask_mode(self, attention_mask, q_len, kv_len):
        if attention_mask is None:
            return "none"
        if attention_mask.dtype != torch.bool or attention_mask.dim() != 4:
            return "custom"
        if attention_mask.shape[-2] != q_len or attention_mask.shape[-1] != kv_len:
            return "custom"
        if attention_mask.shape[1] != 1:
            return "custom"

        mask = attention_mask[:, 0]
        if torch.all(mask):
            return "full"
        if q_len == kv_len:
            causal_mask = torch.ones((q_len, kv_len), device=attention_mask.device, dtype=torch.bool).tril()
            if torch.equal(mask, causal_mask.expand_as(mask)):
                return "causal"
        return "custom"

    def _warn_attention_fallback_once(self, attn_impl, mask_mode):
        key = (attn_impl, mask_mode)
        if key in self._attn_fallback_warnings:
            return
        self._attn_fallback_warnings.add(key)
        logger.warning(
            "HunyuanImage3 attn_impl='{}' does not support {} attention masks in the low-intrusion path; falling back to PyTorch SDPA for this attention call.",
            attn_impl,
            mask_mode,
        )

    def _sdpa_attention(self, query_states, key_states, value_states, attention_mask):
        if query_states.shape[1] != key_states.shape[1]:
            repeat_groups = query_states.shape[1] // key_states.shape[1]
            key_states = repeat_kv(key_states, repeat_groups)
            value_states = repeat_kv(value_states, repeat_groups)
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0)

    def _apply_registered_attention_kernel_bshd(self, q, k, v, causal, attn_impl):
        batch, q_len, query_heads, _ = q.shape
        kv_len = k.shape[1]
        attn_kernel = self._active_attention_kernel(attn_impl)
        cu_seqlens_q = self._get_cu_seqlens("q", batch, q_len, q.device, attn_impl)
        cu_seqlens_kv = self._get_cu_seqlens("kv", batch, kv_len, k.device, attn_impl)
        attn_output = attn_kernel.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=q_len,
            max_seqlen_kv=kv_len,
            causal=causal,
        )
        return attn_output.reshape(batch, q_len, query_heads, self.head_dim)

    def _apply_registered_attention_kernel(self, query_states, key_states, value_states, causal, attn_impl):
        original_dtype = query_states.dtype
        q = self._normalize_attention_dtype(query_states.transpose(1, 2)).contiguous()
        k = self._normalize_attention_dtype(key_states.transpose(1, 2)).contiguous()
        v = self._normalize_attention_dtype(value_states.transpose(1, 2)).contiguous()
        attn_output = self._apply_registered_attention_kernel_bshd(q, k, v, causal, attn_impl)
        return attn_output.to(original_dtype).transpose(1, 2)

    @staticmethod
    def _find_full_attn_slice(full_slices, position):
        for start, stop in full_slices:
            if start <= position < stop:
                return start, stop
        return None

    def _build_segment_specs(self, position_ids, full_slices, kv_len):
        positions = [int(position) for position in position_ids.detach().cpu().reshape(-1).tolist()]
        segments = []
        local_start = 0
        while local_start < len(positions):
            pos = positions[local_start]
            full_slice = self._find_full_attn_slice(full_slices, pos)
            causal = full_slice is None
            kv_end = pos + 1 if causal else full_slice[1]
            if kv_end <= 0 or kv_end > kv_len:
                return None

            local_end = local_start + 1
            previous_pos = pos
            while local_end < len(positions):
                next_pos = positions[local_end]
                next_full_slice = self._find_full_attn_slice(full_slices, next_pos)
                next_causal = next_full_slice is None
                if next_causal != causal or next_pos != previous_pos + 1:
                    break
                if causal:
                    next_kv_end = next_pos + 1
                else:
                    if next_full_slice != full_slice:
                        break
                    next_kv_end = full_slice[1]
                if next_kv_end <= 0 or next_kv_end > kv_len:
                    return None
                previous_pos = next_pos
                kv_end = next_kv_end
                local_end += 1

            segments.append((local_start, local_end, kv_end, causal))
            local_start = local_end
        return segments

    def _segment_specs_cache_key(self, position_ids, batch_full_slices, kv_len):
        return (
            id(position_ids),
            tuple(position_ids.shape),
            tuple(position_ids.stride()),
            int(position_ids.storage_offset()),
            int(position_ids._version),
            int(kv_len),
            tuple(tuple(sample_slices) for sample_slices in batch_full_slices),
        )

    def _lookup_segment_specs_cache(self, key, position_ids):
        entry = self._attn_segment_specs_cache.get(key)
        if entry is None:
            return None
        tensor_ref, segment_specs = entry
        if tensor_ref() is position_ids:
            return segment_specs
        self._attn_segment_specs_cache.pop(key, None)
        return None

    def _store_segment_specs_cache(self, key, position_ids, segment_specs):
        if len(self._attn_segment_specs_cache) >= 32:
            stale_keys = [cache_key for cache_key, (tensor_ref, _) in self._attn_segment_specs_cache.items() if tensor_ref() is None]
            for cache_key in stale_keys:
                self._attn_segment_specs_cache.pop(cache_key, None)
            if len(self._attn_segment_specs_cache) >= 32:
                self._attn_segment_specs_cache.pop(next(iter(self._attn_segment_specs_cache)))
        self._attn_segment_specs_cache[key] = (weakref.ref(position_ids), segment_specs)

    def _segmented_flash_attention(
        self,
        query_states,
        key_states,
        value_states,
        position_ids,
        full_attn_slices,
        *,
        attn_impl,
        segment_specs=None,
    ):
        if position_ids is None:
            return None
        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        if position_ids.shape != (batch, q_len):
            return None

        if segment_specs is None:
            if full_attn_slices is None:
                return None
            batch_full_slices = full_attn_slices
            if not any(batch_full_slices):
                return None
            segment_specs = [self._build_segment_specs(position_ids[batch_idx], batch_full_slices[batch_idx], kv_len) for batch_idx in range(batch)]
        if len(segment_specs) != batch or any(specs is None for specs in segment_specs):
            return None

        original_dtype = query_states.dtype
        q = self._normalize_attention_dtype(query_states.transpose(1, 2)).contiguous()
        k = self._normalize_attention_dtype(key_states.transpose(1, 2)).contiguous()
        v = self._normalize_attention_dtype(value_states.transpose(1, 2)).contiguous()
        output = torch.empty_like(q)
        for batch_idx in range(batch):
            for q_start, q_stop, kv_stop, causal in segment_specs[batch_idx]:
                segment_output = self._apply_registered_attention_kernel_bshd(
                    q[batch_idx : batch_idx + 1, q_start:q_stop],
                    k[batch_idx : batch_idx + 1, :kv_stop],
                    v[batch_idx : batch_idx + 1, :kv_stop],
                    causal=causal,
                    attn_impl=attn_impl,
                )
                output[batch_idx : batch_idx + 1, q_start:q_stop] = segment_output
        return output.to(original_dtype).transpose(1, 2)

    def _registered_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        position_ids=None,
        full_attn_slices=None,
        segment_specs=None,
    ):
        attn_impl = self._active_attention_impl()
        if attn_impl in ("flash_attn2", "flash_attn3") and segment_specs is not None:
            segmented_output = self._segmented_flash_attention(
                query_states,
                key_states,
                value_states,
                position_ids,
                full_attn_slices,
                segment_specs=segment_specs,
                attn_impl=attn_impl,
            )
            if segmented_output is not None:
                return segmented_output
            raise RuntimeError("HunyuanImage3 precomputed segmented attention plan does not match the current Q/KV layout.")

        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        mask_mode = self._attention_mask_mode(attention_mask, q_len, kv_len)

        if attn_impl == "torch_sdpa":
            return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
        if attn_impl in ("flash_attn2", "flash_attn3"):
            if mask_mode not in ("none", "full", "causal"):
                segmented_output = self._segmented_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    position_ids,
                    full_attn_slices,
                    segment_specs=segment_specs,
                    attn_impl=attn_impl,
                )
                if segmented_output is not None:
                    return segmented_output
                self._warn_attention_fallback_once(attn_impl, mask_mode)
                return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
            causal = mask_mode == "causal"
        elif attn_impl in ("sage_attn2", "sage_attn3"):
            if mask_mode not in ("none", "full"):
                self._warn_attention_fallback_once(attn_impl, mask_mode)
                return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
            causal = False
        else:
            raise ValueError(f"Unsupported HunyuanImage3 normalized attn_impl={attn_impl!r}.")

        return self._apply_registered_attention_kernel(
            query_states,
            key_states,
            value_states,
            causal=causal,
            attn_impl=attn_impl,
        )

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self._pre_infer_device_cache = {}
        pre_infer_out.attention_segment_specs = self._prepare_attention_segment_specs(pre_infer_out)
        hidden_states = pre_infer_out.hidden_states
        for block_idx, block in enumerate(weights.blocks):
            hidden_states = self.infer_block(block_idx, block, hidden_states, pre_infer_out)
        return hidden_states

    def infer_block(self, block_idx, block, hidden_states, pre_infer_out):
        attention_phase = block.compute_phases[0]
        mlp_phase = block.compute_phases[1]
        device = first_weight_device(attention_phase)
        if device is not None and device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device.index)
        hidden_states = to_device(hidden_states, device)
        use_segment_specs = self._active_attention_impl() in ("flash_attn2", "flash_attn3") and pre_infer_out.attention_segment_specs is not None
        attention_mask = None if use_segment_specs else self._cached_pre_infer_to_device("attention_mask", pre_infer_out.attention_mask, device)
        position_ids = self._cached_pre_infer_to_device("position_ids", pre_infer_out.position_ids, device)
        custom_pos_emb = self._cached_pre_infer_to_device("custom_pos_emb", pre_infer_out.custom_pos_emb, device)

        residual = hidden_states
        normed = self._apply_block_rms_norm(attention_phase.input_layernorm, hidden_states)
        attn_out = self.infer_attention(
            block_idx,
            attention_phase,
            normed,
            attention_mask,
            position_ids,
            custom_pos_emb,
            pre_infer_out.full_attn_slices,
            pre_infer_out.past_key_values if pre_infer_out.use_cache else None,
            pre_infer_out.sequence_parallel_state,
            pre_infer_out.attention_segment_specs,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        normed = self._apply_block_rms_norm(mlp_phase.post_attention_layernorm, hidden_states)
        mlp_out = self.infer_mlp(mlp_phase, normed)
        return residual + mlp_out

    def infer_attention(
        self,
        block_idx,
        phase,
        hidden_states,
        attention_mask,
        position_ids,
        custom_pos_emb,
        full_attn_slices=None,
        past_key_values=None,
        sequence_parallel_state=None,
        segment_specs=None,
    ):
        batch, q_len, _ = hidden_states.shape
        attn_impl = self._active_attention_impl()
        _, _, active_tp_size, _ = self._active_tp_state()
        if self.global_num_heads % active_tp_size or self.global_num_key_value_heads % active_tp_size:
            raise ValueError(f"HunyuanImage3 active TP size must divide Q and KV heads: Q={self.global_num_heads}, KV={self.global_num_key_value_heads}, active_tp_size={active_tp_size}.")
        num_heads = self.global_num_heads // active_tp_size
        num_key_value_heads = self.global_num_key_value_heads // active_tp_size
        qkv_states = apply_linear(phase.qkv_proj, hidden_states.reshape(-1, hidden_states.shape[-1]))
        qkv_states = qkv_states.reshape(
            batch,
            q_len,
            num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)
        query_states = query_states.reshape(batch, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        if custom_pos_emb is not None:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if getattr(phase, "query_layernorm", None) is not None:
            query_states, key_states = self._apply_attention_qk_norm(phase, query_states, key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        cache_position_ids = position_ids
        if sequence_parallel_state is not None:
            if sequence_parallel_state.attn_type == "kv_all_gather":
                key_states, value_states = self._sequence_parallel_gather_kv(
                    key_states,
                    value_states,
                    sequence_parallel_state,
                )
                cache_position_ids = self._cached_pre_infer_to_device(
                    "sp_global_position_ids",
                    sequence_parallel_state.global_position_ids,
                    key_states.device,
                )
            elif sequence_parallel_state.attn_type == "ulysses":
                query_states, key_states, value_states = self._sequence_parallel_ulysses_seq_to_head(
                    query_states,
                    key_states,
                    value_states,
                    sequence_parallel_state,
                )
                cache_position_ids = self._cached_pre_infer_to_device(
                    "sp_global_position_ids",
                    sequence_parallel_state.global_position_ids,
                    key_states.device,
                )
                position_ids = cache_position_ids
                if not (attn_impl in ("flash_attn2", "flash_attn3") and segment_specs is not None):
                    attention_mask = self._cached_pre_infer_to_device(
                        "sp_global_attention_mask",
                        sequence_parallel_state.global_attention_mask,
                        key_states.device,
                    )
            else:
                raise ValueError(f"Unsupported HunyuanImage3 sequence parallel attention type: {sequence_parallel_state.attn_type!r}.")

        paged_decode = past_key_values is not None and past_key_values.paged and past_key_values.decode_ready and q_len == 1

        if paged_decode:
            key_cache, value_cache = past_key_values.get_paged_layer(block_idx)
            query_states = query_states.to(key_cache.dtype)
            key_states = key_states.to(key_cache.dtype)
            value_states = value_states.to(value_cache.dtype)
            attn_output = self.ar_decode_attn_kernel.apply_decode(
                query_states,
                key_states,
                value_states,
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=past_key_values.page_table,
                cache_seqlens=past_key_values.cache_seqlens,
                scheduler_metadata=past_key_values.scheduler_metadata,
                max_num_splits=int(self.config.get("ar_flash_attn_max_num_splits", 32)),
            )
        elif past_key_values is not None:
            if cache_position_ids is None:
                raise ValueError("HunyuanImage3 KV cache requires position_ids.")
            key_states, value_states = past_key_values.update(key_states, value_states, block_idx, cache_position_ids)
            query_states = query_states.to(key_states.dtype)

        if not paged_decode and attn_impl not in ("flash_attn2", "flash_attn3"):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        if not paged_decode and sequence_parallel_state is not None and sequence_parallel_state.attn_type == "kv_all_gather":
            valid_q_len = sequence_parallel_state.valid_local_seq_len
            attn_output = torch.zeros_like(query_states)
            if valid_q_len:
                attn_output[:, :, :valid_q_len] = self._registered_attention(
                    query_states[:, :, :valid_q_len],
                    key_states,
                    value_states,
                    None if attention_mask is None else attention_mask[:, :, :valid_q_len],
                    position_ids=None if position_ids is None else position_ids[:, :valid_q_len],
                    full_attn_slices=full_attn_slices,
                    segment_specs=segment_specs,
                )
        elif not paged_decode:
            attn_output = self._registered_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                position_ids=position_ids,
                full_attn_slices=full_attn_slices,
                segment_specs=segment_specs,
            )

        if sequence_parallel_state is not None and sequence_parallel_state.attn_type == "ulysses":
            attn_output = self._sequence_parallel_ulysses_head_to_seq(attn_output, sequence_parallel_state)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, q_len, -1)
        attn_output = apply_linear(phase.o_proj, attn_output.reshape(-1, attn_output.shape[-1]))
        return attn_output.reshape(batch, q_len, -1)

    def _prepare_attention_segment_specs(self, pre_infer_out):
        if self._active_attention_impl() not in ("flash_attn2", "flash_attn3"):
            return None

        state = pre_infer_out.sequence_parallel_state
        if state is not None and state.attn_type == "ulysses":
            position_ids = state.global_position_ids
            attention_mask = state.global_attention_mask
        else:
            position_ids = pre_infer_out.position_ids
            attention_mask = pre_infer_out.attention_mask
            if state is not None:
                position_ids = position_ids[:, : state.valid_local_seq_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, : state.valid_local_seq_len]

        if position_ids is None or attention_mask is None:
            return None
        if pre_infer_out.full_attn_slices is None:
            # A generic custom mask has no lossless causal/full-slice description.
            # Keep the dense mask and let registered attention fall back to SDPA.
            return None
        batch, q_len = position_ids.shape
        kv_len = attention_mask.shape[-1]
        if attention_mask.dtype != torch.bool or attention_mask.dim() != 4 or attention_mask.shape[0] != batch or attention_mask.shape[1] != 1 or attention_mask.shape[-2] != q_len:
            return None
        batch_full_slices = pre_infer_out.full_attn_slices
        if not any(batch_full_slices):
            return None
        cache_key = self._segment_specs_cache_key(position_ids, batch_full_slices, kv_len)
        cached_specs = self._lookup_segment_specs_cache(cache_key, position_ids)
        if cached_specs is not None:
            return cached_specs
        segment_specs = [self._build_segment_specs(position_ids[batch_idx], batch_full_slices[batch_idx], kv_len) for batch_idx in range(batch)]
        if any(specs is None for specs in segment_specs):
            return None
        self._store_segment_specs_cache(cache_key, position_ids, segment_specs)
        return segment_specs

    def _cached_pre_infer_to_device(self, name, value, device):
        if value is None:
            return None
        key = (name, device, id(value))
        cached = self._pre_infer_device_cache.get(key)
        if cached is None:
            cached = to_device(value, device)
            self._pre_infer_device_cache[key] = cached
        return cached

    def _sequence_parallel_gather_kv(self, key_states, value_states, state):
        seq_group = self._active_seq_group()
        if seq_group is None:
            raise RuntimeError("HunyuanImage3 sequence parallel is active without an active sequence process group.")
        world_size = dist.get_world_size(seq_group)
        local = torch.stack((key_states, value_states), dim=2).permute(3, 0, 2, 1, 4).contiguous()
        output_shape = (local.shape[0] * world_size, *local.shape[1:])
        buffer_key = ("kv", self._active_phase(), id(seq_group), local.device, local.dtype, output_shape)
        gathered = self._sp_gather_buffers.get(buffer_key)
        if gathered is None or gathered.shape != output_shape:
            gathered = torch.empty(output_shape, device=local.device, dtype=local.dtype)
            self._sp_gather_buffers[buffer_key] = gathered
        dist.all_gather_into_tensor(gathered, local, group=seq_group)
        valid = gathered[: state.original_seq_len]
        key_value = valid.permute(2, 1, 3, 0, 4).contiguous()
        return key_value[0], key_value[1]

    def _sequence_parallel_ulysses_seq_to_head(self, query_states, key_states, value_states, state):
        if query_states.shape[0] != 1:
            raise ValueError("HunyuanImage3 Ulysses expects batch size 1; use parallel.cfg_mode='serial' when cfg_p_size=1 (including TP+SP), or 'parallel' for CFG+SP.")
        seq_group = self._active_seq_group()
        if seq_group is None:
            raise RuntimeError("HunyuanImage3 Ulysses is active without an active sequence process group.")
        query = all2all_seq2head(query_states[0].transpose(0, 1).contiguous(), group=seq_group)
        key = all2all_seq2head(key_states[0].transpose(0, 1).contiguous(), group=seq_group)
        value = all2all_seq2head(value_states[0].transpose(0, 1).contiguous(), group=seq_group)
        original_seq_len = state.original_seq_len
        query = query[:original_seq_len].transpose(0, 1).unsqueeze(0).contiguous()
        key = key[:original_seq_len].transpose(0, 1).unsqueeze(0).contiguous()
        value = value[:original_seq_len].transpose(0, 1).unsqueeze(0).contiguous()
        return query, key, value

    def _sequence_parallel_ulysses_head_to_seq(self, attn_output, state):
        output = attn_output[0].transpose(0, 1).contiguous()
        padding_size = state.padded_seq_len - state.original_seq_len
        if padding_size:
            padding = output.new_zeros(padding_size, output.shape[1], output.shape[2])
            output = torch.cat((output, padding), dim=0)
        seq_group = self._active_seq_group()
        if seq_group is None:
            raise RuntimeError("HunyuanImage3 Ulysses is active without an active sequence process group.")
        output = all2all_head2seq(output, group=seq_group)
        return output.unsqueeze(0).transpose(1, 2).contiguous()

    def _apply_phase_mlp(self, gate_and_up_proj, down_proj, hidden_states):
        phase_mlp = getattr(gate_and_up_proj, "apply_phase_mlp", None)
        if callable(phase_mlp):
            return phase_mlp(hidden_states, down_proj, self.hidden_act)

        gate_up_activation = getattr(gate_and_up_proj, "apply_gate_up_activation", None)
        if callable(gate_up_activation):
            original_shape = hidden_states.shape
            flat = hidden_states.reshape(-1, original_shape[-1])
            activated = gate_up_activation(flat)
            output = apply_linear(down_proj, activated)
            return output.reshape(*original_shape)

        return apply_mlp(gate_and_up_proj, down_proj, hidden_states, self.hidden_act)

    def infer_mlp(self, phase, hidden_states):
        if not phase.is_moe:
            return self._apply_phase_mlp(phase.gate_and_up_proj, phase.down_proj, hidden_states)

        moe = phase.moe
        original_dtype = hidden_states.dtype
        compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        active_phase = self._active_phase()
        if moe.moe_backend == "multi_micro" and active_phase == "denoise" and compute_dtype != torch.bfloat16:
            raise TypeError(f"HunyuanImage3 multi_micro requires BF16 inference; got compute dtype {compute_dtype}.")

        shared_mlp = getattr(moe, "shared_mlp", None)
        overlap_shared = self.ar_decode_overlap_shared_expert and shared_mlp is not None and self._is_single_token_ar_decode(hidden_states) and hidden_states.is_cuda
        shared_out = None
        shared_stream = None
        main_stream = None
        if overlap_shared:
            main_stream = torch.cuda.current_stream(hidden_states.device)
            shared_stream = self._ar_shared_expert_stream(hidden_states.device)
            shared_stream.wait_stream(main_stream)
            with torch.cuda.stream(shared_stream):
                shared_out = self._apply_phase_mlp(
                    shared_mlp.gate_and_up_proj,
                    shared_mlp.down_proj,
                    hidden_states,
                )

        flat, topk_weight, topk_idx = self._moe_topk(moe, hidden_states)
        fused_input = flat.to(dtype=compute_dtype).contiguous()
        fused_moe = moe.get_fused_moe(active_phase, fused_input.device, compute_dtype)
        output = fused_moe.apply(fused_input, topk_idx, topk_weight).reshape_as(hidden_states).to(original_dtype)

        if shared_mlp is not None:
            if overlap_shared:
                main_stream.wait_stream(shared_stream)
            else:
                shared_out = self._apply_phase_mlp(shared_mlp.gate_and_up_proj, shared_mlp.down_proj, hidden_states)
            output = output + shared_out.to(output.dtype)

        tp_group, _, tp_size, _ = self._active_tp_state()
        if tp_size > 1:
            if tp_group is None:
                raise RuntimeError("HunyuanImage3 active tensor parallelism requires an active TP process group.")
            if self.parallel_context is not None:
                output = self.parallel_context.tensor_parallel_all_reduce(output)
            else:
                dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
        return output

    def _moe_topk(self, moe, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = apply_linear(moe.gate, flat)
        if self.ar_decode_use_compact_moe_router and self._is_single_token_ar_decode(hidden_states):
            topk_logits, topk_idx = torch.topk(logits, moe.moe_topk, dim=-1)
            return flat, torch.softmax(topk_logits, dim=-1), topk_idx
        topk_weight, topk_idx = torch.topk(torch.softmax(logits, dim=-1), moe.moe_topk, dim=-1)
        topk_weight = topk_weight / torch.clamp(topk_weight.sum(dim=-1, keepdim=True), min=1e-8)
        return flat, topk_weight, topk_idx

    def _ar_shared_expert_stream(self, device):
        device = torch.device(device)
        key = (device.type, device.index)
        stream = self._ar_shared_expert_streams.get(key)
        if stream is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HunyuanImage3 shared-expert overlap stream must be initialized before CUDA Graph capture.")
            stream = torch.cuda.Stream(device=device)
            self._ar_shared_expert_streams[key] = stream
        return stream
