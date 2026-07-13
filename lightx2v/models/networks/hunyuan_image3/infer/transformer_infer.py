import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.ops.attn import *  # noqa: F403,F401 - registers LightX2V attention kernels
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.models.networks.hunyuan_image3.infer.utils import apply_linear, apply_mlp, apply_rotary_pos_emb, first_weight_device, repeat_kv, to_device
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except Exception:
    try:
        import flashinfer

        flashinfer_cutlass_fused_moe = flashinfer.fused_moe.cutlass_fused_moe
    except Exception:
        flashinfer_cutlass_fused_moe = None


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
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads") or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.get("attention_head_dim", self.hidden_size // self.num_heads)
        self.hidden_act = config.get("hidden_act", "silu")
        self.flashinfer_tune_max_num_tokens = int(config.get("flashinfer_tune_max_num_tokens", 8192))
        self.attn_impl = self._normalize_attention_impl(config.get("attn_impl", "torch_sdpa"))
        self.attn_kernel = None if self.attn_impl == "torch_sdpa" else self._build_attention_kernel(self.attn_impl)
        self._attn_cu_seqlens_cache = {}
        self._attn_fallback_warnings = set()
        self._sp_gather_buffers = {}
        self._pre_infer_device_cache = {}
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

    def _get_cu_seqlens(self, name, batch, seq_len, device):
        key = (name, batch, seq_len, device.type, device.index)
        cu_seqlens = self._attn_cu_seqlens_cache.get(key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(0, batch * seq_len + 1, seq_len, dtype=torch.int32)
            if self.attn_impl in ("flash_attn2", "flash_attn3"):
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

    def _warn_attention_fallback_once(self, mask_mode):
        key = (self.attn_impl, mask_mode)
        if key in self._attn_fallback_warnings:
            return
        self._attn_fallback_warnings.add(key)
        logger.warning(
            "HunyuanImage3 attn_impl='{}' does not support {} attention masks in the low-intrusion path; "
            "falling back to PyTorch SDPA for this attention call.",
            self.attn_impl,
            mask_mode,
        )

    def _sdpa_attention(self, query_states, key_states, value_states, attention_mask):
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0)

    def _apply_registered_attention_kernel(self, query_states, key_states, value_states, causal):
        batch, query_heads, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        original_dtype = query_states.dtype
        q = self._normalize_attention_dtype(query_states.transpose(1, 2)).contiguous()
        k = self._normalize_attention_dtype(key_states.transpose(1, 2)).contiguous()
        v = self._normalize_attention_dtype(value_states.transpose(1, 2)).contiguous()
        cu_seqlens_q = self._get_cu_seqlens("q", batch, q_len, q.device)
        cu_seqlens_kv = self._get_cu_seqlens("kv", batch, kv_len, k.device)
        attn_output = self.attn_kernel.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=q_len,
            max_seqlen_kv=kv_len,
            causal=causal,
        )
        if attn_output.dim() == 2:
            attn_output = attn_output.reshape(batch, q_len, query_heads, self.head_dim)
        elif attn_output.dim() == 3:
            attn_output = attn_output.reshape(batch, q_len, query_heads, self.head_dim)
        else:
            raise RuntimeError(f"HunyuanImage3 attn_impl={self.attn_impl!r} returned unexpected shape {tuple(attn_output.shape)}.")
        return attn_output.to(original_dtype).transpose(1, 2)

    def _normalize_full_attn_slices(self, full_attn_slices, batch):
        if full_attn_slices is None:
            return [[] for _ in range(batch)]
        if len(full_attn_slices) == 0:
            return [[] for _ in range(batch)]

        first_item = full_attn_slices[0]
        if isinstance(first_item, slice) or (
            isinstance(first_item, (list, tuple)) and len(first_item) == 2 and all(isinstance(v, int) for v in first_item)
        ):
            full_attn_slices = [full_attn_slices for _ in range(batch)]
        elif len(full_attn_slices) == 1 and batch > 1:
            full_attn_slices = list(full_attn_slices) * batch

        normalized = []
        for sample_slices in full_attn_slices:
            sample = []
            for item in sample_slices:
                if isinstance(item, slice):
                    start = 0 if item.start is None else int(item.start)
                    stop = int(item.stop)
                else:
                    start, stop = int(item[0]), int(item[1])
                if stop > start:
                    sample.append((start, stop))
            normalized.append(sorted(sample))
        if len(normalized) != batch:
            return None
        return normalized

    @staticmethod
    def _find_full_attn_slice(full_slices, position):
        for start, stop in full_slices:
            if start <= position < stop:
                return start, stop
        return None

    def _build_segment_specs(self, position_ids, full_slices, kv_len):
        if torch.is_tensor(position_ids):
            positions = [int(position) for position in position_ids.detach().cpu().reshape(-1).tolist()]
        else:
            positions = [int(position) for position in position_ids]
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

    def _segmented_flash_attention(self, query_states, key_states, value_states, position_ids, full_attn_slices, segment_specs=None):
        if position_ids is None:
            return None
        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        if position_ids.shape != (batch, q_len):
            return None

        if segment_specs is None:
            if full_attn_slices is None:
                return None
            batch_full_slices = self._normalize_full_attn_slices(full_attn_slices, batch)
            if batch_full_slices is None or not any(batch_full_slices):
                return None
            segment_specs = [
                self._build_segment_specs(position_ids[batch_idx], batch_full_slices[batch_idx], kv_len)
                for batch_idx in range(batch)
            ]
        if len(segment_specs) != batch or any(specs is None for specs in segment_specs):
            return None

        output = torch.empty_like(query_states)
        for batch_idx in range(batch):
            for q_start, q_stop, kv_stop, causal in segment_specs[batch_idx]:
                segment_output = self._apply_registered_attention_kernel(
                    query_states[batch_idx : batch_idx + 1, :, q_start:q_stop],
                    key_states[batch_idx : batch_idx + 1, :, :kv_stop],
                    value_states[batch_idx : batch_idx + 1, :, :kv_stop],
                    causal=causal,
                )
                output[batch_idx : batch_idx + 1, :, q_start:q_stop] = segment_output
        return output

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
        if self.attn_impl in ("flash_attn2", "flash_attn3") and segment_specs is not None:
            segmented_output = self._segmented_flash_attention(
                query_states,
                key_states,
                value_states,
                position_ids,
                full_attn_slices,
                segment_specs=segment_specs,
            )
            if segmented_output is not None:
                return segmented_output
            raise RuntimeError("HunyuanImage3 precomputed segmented attention plan does not match the current Q/KV layout.")

        batch, _, q_len, _ = query_states.shape
        kv_len = key_states.shape[2]
        mask_mode = self._attention_mask_mode(attention_mask, q_len, kv_len)

        if self.attn_impl == "torch_sdpa":
            return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
        if self.attn_impl in ("flash_attn2", "flash_attn3"):
            if mask_mode not in ("none", "full", "causal"):
                segmented_output = self._segmented_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    position_ids,
                    full_attn_slices,
                    segment_specs=segment_specs,
                )
                if segmented_output is not None:
                    return segmented_output
                self._warn_attention_fallback_once(mask_mode)
                return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
            causal = mask_mode == "causal"
        elif self.attn_impl in ("sage_attn2", "sage_attn3"):
            if mask_mode not in ("none", "full"):
                self._warn_attention_fallback_once(mask_mode)
                return self._sdpa_attention(query_states, key_states, value_states, attention_mask)
            causal = False
        else:
            raise ValueError(f"Unsupported HunyuanImage3 normalized attn_impl={self.attn_impl!r}.")

        return self._apply_registered_attention_kernel(query_states, key_states, value_states, causal=causal)

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
        use_segment_specs = self.attn_impl in ("flash_attn2", "flash_attn3") and pre_infer_out.attention_segment_specs is not None
        attention_mask = None if use_segment_specs else self._cached_pre_infer_to_device("attention_mask", pre_infer_out.attention_mask, device)
        position_ids = self._cached_pre_infer_to_device("position_ids", pre_infer_out.position_ids, device)
        custom_pos_emb = self._cached_pre_infer_to_device("custom_pos_emb", pre_infer_out.custom_pos_emb, device)

        residual = hidden_states
        normed = attention_phase.input_layernorm.apply(hidden_states)
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
        normed = mlp_phase.post_attention_layernorm.apply(hidden_states)
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
        qkv_states = apply_linear(phase.qkv_proj, hidden_states.reshape(-1, hidden_states.shape[-1]))
        qkv_states = qkv_states.reshape(
            batch,
            q_len,
            self.num_key_value_heads,
            self.num_key_value_groups + 2,
            self.head_dim,
        )
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_key_value_groups, 1, 1], dim=3)
        query_states = query_states.reshape(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if custom_pos_emb is not None:
            cos, sin = custom_pos_emb
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if getattr(phase, "query_layernorm", None) is not None:
            query_states = phase.query_layernorm.apply(query_states)
            key_states = phase.key_layernorm.apply(key_states)

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
                if not (self.attn_impl in ("flash_attn2", "flash_attn3") and segment_specs is not None):
                    attention_mask = self._cached_pre_infer_to_device(
                        "sp_global_attention_mask",
                        sequence_parallel_state.global_attention_mask,
                        key_states.device,
                    )
            else:
                raise ValueError(f"Unsupported HunyuanImage3 sequence parallel attention type: {sequence_parallel_state.attn_type!r}.")

        if past_key_values is not None:
            if cache_position_ids is None:
                raise ValueError("HunyuanImage3 KV cache requires position_ids.")
            key_states, value_states = past_key_values.update(key_states, value_states, block_idx, cache_position_ids)
            query_states = query_states.to(key_states.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if sequence_parallel_state is not None and sequence_parallel_state.attn_type == "kv_all_gather":
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
        else:
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
        if self.attn_impl not in ("flash_attn2", "flash_attn3"):
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
        if self._attention_mask_mode(attention_mask, q_len, kv_len) != "custom":
            return None
        batch_full_slices = self._normalize_full_attn_slices(pre_infer_out.full_attn_slices, batch)
        if batch_full_slices is None or not any(batch_full_slices):
            return None
        segment_specs = [
            self._build_segment_specs(position_ids[batch_idx], batch_full_slices[batch_idx], kv_len)
            for batch_idx in range(batch)
        ]
        return None if any(specs is None for specs in segment_specs) else segment_specs

    def _cached_pre_infer_to_device(self, name, value, device):
        if value is None:
            return None
        cache = getattr(self, "_pre_infer_device_cache", None)
        if cache is None:
            self._pre_infer_device_cache = {}
            cache = self._pre_infer_device_cache
        key = (name, device, id(value))
        cached = cache.get(key)
        if cached is None:
            cached = to_device(value, device)
            cache[key] = cached
        return cached

    def _sequence_parallel_gather_kv(self, key_states, value_states, state):
        world_size = dist.get_world_size(self.seq_p_group)
        local = torch.stack((key_states, value_states), dim=2).permute(3, 0, 2, 1, 4).contiguous()
        output_shape = (local.shape[0] * world_size, *local.shape[1:])
        buffer_key = ("kv", local.device, local.dtype, output_shape)
        gathered = self._sp_gather_buffers.get(buffer_key)
        if gathered is None or gathered.shape != output_shape:
            gathered = torch.empty(output_shape, device=local.device, dtype=local.dtype)
            self._sp_gather_buffers[buffer_key] = gathered
        dist.all_gather_into_tensor(gathered, local, group=self.seq_p_group)
        valid = gathered[: state.original_seq_len]
        key_value = valid.permute(2, 1, 3, 0, 4).contiguous()
        return key_value[0], key_value[1]

    def _sequence_parallel_ulysses_seq_to_head(self, query_states, key_states, value_states, state):
        if query_states.shape[0] != 1:
            raise ValueError("HunyuanImage3 Ulysses expects batch size 1; set hunyuan_cfg_mode='serial'.")
        query = all2all_seq2head(query_states[0].transpose(0, 1).contiguous(), group=self.seq_p_group)
        key = all2all_seq2head(key_states[0].transpose(0, 1).contiguous(), group=self.seq_p_group)
        value = all2all_seq2head(value_states[0].transpose(0, 1).contiguous(), group=self.seq_p_group)
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
        output = all2all_head2seq(output, group=self.seq_p_group)
        return output.unsqueeze(0).transpose(1, 2).contiguous()

    def infer_mlp(self, phase, hidden_states):
        if not phase.is_moe:
            return apply_mlp(phase.gate_and_up_proj, phase.down_proj, hidden_states, self.hidden_act)

        moe = phase.moe
        moe_impl = getattr(moe, "moe_impl", self.config.get("moe_impl", "eager"))
        if moe_impl == "flashinfer":
            return self._infer_mlp_flashinfer(moe, hidden_states)
        if moe_impl != "eager":
            raise ValueError(f"Unsupported HunyuanImage3 moe_impl={moe_impl!r}. Expected 'eager' or 'flashinfer'.")

        return self._infer_mlp_eager(moe, hidden_states)

    def _moe_easy_topk(self, moe, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = apply_linear(moe.gate, flat)
        topk_weight, topk_idx = torch.topk(torch.softmax(logits, dim=-1), moe.moe_topk, dim=-1)
        topk_weight = topk_weight / torch.clamp(topk_weight.sum(dim=-1, keepdim=True), min=1e-8)
        return flat, topk_weight, topk_idx

    def _infer_mlp_eager(self, moe, hidden_states):
        flat, topk_weight, topk_idx = self._moe_easy_topk(moe, hidden_states)
        repeated = flat.repeat_interleave(moe.moe_topk, dim=0)
        expert_outputs = torch.zeros_like(repeated)
        flat_topk_idx = topk_idx.reshape(-1)
        for expert_idx, expert in enumerate(moe.experts):
            mask = flat_topk_idx == expert_idx
            if not torch.any(mask):
                continue
            expert_out = apply_mlp(expert.gate_and_up_proj, expert.down_proj, repeated[mask], self.hidden_act)
            expert_outputs[mask] = expert_out.to(expert_outputs.dtype)
        combined = (expert_outputs.reshape(flat.shape[0], moe.moe_topk, -1) * topk_weight.to(expert_outputs.dtype).unsqueeze(-1)).sum(dim=1)
        output = combined.reshape_as(hidden_states)
        if getattr(moe, "shared_mlp", None) is not None:
            shared_out = apply_mlp(moe.shared_mlp.gate_and_up_proj, moe.shared_mlp.down_proj, hidden_states, self.hidden_act)
            output = output + shared_out.to(output.dtype)
        return output

    def _infer_mlp_flashinfer(self, moe, hidden_states):
        if flashinfer_cutlass_fused_moe is None:
            raise ImportError("HunyuanImage3 moe_impl='flashinfer' requires flashinfer.fused_moe.cutlass_fused_moe.")
        if self.hidden_act != "silu":
            raise NotImplementedError("HunyuanImage3 moe_impl='flashinfer' currently supports only silu/SwiGLU experts.")
        if not hasattr(moe, "ensure_flashinfer_weights"):
            raise RuntimeError("HunyuanImage3 moe_impl='flashinfer' requires HunyuanImage3MoEWeights.")

        if hidden_states.device.type == "cuda" and hidden_states.device.index is not None:
            torch.cuda.set_device(hidden_states.device.index)

        original_dtype = hidden_states.dtype
        compute_dtype = original_dtype if original_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        flat, topk_weight, topk_idx = self._moe_easy_topk(moe, hidden_states)
        fused_input = flat.to(dtype=compute_dtype).contiguous()
        moe_weight, moe_weight_2 = moe.ensure_flashinfer_weights(fused_input.device, compute_dtype)
        combined_output = torch.zeros_like(fused_input)
        flashinfer_cutlass_fused_moe(
            fused_input,
            topk_idx.to(torch.int32).contiguous(),
            topk_weight.to(torch.float32).contiguous(),
            moe_weight,
            moe_weight_2,
            compute_dtype,
            output=combined_output,
            quant_scales=None,
            tune_max_num_tokens=self.flashinfer_tune_max_num_tokens,
        )
        output = combined_output.reshape_as(hidden_states).to(original_dtype)
        if getattr(moe, "shared_mlp", None) is not None:
            shared_out = apply_mlp(moe.shared_mlp.gate_and_up_proj, moe.shared_mlp.down_proj, hidden_states, self.hidden_act)
            output = output + shared_out.to(output.dtype)
        return output
