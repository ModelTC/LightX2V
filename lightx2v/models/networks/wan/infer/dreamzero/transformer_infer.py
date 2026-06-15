import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.dreamzero.pre_infer import _category_linear
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
from lightx2v.models.networks.wan.infer.triton_ops import apply_rotary_embedding
from lightx2v.models.networks.wan.infer.utils import apply_rope_with_cos_sin_cache_inplace
from lightx2v.utils.envs import GET_DTYPE


class DreamZeroTransformerInfer(WanTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_action_per_block = int(config.get("num_action_per_block", config.get("action_horizon", 24)))
        self.num_state_per_block = int(config.get("num_state_per_block", 1))
        self.frame_seqlen = int(config["frame_seqlen"])
        local_attn_size = config.get("local_attn_size")
        if local_attn_size is None:
            max_chunk_size = int(config.get("max_chunk_size", 4))
            local_attn_size = max_chunk_size * int(config.get("num_frame_per_block", 2)) + 1
        self.max_attention_size = int(local_attn_size) * self.frame_seqlen
        self.kv_caches = {}
        self.cross_attn_kv_caches = {}
        self._cu_seqlens_cache = {}
        self._rope_cache = {}
        self.dreamzero_rope_type = config.get("dreamzero_rope_type", config.get("rope_type", "flashinfer"))

    def create_empty_kv_cache(self, dtype, device):
        capacity = max(int(self.max_attention_size), 1)
        return {
            "capacity": capacity,
            "window_size": int(self.max_attention_size),
            "k": [torch.empty(capacity, self.num_heads, self.head_dim, dtype=dtype, device=device) for _ in range(self.blocks_num)],
            "v": [torch.empty(capacity, self.num_heads, self.head_dim, dtype=dtype, device=device) for _ in range(self.blocks_num)],
            "lengths": [0 for _ in range(self.blocks_num)],
            "scratch_k": [None for _ in range(self.blocks_num)],
            "scratch_v": [None for _ in range(self.blocks_num)],
        }

    def get_kv_cache(self, cache_name, dtype, device):
        cache = self.kv_caches.get(cache_name)
        if cache is None or cache["k"][0].dtype != dtype or cache["k"][0].device != device:
            cache = self.create_empty_kv_cache(dtype=dtype, device=device)
            self.kv_caches[cache_name] = cache
        return cache

    def set_kv_cache(self, cache_name, cache):
        self.kv_caches[cache_name] = cache

    def _ensure_kv_cache_capacity(self, cache, min_capacity, dtype, device):
        if cache["capacity"] >= min_capacity:
            return cache
        new_capacity = max(min_capacity, cache["capacity"] * 2)
        old_capacity = cache["capacity"]
        for block_idx in range(self.blocks_num):
            old_len = min(cache["lengths"][block_idx], old_capacity)
            new_k = torch.empty(new_capacity, self.num_heads, self.head_dim, dtype=dtype, device=device)
            new_v = torch.empty(new_capacity, self.num_heads, self.head_dim, dtype=dtype, device=device)
            if old_len > 0:
                new_k[:old_len].copy_(cache["k"][block_idx][:old_len])
                new_v[:old_len].copy_(cache["v"][block_idx][:old_len])
            cache["k"][block_idx] = new_k
            cache["v"][block_idx] = new_v
        cache["capacity"] = new_capacity
        return cache

    def _get_scratch(self, cache, name, block_idx, length, dtype, device):
        scratch = cache[name][block_idx]
        if scratch is None or scratch.shape[0] < length or scratch.device != device or scratch.dtype != dtype:
            scratch = torch.empty(max(length, 1), self.num_heads, self.head_dim, dtype=dtype, device=device)
            cache[name][block_idx] = scratch
        return scratch[:length]

    def _store_video_kv(self, cache, block_idx, k, v):
        capacity = cache["capacity"]
        if int(cache["window_size"]) <= 0:
            old_len = cache["lengths"][block_idx]
            new_len = old_len + k.shape[0]
            self._ensure_kv_cache_capacity(cache, new_len, k.dtype, k.device)
            cache["k"][block_idx][old_len:new_len].copy_(k)
            cache["v"][block_idx][old_len:new_len].copy_(v)
            cache["lengths"][block_idx] = new_len
            return

        if k.shape[0] >= capacity:
            cache["k"][block_idx].copy_(k[-capacity:])
            cache["v"][block_idx].copy_(v[-capacity:])
            cache["lengths"][block_idx] = capacity
            return

        old_len = cache["lengths"][block_idx]
        keep = min(old_len, capacity - k.shape[0])
        if keep > 0 and old_len != keep:
            cache["k"][block_idx][:keep].copy_(cache["k"][block_idx][old_len - keep : old_len].clone())
            cache["v"][block_idx][:keep].copy_(cache["v"][block_idx][old_len - keep : old_len].clone())
        cache["k"][block_idx][keep : keep + k.shape[0]].copy_(k)
        cache["v"][block_idx][keep : keep + v.shape[0]].copy_(v)
        cache["lengths"][block_idx] = keep + k.shape[0]

    def _materialize_self_attn_kv(self, cache, block_idx, k, v, action_k=None, action_v=None):
        old_len = cache["lengths"][block_idx]
        capacity = cache["capacity"]
        if int(cache["window_size"]) <= 0:
            video_len = old_len + k.shape[0]
            self._ensure_kv_cache_capacity(cache, video_len, k.dtype, k.device)
            capacity = cache["capacity"]
        else:
            video_len = min(old_len + k.shape[0], capacity)
        old_keep = max(0, video_len - k.shape[0])
        cur_keep = video_len - old_keep
        action_len = 0 if action_k is None else action_k.shape[0]
        total_len = video_len + action_len
        attn_k = self._get_scratch(cache, "scratch_k", block_idx, total_len, k.dtype, k.device)
        attn_v = self._get_scratch(cache, "scratch_v", block_idx, total_len, v.dtype, v.device)

        offset = 0
        if old_keep > 0:
            attn_k[:old_keep].copy_(cache["k"][block_idx][old_len - old_keep : old_len])
            attn_v[:old_keep].copy_(cache["v"][block_idx][old_len - old_keep : old_len])
            offset = old_keep
        if cur_keep > 0:
            attn_k[offset : offset + cur_keep].copy_(k[-cur_keep:])
            attn_v[offset : offset + cur_keep].copy_(v[-cur_keep:])
            offset += cur_keep
        if action_len > 0:
            attn_k[offset : offset + action_len].copy_(action_k)
            attn_v[offset : offset + action_len].copy_(action_v)
        return attn_k, attn_v

    def get_cross_attn_kv_cache(self, cache_name):
        cache = self.cross_attn_kv_caches.get(cache_name)
        if cache is None:
            cache = [None for _ in range(self.blocks_num)]
            self.cross_attn_kv_caches[cache_name] = cache
        return cache

    @staticmethod
    def _cross_attn_cache_key(cache_name, context):
        return (cache_name, context.data_ptr(), tuple(context.shape), str(context.device), str(context.dtype))

    def clear_cache(self, cache_name=None):
        if cache_name is None:
            self.kv_caches.clear()
            self.cross_attn_kv_caches.clear()
            self._cu_seqlens_cache.clear()
            self._rope_cache.clear()
        else:
            cache_names = {cache_name, f"{cache_name}_cond", f"{cache_name}_uncond"}
            for name in cache_names:
                self.kv_caches.pop(name, None)
            for key in list(self.cross_attn_kv_caches.keys()):
                key_cache_name = key[0] if isinstance(key, tuple) else key
                if key_cache_name in cache_names:
                    self.cross_attn_kv_caches.pop(key, None)

    @staticmethod
    def _token_modulation(x):
        return x.reshape(-1, x.shape[-1])

    def _split_modulation(self, modulation, embed0):
        table = modulation.tensor.to(embed0.device, embed0.dtype)
        values = table + embed0
        return [self._token_modulation(item) for item in values.chunk(6, dim=1)]

    def _get_cu_seqlens(self, length, device):
        key = (int(length), str(device))
        cached = self._cu_seqlens_cache.get(key)
        if cached is None:
            cached = torch.tensor([0, int(length)], device=device, dtype=torch.int32)
            self._cu_seqlens_cache[key] = cached
        return cached

    def _get_rope_cos_sin(self, freqs):
        key = ("cos_sin", freqs.data_ptr(), tuple(freqs.shape), str(freqs.device), str(freqs.dtype))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        cos = freqs.real.reshape(freqs.shape[0], -1).contiguous()
        sin = freqs.imag.reshape(freqs.shape[0], -1).contiguous()
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def _get_flashinfer_cos_sin(self, freqs):
        key = ("flashinfer", freqs.data_ptr(), tuple(freqs.shape), str(freqs.device), str(freqs.dtype))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        cos, sin = self._get_rope_cos_sin(freqs)
        cos_sin = torch.cat([cos, sin], dim=-1).contiguous()
        self._rope_cache[key] = cos_sin
        return cos_sin

    def _get_rope_positions(self, length, device):
        key = ("positions", int(length), str(device))
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = torch.arange(int(length), device=device, dtype=torch.long)
            self._rope_cache[key] = cached
        return cached

    def _modulate(self, x, scale, shift):
        if x.is_cuda and x.is_contiguous():
            scale_arg = scale
            shift_arg = shift
            if x.dim() == 2 and scale.dim() == 2 and scale.shape[0] == x.shape[0]:
                scale_arg = scale.unsqueeze(0)
                shift_arg = shift.unsqueeze(0)
            return self.modulate_func(x, scale=scale_arg, shift=shift_arg).reshape_as(x)
        return (x.float() * (1.0 + scale.float()) + shift.float()).to(x.dtype)

    @staticmethod
    def _apply_rope_polar(x, freqs):
        x_dtype = x.dtype
        seq_len, num_heads, _ = x.shape
        x_complex = torch.view_as_complex(x.to(torch.float32).reshape(seq_len, num_heads, -1, 2))
        out = torch.view_as_real(x_complex * freqs.to(x.device)).flatten(2)
        return out.to(x_dtype)

    def _apply_rope(self, q, k, freqs):
        if freqs.shape[0] != q.shape[0]:
            raise ValueError(f"DreamZero RoPE length mismatch: freqs={freqs.shape[0]}, q={q.shape[0]}.")

        if q.is_cuda and self.dreamzero_rope_type == "flashinfer" and apply_rope_with_cos_sin_cache_inplace is not None:
            seq_len, num_heads, head_dim = q.shape
            query = q.reshape(seq_len, num_heads * head_dim).contiguous()
            key = k.reshape(seq_len, num_heads * head_dim).contiguous()
            apply_rope_with_cos_sin_cache_inplace(
                positions=self._get_rope_positions(seq_len, q.device),
                query=query,
                key=key,
                head_size=head_dim,
                cos_sin_cache=self._get_flashinfer_cos_sin(freqs),
                is_neox=False,
            )
            return query.view(seq_len, num_heads, head_dim), key.view(seq_len, num_heads, head_dim)

        if q.is_cuda and self.dreamzero_rope_type in {"flashinfer", "triton"}:
            cos, sin = self._get_rope_cos_sin(freqs)
            return (
                apply_rotary_embedding(q.contiguous(), cos, sin, interleaved=False),
                apply_rotary_embedding(k.contiguous(), cos, sin, interleaved=False),
            )

        return self._apply_rope_polar(q, freqs), self._apply_rope_polar(k, freqs)

    def infer_self_attn_with_cache(self, phase, x, shift_msa, scale_msa, pre_infer_out, kv_cache):
        seq_total = x.shape[0]
        norm1_out = phase.norm1.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)
        norm1_out = self._modulate(norm1_out.contiguous(), scale_msa, shift_msa)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(seq_total, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(seq_total, self.num_heads, self.head_dim)
        v = phase.self_attn_v.apply(norm1_out).view(seq_total, self.num_heads, self.head_dim)
        q, k = self._apply_rope(q, k, pre_infer_out.freqs)

        action_k = action_v = None
        if pre_infer_out.action_register_length is not None:
            action_len = pre_infer_out.action_register_length
            action_k = k[-action_len:]
            action_v = v[-action_len:]
            k = k[:-action_len]
            v = v[:-action_len]

        q_attn = q
        k_attn, v_attn = self._materialize_self_attn_kv(kv_cache, self.block_idx, k, v, action_k, action_v)
        if pre_infer_out.update_cache:
            self._store_video_kv(kv_cache, self.block_idx, k, v)

        cu_q = self._get_cu_seqlens(q_attn.shape[0], q_attn.device)
        cu_kv = self._get_cu_seqlens(k_attn.shape[0], k_attn.device)
        attn_out = phase.self_attn_1.apply(
            q=q_attn,
            k=k_attn,
            v=v_attn,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q_attn.shape[0],
            max_seqlen_kv=k_attn.shape[0],
        )
        return phase.self_attn_o.apply(attn_out)

    def infer_cross_attn_dreamzero(self, phase, x, y_out, gate_msa, pre_infer_out):
        x.add_(y_out * gate_msa)
        norm3_out = phase.norm3.apply(x)
        context = pre_infer_out.context
        if context is None:
            raise ValueError("DreamZero cross-attention requires projected context.")

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        cross_attn_kv_cache = self.get_cross_attn_kv_cache(self._cross_attn_cache_key(pre_infer_out.cache_name, context))
        cached_kv = cross_attn_kv_cache[self.block_idx]
        if cached_kv is None:
            context_img = context[:257]
            context_txt = context[257:]
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context_txt)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context_txt).view(-1, n, d)
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            cross_attn_kv_cache[self.block_idx] = (k, v, k_img, v_img)
        else:
            k, v, k_img, v_img = cached_kv

        cu_q = self._get_cu_seqlens(q.shape[0], q.device)
        cu_kv = self._get_cu_seqlens(k.shape[0], k.device)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )

        cu_kv_img = self._get_cu_seqlens(k_img.shape[0], k_img.device)
        img_attn_out = phase.cross_attn_2.apply(
            q=q,
            k=k_img,
            v=v_img,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv_img,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k_img.shape[0],
        )
        attn_out.add_(img_attn_out)
        return x, phase.cross_attn_o.apply(attn_out)

    def infer_ffn_dreamzero(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)
        norm2_out = phase.norm2.apply(x)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.sensitive_layer_dtype)
        norm2_out = self._modulate(norm2_out.contiguous(), c_scale_msa, c_shift_msa)
        if self.sensitive_layer_dtype != self.infer_dtype:
            norm2_out = norm2_out.to(self.infer_dtype)
        y = phase.ffn_0.apply(norm2_out)
        y = F.gelu(y, approximate="tanh")
        return phase.ffn_2.apply(y)

    def infer_block(self, block, x, pre_infer_out, kv_cache):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._split_modulation(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out = self.infer_self_attn_with_cache(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            pre_infer_out,
            kv_cache,
        )
        x, attn_out = self.infer_cross_attn_dreamzero(
            block.compute_phases[1],
            x,
            y_out,
            gate_msa,
            pre_infer_out,
        )
        y = self.infer_ffn_dreamzero(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x.add_(y * c_gate_msa)
        return x

    def infer_main_blocks(self, blocks, pre_infer_out, kv_cache):
        x = pre_infer_out.x
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x = self.infer_block(blocks[block_idx], x, pre_infer_out, kv_cache)
        return x

    def infer_action_decoder(self, weights, x, pre_infer_out):
        if pre_infer_out.action_length == 0:
            return None
        action_tokens = x[pre_infer_out.seq_len : pre_infer_out.seq_len + pre_infer_out.action_length].unsqueeze(0)
        hidden = F.relu(_category_linear(action_tokens, weights.action_decoder.layer1))
        return _category_linear(hidden, weights.action_decoder.layer2)

    def infer_video_head(self, weights, x, pre_infer_out):
        x = x[: pre_infer_out.seq_len]
        embed = pre_infer_out.embed[: pre_infer_out.seq_len]
        table = weights.head_modulation.tensor.to(x.device, x.dtype)
        shift, scale = [self._token_modulation(item) for item in (table + embed[:, None, :]).chunk(2, dim=1)]
        x = weights.norm.apply(x).to(x.dtype)
        x = self._modulate(x.contiguous(), scale, shift)
        x = weights.head.apply(x)
        return self.unpatchify(x, pre_infer_out.grid_size)

    def unpatchify(self, x, grid_size):
        batch_size = 1
        c = self.config["out_dim"]
        p_t, p_h, p_w = tuple(self.config.get("patch_size", (1, 2, 2)))
        f, h, w = grid_size
        if x.shape[0] != math.prod(grid_size):
            raise ValueError(f"DreamZero unpatchify expected {math.prod(grid_size)} tokens, got {x.shape[0]}.")
        x = x.view(batch_size, f, h, w, p_t, p_h, p_w, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(batch_size, c, f * p_t, h * p_h, w * p_w)

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.reset_infer_states()
        kv_cache = self.get_kv_cache(pre_infer_out.cache_name, dtype=pre_infer_out.x.dtype, device=pre_infer_out.x.device)
        x = self.infer_main_blocks(weights.blocks, pre_infer_out, kv_cache)
        video_noise_pred = self.infer_video_head(weights, x, pre_infer_out).to(GET_DTYPE())
        action_noise_pred = self.infer_action_decoder(weights, x, pre_infer_out)
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.to(GET_DTYPE())
        return video_noise_pred, action_noise_pred
