import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.dreamzero.pre_infer import _category_linear
from lightx2v.models.networks.wan.infer.transformer_infer import WanTransformerInfer
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

    def create_empty_kv_cache(self, dtype, device):
        return [torch.zeros(2, 0, self.num_heads, self.head_dim, dtype=dtype, device=device) for _ in range(self.blocks_num)]

    def get_kv_cache(self, cache_name, dtype, device):
        cache = self.kv_caches.get(cache_name)
        if cache is None:
            cache = self.create_empty_kv_cache(dtype=dtype, device=device)
            self.kv_caches[cache_name] = cache
        return cache

    def set_kv_cache(self, cache_name, cache):
        self.kv_caches[cache_name] = cache

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

    @staticmethod
    def _apply_rope_polar(x, freqs):
        x_dtype = x.dtype
        seq_len, num_heads, _ = x.shape
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(seq_len, num_heads, -1, 2))
        out = torch.view_as_real(x_complex * freqs.to(x.device)).flatten(2)
        return out.to(x_dtype)

    def _causal_rope_action_apply(self, x, freqs, freqs_action, freqs_state, action_register_length, current_start_frame):
        if action_register_length is not None:
            action_state_index = (int(current_start_frame) - 1) // int(self.config.get("num_frame_per_block", 2))
            action_state_index = max(action_state_index, 0)
            action_start = action_state_index * self.num_action_per_block
            state_start = action_state_index * self.num_state_per_block
            action_freqs = freqs_action[action_start : action_start + self.num_action_per_block]
            state_freqs = freqs_state[state_start : state_start + self.num_state_per_block]
            freqs = torch.cat(
                [
                    freqs,
                    action_freqs.view(self.num_action_per_block, 1, -1),
                    state_freqs.view(self.num_state_per_block, 1, -1),
                ],
                dim=0,
            )
        return self._apply_rope_polar(x, freqs)

    def infer_self_attn_with_cache(self, phase, x, shift_msa, scale_msa, pre_infer_out, kv_cache):
        seq_total = x.shape[0]
        norm1_out = phase.norm1.apply(x).to(x.dtype)
        norm1_out = (norm1_out.float() * (1.0 + scale_msa.float()) + shift_msa.float()).to(x.dtype)

        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(seq_total, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(seq_total, self.num_heads, self.head_dim)
        v = phase.self_attn_v.apply(norm1_out).view(seq_total, self.num_heads, self.head_dim)
        q = self._causal_rope_action_apply(
            q,
            pre_infer_out.freqs,
            pre_infer_out.freqs_action,
            pre_infer_out.freqs_state,
            pre_infer_out.action_register_length,
            pre_infer_out.current_start_frame,
        )
        k = self._causal_rope_action_apply(
            k,
            pre_infer_out.freqs,
            pre_infer_out.freqs_action,
            pre_infer_out.freqs_state,
            pre_infer_out.action_register_length,
            pre_infer_out.current_start_frame,
        )

        action_q = action_k = action_v = None
        if pre_infer_out.action_register_length is not None:
            action_len = pre_infer_out.action_register_length
            action_q = q[-action_len:]
            action_k = k[-action_len:]
            action_v = v[-action_len:]
            q = q[:-action_len]
            k = k[:-action_len]
            v = v[:-action_len]

        cached_k = kv_cache[self.block_idx][0]
        cached_v = kv_cache[self.block_idx][1]
        new_k = torch.cat([cached_k, k], dim=0)
        new_v = torch.cat([cached_v, v], dim=0)
        if self.max_attention_size > 0:
            new_k = new_k[-self.max_attention_size :]
            new_v = new_v[-self.max_attention_size :]

        if action_q is not None:
            q_attn = torch.cat([q, action_q], dim=0)
            k_attn = torch.cat([new_k, action_k], dim=0)
            v_attn = torch.cat([new_v, action_v], dim=0)
        else:
            q_attn = q
            k_attn = new_k
            v_attn = new_v

        cu_q = torch.tensor([0, q_attn.shape[0]], device=q_attn.device, dtype=torch.int32)
        cu_kv = torch.tensor([0, k_attn.shape[0]], device=k_attn.device, dtype=torch.int32)
        attn_out = phase.self_attn_1.apply(
            q=q_attn,
            k=k_attn,
            v=v_attn,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q_attn.shape[0],
            max_seqlen_kv=k_attn.shape[0],
        )
        updated_cache = None
        if pre_infer_out.update_cache:
            updated_cache = torch.stack([new_k, new_v], dim=0)
        return phase.self_attn_o.apply(attn_out), updated_cache

    def infer_cross_attn_dreamzero(self, phase, x, y_out, gate_msa, pre_infer_out):
        x = x + y_out * gate_msa
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

        cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
        cu_kv = torch.tensor([0, k.shape[0]], device=k.device, dtype=torch.int32)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )

        cu_kv_img = torch.tensor([0, k_img.shape[0]], device=k_img.device, dtype=torch.int32)
        img_attn_out = phase.cross_attn_2.apply(
            q=q,
            k=k_img,
            v=v_img,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv_img,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k_img.shape[0],
        )
        return x, phase.cross_attn_o.apply(attn_out + img_attn_out)

    def infer_ffn_dreamzero(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x = x + attn_out
        norm2_out = phase.norm2.apply(x).to(x.dtype)
        norm2_out = (norm2_out.float() * (1.0 + c_scale_msa.float()) + c_shift_msa.float()).to(x.dtype)
        y = phase.ffn_0.apply(norm2_out)
        y = F.gelu(y, approximate="tanh")
        return phase.ffn_2.apply(y)

    def infer_block(self, block, x, pre_infer_out, kv_cache):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._split_modulation(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out, updated_cache = self.infer_self_attn_with_cache(
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
        x = x + attn_out + y * c_gate_msa
        return x, updated_cache

    def infer_main_blocks(self, blocks, pre_infer_out, kv_cache):
        x = pre_infer_out.x
        updated_caches = [] if pre_infer_out.update_cache else None
        for block_idx in range(len(blocks)):
            self.block_idx = block_idx
            x, updated_cache = self.infer_block(blocks[block_idx], x, pre_infer_out, kv_cache)
            if pre_infer_out.update_cache:
                updated_caches.append(updated_cache)
        return x, updated_caches

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
        x = (x.float() * (1.0 + scale.float()) + shift.float()).to(x.dtype)
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
        x, updated_caches = self.infer_main_blocks(weights.blocks, pre_infer_out, kv_cache)
        if pre_infer_out.update_cache:
            self.set_kv_cache(pre_infer_out.cache_name, updated_caches)
        video_noise_pred = self.infer_video_head(weights, x, pre_infer_out).to(GET_DTYPE())
        action_noise_pred = self.infer_action_decoder(weights, x, pre_infer_out)
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.to(GET_DTYPE())
        return video_noise_pred, action_noise_pred
