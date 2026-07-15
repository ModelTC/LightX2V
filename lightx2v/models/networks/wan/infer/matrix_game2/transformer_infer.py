import torch
from einops import rearrange

from lightx2v.models.networks.wan.infer.matrix_game2.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanMtxg2TransformerInfer(WanSFTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.vae_time_compression_ratio = config["action_config"]["vae_time_compression_ratio"]
        self.windows_size = config["action_config"]["windows_size"]
        self.patch_size = config["action_config"]["patch_size"]
        self.rope_theta = config["action_config"]["rope_theta"]
        self.enable_keyboard = config["action_config"]["enable_keyboard"]
        self.enable_mouse = config["action_config"].get("enable_mouse", False)
        self.heads_num = config["action_config"]["heads_num"]
        self.hidden_size = config["action_config"]["hidden_size"]
        self.rope_dim_list = config["action_config"]["rope_dim_list"]
        self.freqs_cos, self.freqs_sin = self._get_rotary_pos_embed(7500, self.patch_size[1], self.patch_size[2], 64, self.rope_dim_list, start_offset=0)
        if self.enable_mouse:
            self.mouse_dim_in = config["action_config"]["mouse_dim_in"]
            self.mouse_hidden_dim = config["action_config"]["mouse_hidden_dim"]
            self.mouse_qk_dim_list = config["action_config"]["mouse_qk_dim_list"]

    def _get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list=None, start_offset=0):
        target_ndim = 3
        ndim = 5 - 2

        latents_size = [video_length + start_offset, height, width]

        if isinstance(self.patch_size, int):
            assert all(s % self.patch_size == 0 for s in latents_size), f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), but got {latents_size}."
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            assert all(s % self.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.patch_size}), but got {latents_size}."
            )
            rope_sizes = [s // self.patch_size[idx] for idx, s in enumerate(latents_size)]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos[-video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0] :], freqs_sin[-video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0] :]

    def _infer_action_attn(self, phase, q, k, v):
        attn_mod = phase.action_attn_1
        if q.dim() == 3:
            out = attn_mod.apply(q, k, v, max_seqlen_q=q.size(0), max_seqlen_kv=k.size(0))
            return out.view(q.size(0), q.size(1), q.size(2))
        B, Lq, H, D = q.shape
        Lk = k.size(1)
        if B == 1:
            out = attn_mod.apply(q, k, v, max_seqlen_q=Lq, max_seqlen_kv=Lk)
            return out.view(1, Lq, H, D)
        cu_q = torch.arange(B + 1, device=q.device, dtype=torch.int32) * Lq
        cu_k = torch.arange(B + 1, device=k.device, dtype=torch.int32) * Lk
        out = attn_mod.apply(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=Lq,
            max_seqlen_kv=Lk,
        )
        return out.view(B, Lq, H, D)

    def infer_action_model(self, phase, x, grid_sizes, mouse_condition=None, keyboard_condition=None, is_causal=False, use_rope_keyboard=True):
        tt, th, tw = grid_sizes
        current_start = self.scheduler.seg_index * self.num_frame_per_chunk
        start_frame = current_start
        B, N_frames, C = keyboard_condition.shape
        assert tt * th * tw == x.shape[0]
        assert ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1

        freqs_cis = (self.freqs_cos, self.freqs_sin)

        cond1 = N_feats == tt
        cond2 = is_causal and not self.enable_mouse
        cond3 = (N_frames - 1) // self.vae_time_compression_ratio + 1 == current_start + self.num_frame_per_chunk
        assert (cond1 and ((cond2) or not is_causal)) or (cond3 and is_causal)

        msc = getattr(self.kv_cache_manager, "action_mouse_kv_cache", None)
        mouse_offload = msc is not None and getattr(msc, "_kv_offload", False) and self.enable_mouse and mouse_condition is not None and is_causal
        if mouse_offload:
            msc.begin_layer(self.block_idx)
        out = self._infer_action_model_with_kvcache(
            phase,
            x,
            mouse_condition,
            keyboard_condition,
            is_causal,
            use_rope_keyboard,
            tt,
            th,
            tw,
            current_start,
            start_frame,
            B,
            N_feats,
            freqs_cis,
        )
        if mouse_offload:
            msc.end_layer(self.block_idx, next_prefetch=None)
        return out

    def _infer_action_model_with_kvcache(
        self,
        phase,
        x,
        mouse_condition,
        keyboard_condition,
        is_causal,
        use_rope_keyboard,
        tt,
        th,
        tw,
        current_start,
        start_frame,
        B,
        N_feats,
        freqs_cis,
    ):
        x = x.unsqueeze(0)
        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th * tw)
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x

        pad_t = self.vae_time_compression_ratio * self.windows_size
        max_attention_size = self.kv_cache_manager.max_attention_size

        if self.enable_mouse and mouse_condition is not None:
            pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
            mouse_condition = torch.cat([pad, mouse_condition], dim=1)
            if is_causal:
                mouse_condition = mouse_condition[:, self.vae_time_compression_ratio * (N_feats - self.num_frame_per_chunk - self.windows_size) + pad_t :, :]
                group_mouse = [
                    mouse_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(self.num_frame_per_chunk)
                ]
            else:
                group_mouse = [mouse_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]

            group_mouse = torch.stack(group_mouse, dim=1)

            S = th * tw
            group_mouse = group_mouse.unsqueeze(-1).expand(B, self.num_frame_per_chunk, pad_t, C, S)
            group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(B * S, self.num_frame_per_chunk, pad_t * C)

            group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)

            group_mouse = torch.nn.functional.linear(group_mouse, phase.mouse_mlp_0.weight.T, phase.mouse_mlp_0.bias)
            group_mouse = torch.nn.functional.gelu(group_mouse, approximate="tanh")
            group_mouse = torch.nn.functional.linear(group_mouse, phase.mouse_mlp_2.weight.T, phase.mouse_mlp_2.bias)
            group_mouse = torch.nn.functional.layer_norm(group_mouse, (group_mouse.shape[-1],), phase.mouse_mlp_3.weight.T, phase.mouse_mlp_3.bias, 1e-5)

            mouse_qkv = torch.nn.functional.linear(group_mouse, phase.t_qkv.weight.T)

            q0, k0, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            q = q0 * torch.rsqrt(q0.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k0 * torch.rsqrt(k0.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

            q, k = apply_rotary_emb(q, k, freqs_cis, start_offset=start_frame, head_first=False)

            if is_causal:
                action_mouse_kv_cache = self.kv_cache_manager.action_mouse_kv_cache
                current_end = current_start + q.shape[1]
                assert q.shape[1] == self.num_frame_per_chunk
                sink_tokens = 0
                li = self.block_idx
                g = action_mouse_kv_cache.get_global_end(li)
                l_ = action_mouse_kv_cache.get_local_end(li)
                num_new_tokens = q.shape[1]
                kv_cache_size = action_mouse_kv_cache.cache_size
                if current_end > g and num_new_tokens + l_ > kv_cache_size:
                    num_evicted_tokens = num_new_tokens + l_ - kv_cache_size
                    action_mouse_kv_cache.roll_window(li, sink_tokens, num_evicted_tokens)
                    local_end_index = l_ + current_end - g - num_evicted_tokens
                else:
                    local_end_index = l_ + current_end - g
                local_start_index = local_end_index - num_new_tokens
                action_mouse_kv_cache.store_kv(k, v, local_start_index, local_end_index, li)
                action_mouse_kv_cache.set_ends(li, current_end, local_end_index)
                attn_s = max(0, local_end_index - max_attention_size)
                attn_k = action_mouse_kv_cache.k_cache(li, attn_s, local_end_index)
                attn_v = action_mouse_kv_cache.v_cache(li, attn_s, local_end_index)
                attn = self._infer_action_attn(phase, q, attn_k, attn_v)
            else:
                attn = self._infer_action_attn(phase, q, k, v)

            attn = rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)

            attn = phase.proj_mouse.apply(attn[0]).unsqueeze(0)
            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
            keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
            if is_causal:
                keyboard_condition = keyboard_condition[:, self.vae_time_compression_ratio * (N_feats - self.num_frame_per_chunk - self.windows_size) + pad_t :, :]

                keyboard_condition = phase.keyboard_embed_0.apply(keyboard_condition[0])
                keyboard_condition = torch.nn.functional.silu(keyboard_condition)
                keyboard_condition = phase.keyboard_embed_2.apply(keyboard_condition).unsqueeze(0)
                group_keyboard = [
                    keyboard_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(self.num_frame_per_chunk)
                ]
            else:
                keyboard_condition = phase.keyboard_embed_0.apply(keyboard_condition[0])
                keyboard_condition = torch.nn.functional.silu(keyboard_condition)
                keyboard_condition = phase.keyboard_embed_2.apply(keyboard_condition).unsqueeze(0)
                group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]

            group_keyboard = torch.stack(group_keyboard, dim=1)
            group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0], group_keyboard.shape[1], -1))

            mouse_q = phase.mouse_attn_q.apply(hidden_states[0]).unsqueeze(0)
            keyboard_kv = phase.keyboard_attn_kv.apply(group_keyboard[0]).unsqueeze(0)

            B, L, HD = mouse_q.shape
            D = HD // self.heads_num
            q = mouse_q.view(B, L, self.heads_num, D)

            B, L, KHD = keyboard_kv.shape
            k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(2, 0, 1, 3, 4)

            q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            S = th * tw

            action_keyboard_kv_cache = self.kv_cache_manager.action_keyboard_kv_cache
            if use_rope_keyboard:
                B, TS, H, D = q.shape
                T_ = TS // S
                q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
                q, k = apply_rotary_emb(q, k, freqs_cis, start_offset=start_frame, head_first=False)

                k = k.expand(S, k.shape[1], k.shape[2], k.shape[3])
                v = v.expand(S, v.shape[1], v.shape[2], v.shape[3])

                if is_causal:
                    current_end = current_start + k.shape[1]
                    assert k.shape[1] == self.num_frame_per_chunk
                    sink_tokens = 0
                    li = self.block_idx
                    g = action_keyboard_kv_cache.get_global_end(li)
                    l_ = action_keyboard_kv_cache.get_local_end(li)
                    num_new_tokens = k.shape[1]
                    kv_cache_size = action_keyboard_kv_cache.cache_size
                    if current_end > g and num_new_tokens + l_ > kv_cache_size:
                        num_evicted_tokens = num_new_tokens + l_ - kv_cache_size
                        action_keyboard_kv_cache.roll_window(li, sink_tokens, num_evicted_tokens)
                        local_end_index = l_ + current_end - g - num_evicted_tokens
                    else:
                        local_end_index = l_ + current_end - g
                    local_start_index = local_end_index - num_new_tokens
                    assert k.shape[0] == S
                    action_keyboard_kv_cache.store_kv(k[:1][0], v[:1][0], local_start_index, local_end_index, li)
                    action_keyboard_kv_cache.set_ends(li, current_end, local_end_index)
                    attn_s = max(0, local_end_index - max_attention_size)
                    slice_k = action_keyboard_kv_cache.k_cache(li, attn_s, local_end_index).unsqueeze(0).expand(S, -1, -1, -1)
                    slice_v = action_keyboard_kv_cache.v_cache(li, attn_s, local_end_index).unsqueeze(0).expand(S, -1, -1, -1)
                    attn = self._infer_action_attn(phase, q, slice_k, slice_v)
                else:
                    attn = self._infer_action_attn(phase, q, k, v)
                attn = rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)

            else:
                if is_causal:
                    current_end = start_frame + k.shape[1]
                    assert k.shape[1] == self.num_frame_per_chunk
                    sink_tokens = 0
                    li = self.block_idx
                    g = action_keyboard_kv_cache.get_global_end(li)
                    l_ = action_keyboard_kv_cache.get_local_end(li)
                    num_new_tokens = k.shape[1]
                    kv_cache_size = action_keyboard_kv_cache.cache_size
                    if current_end > g and num_new_tokens + l_ > kv_cache_size:
                        num_evicted_tokens = num_new_tokens + l_ - kv_cache_size
                        action_keyboard_kv_cache.roll_window(li, sink_tokens, num_evicted_tokens)
                        local_end_index = l_ + current_end - g - num_evicted_tokens
                    else:
                        local_end_index = l_ + current_end - g
                    local_start_index = local_end_index - num_new_tokens
                    k_wr = k[0] if k.dim() == 4 else k
                    v_wr = v[0] if v.dim() == 4 else v
                    action_keyboard_kv_cache.store_kv(k_wr, v_wr, local_start_index, local_end_index, li)
                    action_keyboard_kv_cache.set_ends(li, current_end, local_end_index)
                    attn_s = max(0, local_end_index - max_attention_size)
                    attn = self._infer_action_attn(
                        phase,
                        q,
                        action_keyboard_kv_cache.k_cache(li, attn_s, local_end_index).unsqueeze(0),
                        action_keyboard_kv_cache.v_cache(li, attn_s, local_end_index).unsqueeze(0),
                    )
                else:
                    attn = self._infer_action_attn(phase, q, k, v)
                attn = rearrange(attn, "B L H D -> B L (H D)")
            attn = phase.proj_keyboard.apply(attn[0]).unsqueeze(0)
            hidden_states = hidden_states + attn
            hidden_states = hidden_states.squeeze(0)

        return hidden_states

    def infer_ffn(self, phase, x, c_shift_msa, c_scale_msa):
        num_frames = c_shift_msa.shape[0]
        frame_seqlen = x.shape[0] // c_shift_msa.shape[0]

        x = phase.norm2.apply(x).unsqueeze(0)
        x = x.unflatten(dim=1, sizes=(num_frames, frame_seqlen))

        c_scale_msa = c_scale_msa.unsqueeze(0)
        c_shift_msa = c_shift_msa.unsqueeze(0)
        x = x * (1 + c_scale_msa) + c_shift_msa
        x = x.flatten(1, 2).squeeze(0)

        y = phase.ffn_0.apply(x)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = phase.ffn_2.apply(y)

        return y

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
        if hasattr(block.compute_phases[0], "before_proj"):
            x = block.compute_phases[0].before_proj.apply(x) + pre_infer_out.x

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
        )
        x, attn_out = self.infer_cross_attn_with_kvcache(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )
        x = x + attn_out
        if len(block.compute_phases) == 4:
            if self.config["mode"] != "templerun":
                x = self.infer_action_model(
                    phase=block.compute_phases[2],
                    x=x,
                    grid_sizes=pre_infer_out.grid_sizes.tensor[0],
                    mouse_condition=pre_infer_out.conditional_dict["mouse_cond"],
                    keyboard_condition=pre_infer_out.conditional_dict["keyboard_cond"],
                    is_causal=True,
                    use_rope_keyboard=True,
                )
            else:
                x = self.infer_action_model(
                    phase=block.compute_phases[2],
                    x=x,
                    grid_sizes=pre_infer_out.grid_sizes.tensor[0],
                    keyboard_condition=pre_infer_out.conditional_dict["keyboard_cond"],
                    is_causal=True,
                    use_rope_keyboard=True,
                )
            y = self.infer_ffn(block.compute_phases[3], x, c_shift_msa, c_scale_msa)

        elif len(block.compute_phases) == 3:
            y = self.infer_ffn(block.compute_phases[2], x, c_shift_msa, c_scale_msa)

        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        return x
