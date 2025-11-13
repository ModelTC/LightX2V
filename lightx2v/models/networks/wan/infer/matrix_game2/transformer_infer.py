import math

import torch
from einops import rearrange

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    from flash_attn import flash_attn_func

    FLASH_ATTN_3_AVAILABLE = False


from lightx2v.models.networks.wan.infer.matrix_game2.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import WanSFTransformerInfer, causal_rope_apply


class WanMtxg2TransformerInfer(WanSFTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self._initialize_kv_cache_mouse_and_keyboard(self.device, self.dtype)
        self.sink_size = 0
        self.vae_time_compression_ratio = config["action_config"]["vae_time_compression_ratio"]
        self.windows_size = config["action_config"]["windows_size"]
        self.patch_size = config["action_config"]["patch_size"]

        self.rope_theta = config["action_config"]["rope_theta"]
        self.enable_keyboard = config["action_config"]["enable_keyboard"]
        self.heads_num = config["action_config"]["heads_num"]
        self.hidden_size = config["action_config"]["hidden_size"]
        self.img_hidden_size = config["action_config"]["img_hidden_size"]
        self.keyboard_dim_in = config["action_config"]["keyboard_dim_in"]
        self.keyboard_hidden_dim = config["action_config"]["keyboard_hidden_dim"]

        self.qk_norm = config["action_config"]["qk_norm"]
        self.qkv_bias = config["action_config"]["qkv_bias"]
        self.rope_dim_list = config["action_config"]["rope_dim_list"]
        self.freqs_cos, self.freqs_sin = self.get_rotary_pos_embed(7500, self.patch_size[1], self.patch_size[2], 64, self.rope_dim_list, start_offset=0)

        self.enable_mouse = config["action_config"]["enable_mouse"]
        if self.enable_mouse:
            self.mouse_dim_in = config["action_config"]["mouse_dim_in"]
            self.mouse_hidden_dim = config["action_config"]["mouse_hidden_dim"]
            self.mouse_qk_dim_list = config["action_config"]["mouse_qk_dim_list"]

    def get_rotary_pos_embed(self, video_length, height, width, head_dim, rope_dim_list=None, start_offset=0):
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
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

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

    def _initialize_kv_cache(self, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros((kv_cache_size, 12, 128)).to(dtype).to(device),
                    "v": torch.zeros((kv_cache_size, 12, 128)).to(dtype).to(device),
                    "global_end_index": 0,
                    "local_end_index": 0,
                }
            )

        self.kv_cache1_default = kv_cache1

    def _initialize_kv_cache_mouse_and_keyboard(self, device, dtype):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append(
                {
                    "k": torch.zeros([1, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([1, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
            kv_cache_mouse.append(
                {
                    "k": torch.zeros([self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "v": torch.zeros([self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        self.kv_cache_keyboard = kv_cache_keyboard
        self.kv_cache_mouse = kv_cache_mouse

    def infer_self_attn_with_kvcache(self, phase, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa):
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa.squeeze()
            norm1_bias = shift_msa.squeeze()

        norm1_out = phase.norm1.apply(x)

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.sensitive_layer_dtype)

        norm1_out.mul_(norm1_weight[0:1, :]).add_(norm1_bias[0:1, :])

        if self.sensitive_layer_dtype != self.infer_dtype:  # False
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim

        q0 = phase.self_attn_q.apply(norm1_out)
        k0 = phase.self_attn_k.apply(norm1_out)

        q = phase.self_attn_norm_q.apply(q0).view(s, n, d)
        k = phase.self_attn_norm_k.apply(k0).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        seg_index = self.scheduler.seg_index

        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        current_start = seg_index * self.num_frame_per_block * self.frame_seq_length
        current_start_frame = current_start // frame_seqlen

        q = causal_rope_apply(q.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
        k = causal_rope_apply(k.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]

        current_end = current_start + q.shape[0]
        sink_tokens = self.sink_size * frame_seqlen

        kv_cache_size = self.kv_cache1[self.block_idx]["k"].shape[0]
        num_new_tokens = q.shape[0]

        if (current_end > self.kv_cache1[self.block_idx]["global_end_index"]) and (num_new_tokens + self.kv_cache1[self.block_idx]["local_end_index"] > kv_cache_size):
            num_evicted_tokens = num_new_tokens + self.kv_cache1[self.block_idx]["local_end_index"] - kv_cache_size
            num_rolled_tokens = self.kv_cache1[self.block_idx]["local_end_index"] - num_evicted_tokens - sink_tokens

            self.kv_cache1[self.block_idx]["k"][sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache1[self.block_idx]["k"][
                sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            self.kv_cache1[self.block_idx]["v"][sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache1[self.block_idx]["v"][
                sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()

            # Insert the new keys/values at the end
            local_end_index = self.kv_cache1[self.block_idx]["local_end_index"] + current_end - self.kv_cache1[self.block_idx]["global_end_index"] - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            self.kv_cache1[self.block_idx]["k"][local_start_index:local_end_index] = k
            self.kv_cache1[self.block_idx]["v"][local_start_index:local_end_index] = v
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = self.kv_cache1[self.block_idx]["local_end_index"] + current_end - self.kv_cache1[self.block_idx]["global_end_index"]
            local_start_index = local_end_index - num_new_tokens
            self.kv_cache1[self.block_idx]["k"][local_start_index:local_end_index] = k
            self.kv_cache1[self.block_idx]["v"][local_start_index:local_end_index] = v

        attn_k = self.kv_cache1[self.block_idx]["k"][max(0, local_end_index - self.max_attention_size) : local_end_index]
        attn_v = self.kv_cache1[self.block_idx]["v"][max(0, local_end_index - self.max_attention_size) : local_end_index]

        self.kv_cache1[self.block_idx]["local_end_index"] = local_end_index
        self.kv_cache1[self.block_idx]["global_end_index"] = current_end

        k_lens = torch.empty_like(seq_lens).fill_(attn_k.size(0))
        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)

        if self.clean_cuda_cache:
            del freqs_i, norm1_out, norm1_weight, norm1_bias
            torch.cuda.empty_cache()

        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                model_cls=self.config["model_cls"],
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=attn_k.size(0),
                model_cls=self.config["model_cls"],
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch.cuda.empty_cache()

        return y

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa):
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // gate_msa.shape[0]

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))
        norm3_out = phase.norm3.apply(x)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_q.apply(norm3_out)
        q = phase.cross_attn_norm_q.apply(q).view(-1, n, d)

        if not self.crossattn_cache[self.block_idx]["is_init"]:
            self.crossattn_cache[self.block_idx]["is_init"] = True
            k = phase.cross_attn_k.apply(context)
            k = phase.cross_attn_norm_k.apply(k).view(-1, n, d)
            v = phase.cross_attn_v.apply(context)
            v = v.view(-1, n, d)
            self.crossattn_cache[self.block_idx]["k"] = k
            self.crossattn_cache[self.block_idx]["v"] = v
        else:
            k = self.crossattn_cache[self.block_idx]["k"]
            v = self.crossattn_cache[self.block_idx]["v"]

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )

        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=self.config["model_cls"],
        )

        attn_out = phase.cross_attn_o.apply(attn_out)
        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch.cuda.empty_cache()

        return x, attn_out

    def infer_action_model(self, phase, x, grid_sizes, seq_lens, mouse_condition=None, keyboard_condition=None, is_causal=False, use_rope_keyboard=True):
        tt, th, tw = grid_sizes
        current_start = self.scheduler.seg_index * self.num_frame_per_block
        start_frame = current_start
        B, N_frames, C = keyboard_condition.shape
        assert tt * th * tw == x.shape[0]
        assert ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1

        # Defined freqs_cis early so it's available for both mouse and keyboard
        freqs_cis = (self.freqs_cos, self.freqs_sin)

        cond1 = N_feats == tt
        cond2 = is_causal and not self.kv_cache_mouse
        cond3 = (N_frames - 1) // self.vae_time_compression_ratio + 1 == current_start + self.num_frame_per_block
        assert (cond1 and ((cond2) or not is_causal)) or (cond3 and is_causal)

        x = x.unsqueeze(0)
        if self.enable_mouse and mouse_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th * tw)  # 65*272*480 -> 17*(272//16)*(480//16) -> 8670
            B, N_frames, C = mouse_condition.shape
        else:
            hidden_states = x

        pad_t = self.vae_time_compression_ratio * self.windows_size
        if self.enable_mouse and mouse_condition is not None:
            pad = mouse_condition[:, 0:1, :].expand(-1, pad_t, -1)
            mouse_condition = torch.cat([pad, mouse_condition], dim=1)
            if is_causal and self.kv_cache_mouse is not None:
                mouse_condition = mouse_condition[:, self.vae_time_compression_ratio * (N_feats - self.num_frame_per_block - self.windows_size) + pad_t :, :]
                group_mouse = [
                    mouse_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(self.num_frame_per_block)
                ]
            else:
                group_mouse = [mouse_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]

            group_mouse = torch.stack(group_mouse, dim=1)

            S = th * tw
            group_mouse = group_mouse.unsqueeze(-1).expand(B, self.num_frame_per_block, pad_t, C, S)
            group_mouse = group_mouse.permute(0, 4, 1, 2, 3).reshape(B * S, self.num_frame_per_block, pad_t * C)

            group_mouse = torch.cat([hidden_states, group_mouse], dim=-1)

            # mouse_mlp
            # 注释：Batch维度不可避免，因此用 torch.nn.functional
            group_mouse = torch.nn.functional.linear(group_mouse, phase.mouse_mlp_0.weight.T, phase.mouse_mlp_0.bias)
            group_mouse = torch.nn.functional.gelu(group_mouse, approximate="tanh")
            group_mouse = torch.nn.functional.linear(group_mouse, phase.mouse_mlp_2.weight.T, phase.mouse_mlp_2.bias)
            group_mouse = torch.nn.functional.layer_norm(group_mouse, (group_mouse.shape[-1],), phase.mouse_mlp_3.weight.T, phase.mouse_mlp_3.bias, 1e-5)

            # qkvc
            mouse_qkv = torch.nn.functional.linear(group_mouse, phase.t_qkv.weight.T)

            q0, k0, v = rearrange(mouse_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)  # BHW F  H C # torch.Size([880, 3, 16, 64])
            q = q0 * torch.rsqrt(q0.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k0 * torch.rsqrt(k0.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

            q, k = apply_rotary_emb(q, k, freqs_cis, start_offset=start_frame, head_first=False)

            ## TODO: adding cache here
            if is_causal:
                current_end = current_start + q.shape[1]

                assert q.shape[1] == self.num_frame_per_block
                sink_size = 0
                max_attention_size = self.local_attn_size
                sink_tokens = sink_size * 1
                kv_cache_size = self.kv_cache_mouse[self.block_idx]["k"].shape[1]
                num_new_tokens = q.shape[1]

                if (current_end > self.kv_cache_mouse[self.block_idx]["global_end_index"].item()) and (num_new_tokens + self.kv_cache_mouse[self.block_idx]["local_end_index"].item() > kv_cache_size):
                    num_evicted_tokens = num_new_tokens + self.kv_cache_mouse[self.block_idx]["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = self.kv_cache_mouse[self.block_idx]["local_end_index"].item() - num_evicted_tokens - sink_tokens
                    self.kv_cache_mouse[self.block_idx]["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_mouse[self.block_idx]["k"][
                        :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                    ].clone()
                    self.kv_cache_mouse[self.block_idx]["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_mouse[self.block_idx]["v"][
                        :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                    ].clone()
                    # Insert the new keys/values at the end
                    local_end_index = self.kv_cache_mouse[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_mouse[self.block_idx]["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                else:
                    local_end_index = self.kv_cache_mouse[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_mouse[self.block_idx]["global_end_index"].item()
                    local_start_index = local_end_index - num_new_tokens

                self.kv_cache_mouse[self.block_idx]["k"][:, local_start_index:local_end_index] = k
                self.kv_cache_mouse[self.block_idx]["v"][:, local_start_index:local_end_index] = v

                attn_k = self.kv_cache_mouse[self.block_idx]["k"][:, max(0, local_end_index - max_attention_size) : local_end_index]
                attn_v = self.kv_cache_mouse[self.block_idx]["v"][:, max(0, local_end_index - max_attention_size) : local_end_index]

                attn = flash_attn_interface.flash_attn_func(
                    q,
                    attn_k,
                    attn_v,
                )

                self.kv_cache_mouse[self.block_idx]["global_end_index"].fill_(current_end)
                self.kv_cache_mouse[self.block_idx]["local_end_index"].fill_(local_end_index)
            else:
                attn = flash_attn_func(
                    q,
                    k,
                    v,
                )
            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            attn = rearrange(attn, "(b S) T h d -> b (T S) (h d)", b=B)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B)

            attn = phase.proj_mouse.apply(attn[0]).unsqueeze(0)
            hidden_states = hidden_states + attn

        if self.enable_keyboard and keyboard_condition is not None:
            pad = keyboard_condition[:, 0:1, :].expand(-1, pad_t, -1)
            keyboard_condition = torch.cat([pad, keyboard_condition], dim=1)
            if is_causal and self.kv_cache_keyboard is not None:
                keyboard_condition = keyboard_condition[
                    :, self.vae_time_compression_ratio * (N_feats - self.num_frame_per_block - self.windows_size) + pad_t :, :
                ]  # keyboard_condition[:, self.vae_time_compression_ratio*(start_frame - self.windows_size) + pad_t:start_frame * self.vae_time_compression_ratio + pad_t,:]

                keyboard_condition = phase.keyboard_embed_0.apply(keyboard_condition[0])
                keyboard_condition = torch.nn.functional.silu(keyboard_condition)
                keyboard_condition = phase.keyboard_embed_2.apply(keyboard_condition).unsqueeze(0)
                group_keyboard = [
                    keyboard_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(self.num_frame_per_block)
                ]
            else:
                keyboard_condition = phase.keyboard_embed_0.apply(keyboard_condition[0])
                keyboard_condition = torch.nn.functional.silu(keyboard_condition)
                keyboard_condition = phase.keyboard_embed_2.apply(keyboard_condition).unsqueeze(0)
                group_keyboard = [keyboard_condition[:, self.vae_time_compression_ratio * (i - self.windows_size) + pad_t : i * self.vae_time_compression_ratio + pad_t, :] for i in range(N_feats)]

            group_keyboard = torch.stack(group_keyboard, dim=1)  # B F RW C
            group_keyboard = group_keyboard.reshape(shape=(group_keyboard.shape[0], group_keyboard.shape[1], -1))

            # apply cross attn
            mouse_q = phase.mouse_attn_q.apply(hidden_states[0]).unsqueeze(0)
            keyboard_kv = phase.keyboard_attn_kv.apply(group_keyboard[0]).unsqueeze(0)

            B, L, HD = mouse_q.shape
            D = HD // self.heads_num
            q = mouse_q.view(B, L, self.heads_num, D)

            B, L, KHD = keyboard_kv.shape
            k, v = keyboard_kv.view(B, L, 2, self.heads_num, D).permute(2, 0, 1, 3, 4)

            # Compute cu_squlens and max_seqlen for flash attention
            # qk norm
            q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

            S = th * tw
            assert S == 880
            # position embed
            if use_rope_keyboard:
                B, TS, H, D = q.shape
                T_ = TS // S
                q = q.view(B, T_, S, H, D).transpose(1, 2).reshape(B * S, T_, H, D)
                q, k = apply_rotary_emb(q, k, freqs_cis, start_offset=start_frame, head_first=False)

                k1, k2, k3, k4 = k.shape
                k = k.expand(S, k2, k3, k4)
                v = v.expand(S, k2, k3, k4)

                if is_causal:
                    current_end = current_start + k.shape[1]
                    assert k.shape[1] == self.num_frame_per_block
                    sink_size = 0
                    max_attention_size = self.local_attn_size
                    sink_tokens = sink_size * 1
                    kv_cache_size = self.kv_cache_keyboard[self.block_idx]["k"].shape[1]
                    num_new_tokens = k.shape[1]

                    if (current_end > self.kv_cache_keyboard[self.block_idx]["global_end_index"].item()) and (
                        num_new_tokens + self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() > kv_cache_size
                    ):
                        num_evicted_tokens = num_new_tokens + self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() - kv_cache_size
                        num_rolled_tokens = self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() - num_evicted_tokens - sink_tokens

                        self.kv_cache_keyboard[self.block_idx]["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_keyboard[self.block_idx]["k"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()
                        self.kv_cache_keyboard[self.block_idx]["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_keyboard[self.block_idx]["v"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()

                        # Insert the new keys/values at the end
                        local_end_index = (
                            self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_keyboard[self.block_idx]["global_end_index"].item() - num_evicted_tokens
                        )
                        local_start_index = local_end_index - num_new_tokens
                    else:
                        local_end_index = self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_keyboard[self.block_idx]["global_end_index"].item()
                        local_start_index = local_end_index - num_new_tokens

                    assert k.shape[0] == 880  # BS == 1 or the cache should not be saved/ load method should be modified
                    self.kv_cache_keyboard[self.block_idx]["k"][:, local_start_index:local_end_index] = k[:1]
                    self.kv_cache_keyboard[self.block_idx]["v"][:, local_start_index:local_end_index] = v[:1]

                    if FLASH_ATTN_3_AVAILABLE:
                        attn_k = self.kv_cache_keyboard[self.block_idx]["k"][:, max(0, local_end_index - max_attention_size) : local_end_index].repeat(S, 1, 1, 1)
                        attn_v = self.kv_cache_keyboard[self.block_idx]["v"][:, max(0, local_end_index - max_attention_size) : local_end_index].repeat(S, 1, 1, 1)
                        attn = flash_attn_interface.flash_attn_func(
                            q,
                            attn_k,
                            attn_v,
                        )
                    else:
                        attn = flash_attn_func(
                            q,
                            self.kv_cache_keyboard[self.block_idx]["k"][max(0, local_end_index - max_attention_size) : local_end_index].repeat(S, 1, 1, 1),
                            self.kv_cache_keyboard[self.block_idx]["v"][max(0, local_end_index - max_attention_size) : local_end_index].repeat(S, 1, 1, 1),
                        )

                    self.kv_cache_keyboard[self.block_idx]["global_end_index"].fill_(current_end)
                    self.kv_cache_keyboard[self.block_idx]["local_end_index"].fill_(local_end_index)
                else:
                    attn = flash_attn_func(
                        q,
                        k,
                        v,
                        causal=False,
                    )
                attn = rearrange(attn, "(B S) T H D -> B (T S) (H D)", S=S)

            else:
                if is_causal:
                    current_start = start_frame
                    current_end = current_start + k.shape[1]
                    assert k.shape[1] == self.num_frame_per_block
                    sink_size = 0
                    local_attn_size = self.local_attn_size
                    max_attention_size = self.local_attn_size
                    sink_tokens = sink_size * 1
                    kv_cache_size = self.kv_cache_keyboard[self.block_idx]["k"].shape[1]
                    num_new_tokens = k.shape[1]

                    if (current_end > self.kv_cache_keyboard[self.block_idx]["global_end_index"].item()) and (
                        num_new_tokens + self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() > kv_cache_size
                    ):
                        num_evicted_tokens = num_new_tokens + self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() - kv_cache_size
                        num_rolled_tokens = self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() - num_evicted_tokens - sink_tokens
                        self.kv_cache_keyboard[self.block_idx]["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_keyboard[self.block_idx]["k"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()
                        self.kv_cache_keyboard[self.block_idx]["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = self.kv_cache_keyboard[self.block_idx]["v"][
                            :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                        ].clone()
                        # Insert the new keys/values at the end
                        local_end_index = (
                            self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_keyboard[self.block_idx]["global_end_index"].item() - num_evicted_tokens
                        )
                        local_start_index = local_end_index - num_new_tokens
                    else:
                        local_end_index = self.kv_cache_keyboard[self.block_idx]["local_end_index"].item() + current_end - self.kv_cache_keyboard[self.block_idx]["global_end_index"].item()
                        local_start_index = local_end_index - num_new_tokens
                    self.kv_cache_keyboard[self.block_idx]["k"][:, local_start_index:local_end_index] = k
                    self.kv_cache_keyboard[self.block_idx]["v"][:, local_start_index:local_end_index] = v
                    attn = flash_attn_func(
                        q,
                        self.kv_cache_keyboard[self.block_idx]["k"][:, max(0, local_end_index - max_attention_size) : local_end_index],
                        self.kv_cache_keyboard[self.block_idx]["v"][:, max(0, local_end_index - max_attention_size) : local_end_index],
                    )
                    self.kv_cache_keyboard[self.block_idx]["global_end_index"].fill_(current_end)
                    self.kv_cache_keyboard[self.block_idx]["local_end_index"].fill_(local_end_index)
                else:
                    attn = flash_attn_func(
                        q,
                        k,
                        v,
                    )
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

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        x = x + y * c_gate_msa[0]
        x = x.squeeze(0)
        return x

    def infer_block_witch_kvcache(self, block, x, pre_infer_out):
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
                    seq_lens=pre_infer_out.seq_lens,
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
                    seq_lens=pre_infer_out.seq_lens,
                    keyboard_condition=pre_infer_out.conditional_dict["keyboard_cond"],
                    is_causal=True,
                    use_rope_keyboard=True,
                )
            y = self.infer_ffn(block.compute_phases[3], x, c_shift_msa, c_scale_msa)

        elif len(block.compute_phases) == 3:
            y = self.infer_ffn(block.compute_phases[2], x, c_shift_msa, c_scale_msa)

        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

        return x

    def infer_non_blocks(self, weights, x, e):
        num_frames = e.shape[0]
        frame_seqlen = x.shape[0] // e.shape[0]
        e = e.unsqueeze(0).unsqueeze(2)

        x = weights.norm.apply(x).unsqueeze(0)
        x = x.unflatten(dim=1, sizes=(num_frames, frame_seqlen))

        modulation = weights.head_modulation.tensor
        e = (modulation.unsqueeze(1) + e).chunk(2, dim=2)

        x = x * (1 + e[1]) + e[0]

        x = torch.nn.functional.linear(x, weights.head.weight.T, weights.head.bias)

        if self.clean_cuda_cache:
            del e
            torch.cuda.empty_cache()

        return x
