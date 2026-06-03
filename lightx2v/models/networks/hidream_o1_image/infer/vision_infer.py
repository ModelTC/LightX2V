import torch
from transformers.activations import ACT2FN


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class HidreamO1ImageVisionInfer:
    def __init__(self, config):
        self.config = config

    def infer(self, weights, pixel_values, grid_thw):
        pixel_values = pixel_values.to(weights.patch_embed.weight.dtype)
        hidden_states = pixel_values.view(
            -1,
            weights.in_channels,
            weights.temporal_patch_size,
            weights.patch_size,
            weights.patch_size,
        )
        hidden_states = weights.patch_embed.apply(hidden_states).view(-1, weights.hidden_size)
        hidden_states = hidden_states + self._fast_pos_embed_interpolate(weights, grid_thw)

        rotary_pos_emb = self._rot_pos_emb(weights, grid_thw)
        seq_len = hidden_states.shape[0]
        emb = torch.cat((rotary_pos_emb.reshape(seq_len, -1), rotary_pos_emb.reshape(seq_len, -1)), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(weights.blocks):
            hidden_states = self._infer_block(block, hidden_states, cu_seqlens, position_embeddings)
            if layer_num in weights.deepstack_visual_indexes:
                merger_idx = weights.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(self._infer_merger(weights.deepstack_merger_list[merger_idx], hidden_states))

        hidden_states = self._infer_merger(weights.merger, hidden_states)
        return [hidden_states], deepstack_feature_lists

    def _rotary_embedding(self, seqlen, dim, device):
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)

    def _rot_pos_emb(self, weights, grid_thw):
        merge_size = weights.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self._rotary_embedding(max_hw, weights.head_dim // 2, grid_thw.device)
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=grid_thw.device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=grid_thw.device)
            block_cols = torch.arange(merged_w, device=grid_thw.device)
            intra_row = torch.arange(merge_size, device=grid_thw.device)
            intra_col = torch.arange(merge_size, device=grid_thw.device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        return freq_table[pos_ids].flatten(1)

    def _fast_pos_embed_interpolate(self, weights, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, weights.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, weights.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=weights.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=weights.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * weights.num_grid_per_side
            base_h_ceil = h_ceil * weights.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            interp_weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(interp_weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weights.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=weights.pos_embed.weight.dtype, device=weights.pos_embed.weight.device)
        pos_embeds = weights.pos_embed.apply(idx_tensor.reshape(-1)).reshape(4, -1, weights.hidden_size)
        patch_pos_embeds = (pos_embeds * weight_tensor[:, :, None]).sum(dim=0)
        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])

        out = []
        merge_size = weights.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            out.append(pos_embed)
        return torch.cat(out, dim=0)

    def _infer_block(self, weights, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self._infer_attn(weights, weights.norm1.apply(hidden_states), cu_seqlens, position_embeddings)
        hidden_states = hidden_states + self._infer_mlp(weights, weights.norm2.apply(hidden_states))
        return hidden_states

    def _infer_attn(self, weights, hidden_states, cu_seqlens, position_embeddings):
        seq_len = hidden_states.shape[0]
        qkv = weights.qkv.apply(hidden_states).reshape(seq_len, 3, weights.num_heads, weights.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = []
        start = 0
        for length in lengths:
            length = int(length)
            end = start + length
            outs.append(weights.attn.apply(q[start:end], k[start:end], v[start:end], causal=False, max_seqlen_q=length, max_seqlen_kv=length, model_cls="hidream_o1_image"))
            start = end
        return weights.proj.apply(torch.cat(outs, dim=0).reshape(seq_len, -1))

    def _infer_mlp(self, weights, hidden_states):
        act_fn = ACT2FN[weights.act_fn_name]
        return weights.linear_fc2.apply(act_fn(weights.linear_fc1.apply(hidden_states)))

    def _infer_merger(self, weights, hidden_states):
        if weights.use_postshuffle_norm:
            hidden_states = weights.norm.apply(hidden_states.view(-1, weights.hidden_size)).view(-1, weights.hidden_size)
        else:
            hidden_states = weights.norm.apply(hidden_states).view(-1, weights.hidden_size)
        hidden_states = torch.nn.functional.gelu(weights.linear_fc1.apply(hidden_states), approximate="none")
        return weights.linear_fc2.apply(hidden_states)
