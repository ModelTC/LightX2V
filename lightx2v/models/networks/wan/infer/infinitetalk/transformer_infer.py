import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import GET_DTYPE


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    return normalized * (new_max - new_min) + new_min


class RotaryPositionalEmbedding1D:
    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.base = 10000

    def __call__(self, x, pos_indices):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=pos_indices.device).float() / self.head_dim))
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        cos = rearrange(freqs.cos().float(), "n d -> 1 1 n d").to(x.device)
        sin = rearrange(freqs.sin().float(), "n d -> 1 1 n d").to(x.device)
        x_float = x.float()
        return (x_float * cos + rotate_half(x_float) * sin).type_as(x)


class WanInfiniteTalkTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        offload_granularity = config.get("offload_granularity", "block")
        if config.get("cpu_offload", False) and offload_granularity not in {"block", "model"}:
            raise NotImplementedError(f"InfiniteTalk currently supports block/model offload, not {offload_granularity} offload.")
        super().__init__(config)
        self.phases_num = 4
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)
        self.audio_attn_cu_seqlens_q = None
        self.audio_attn_cu_seqlens_kv = None
        self.class_interval = config.get("infinitetalk_class_interval", 4)
        self.class_range = config.get("infinitetalk_class_range", 24)
        self.rope_h1 = (0, self.class_interval)
        self.rope_h2 = (self.class_range - self.class_interval, self.class_range)
        self.rope_bak = int(self.class_range // 2)

    def reset_infer_states(self):
        super().reset_infer_states()
        self.audio_attn_cu_seqlens_q = None
        self.audio_attn_cu_seqlens_kv = None

    def _seq_parallel_token_count(self, pre_infer_out):
        grid_t, grid_h, grid_w = pre_infer_out.grid_sizes.tuple
        return int(grid_t * grid_h * grid_w)

    def _seq_parallel_gather_tokens(self, x):
        gathered = [torch.empty_like(x) for _ in range(dist.get_world_size(self.seq_p_group))]
        dist.all_gather(gathered, x, group=self.seq_p_group)
        return torch.cat(gathered, dim=0)

    def _seq_parallel_chunk_tokens(self, x, local_len):
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        padded_len = local_len * world_size
        if x.shape[0] < padded_len:
            pad_shape = list(x.shape)
            pad_shape[0] = padded_len - x.shape[0]
            x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
        return torch.chunk(x, world_size, dim=0)[cur_rank]

    def infer_block(self, block, x, pre_infer_out):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        y_out, x_ref_attn_map = self.infer_self_attn(
            block.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            pre_infer_out,
        )
        x, attn_out = self.infer_cross_attn(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )
        x.add_(attn_out)
        if self.config["seq_parallel"]:
            local_len = x.shape[0]
            token_count = self._seq_parallel_token_count(pre_infer_out)
            full_x = self._seq_parallel_gather_tokens(x)[:token_count]
            audio_out = self.infer_audio_cross_attn(block.compute_phases[2], full_x, pre_infer_out, x_ref_attn_map)
            audio_out = self._seq_parallel_chunk_tokens(audio_out, local_len)
        else:
            audio_out = self.infer_audio_cross_attn(block.compute_phases[2], x, pre_infer_out, x_ref_attn_map)
        y = self.infer_ffn(block.compute_phases[3], x, audio_out, c_shift_msa, c_scale_msa, c_gate_msa)
        return self.post_process(x, y, c_gate_msa, pre_infer_out)

    def infer_self_attn(self, phase, x, shift_msa, scale_msa, pre_infer_out):
        cos_sin = self.cos_sin
        norm1_out = phase.norm1.apply(x)
        norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()
        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)
        q, k = self.apply_rope_func(q, k, cos_sin)

        x_ref_attn_map = None
        ref_target_masks = pre_infer_out.adapter_args.get("ref_target_masks")
        if pre_infer_out.adapter_args.get("human_num", 1) > 1 and ref_target_masks is not None:
            if self.config["seq_parallel"]:
                token_count = self._seq_parallel_token_count(pre_infer_out)
                map_q = self._seq_parallel_gather_tokens(q)[:token_count]
                map_k = self._seq_parallel_gather_tokens(k)[:token_count]
            else:
                map_q, map_k = q, k
            x_ref_attn_map = self._get_attn_map_with_target(map_q.unsqueeze(0), map_k.unsqueeze(0), pre_infer_out.grid_sizes.tuple, ref_target_masks)

        img_qkv_len = q.shape[0]
        if self.self_attn_cu_seqlens_qkv is None:
            self.self_attn_cu_seqlens_qkv = torch.tensor([0, q.shape[0]]).cumsum(0, dtype=torch.int32)
        attn_running_args = {
            "block_idx": self.block_idx,
            "scheduler": self.scheduler,
        }
        if self.config["seq_parallel"]:
            attn_out = phase.self_attn_1_parallel.apply(
                q=q,
                k=k,
                v=v,
                slice_qkv_len=img_qkv_len,
                cu_seqlens_qkv=self.self_attn_cu_seqlens_qkv,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                use_fp8_comm=self.seq_p_fp8_comm,
                use_fp4_comm=self.seq_p_fp4_comm,
                use_tensor_fusion=self.seq_p_tensor_fusion,
                enable_head_parallel=self.enable_head_parallel,
                **attn_running_args,
            )
        else:
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self.self_attn_cu_seqlens_qkv,
                cu_seqlens_kv=self.self_attn_cu_seqlens_qkv,
                max_seqlen_q=img_qkv_len,
                max_seqlen_kv=img_qkv_len,
                **attn_running_args,
            )
        y = phase.self_attn_o.apply(attn_out)
        return y, x_ref_attn_map

    def _get_attn_map_with_target(self, visual_q, ref_k, shape, ref_target_masks, split_num=2):
        _, grid_h, grid_w = shape
        ref_seqlen = grid_h * grid_w
        ref_k = ref_k[:, :ref_seqlen]
        _, seq_lens, heads, head_dim = visual_q.shape
        class_num, _ = ref_target_masks.shape
        x_ref_attn_maps = torch.zeros(class_num, seq_lens, device=visual_q.device, dtype=visual_q.dtype)
        split_chunk = max(1, heads // split_num)
        split_count = 0
        for start in range(0, heads, split_chunk):
            end = min(start + split_chunk, heads)
            maps = self._calculate_x_ref_attn_map(visual_q[:, :, start:end], ref_k[:, :, start:end], ref_target_masks, head_dim)
            x_ref_attn_maps += maps
            split_count += 1
        return x_ref_attn_maps / max(1, split_count)

    @staticmethod
    def _calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, head_dim):
        ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
        visual_q = (visual_q * (head_dim**-0.5)).transpose(1, 2)
        ref_k = ref_k.transpose(1, 2)
        attn = visual_q @ ref_k.transpose(-2, -1)
        attn = attn.softmax(-1)
        ref_target_masks = ref_target_masks.to(visual_q.dtype).to(visual_q.device)

        x_ref_attn_maps = []
        for ref_target_mask in ref_target_masks:
            mask = ref_target_mask[None, None, None, :]
            x_ref_attnmap = (attn * mask).sum(-1) / mask.sum().clamp_min(1.0)
            x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1).mean(-1)
            x_ref_attn_maps.append(x_ref_attnmap)
        return torch.concat(x_ref_attn_maps, dim=0)

    def infer_audio_cross_attn(self, phase, x, pre_infer_out, x_ref_attn_map):
        audio_embedding = pre_infer_out.adapter_args["audio_embedding"].to(device=x.device, dtype=GET_DTYPE())
        human_num = pre_infer_out.adapter_args.get("human_num", 1)
        grid_t, grid_h, grid_w = pre_infer_out.grid_sizes.tuple
        spatial_tokens = grid_h * grid_w

        x_norm = phase.norm_x.apply(x)
        x_frames = x_norm.view(grid_t, spatial_tokens, -1)

        q = phase.q_linear.apply(x_frames.reshape(grid_t * spatial_tokens, -1)).view(grid_t, spatial_tokens, self.num_heads, self.head_dim)
        audio_tokens = audio_embedding.shape[1]
        kv = phase.kv_linear.apply(audio_embedding.reshape(grid_t * audio_tokens, -1)).view(grid_t, audio_tokens, 2, self.num_heads, self.head_dim)
        encoder_k, encoder_v = kv.unbind(dim=2)

        if human_num > 1:
            q, encoder_k = self._apply_multi_human_audio_rope(q, encoder_k, x_ref_attn_map, grid_t)

        if self.audio_attn_cu_seqlens_q is None:
            self.audio_attn_cu_seqlens_q = torch.arange(0, (grid_t + 1) * spatial_tokens, spatial_tokens, dtype=torch.int32)
        if self.audio_attn_cu_seqlens_kv is None:
            self.audio_attn_cu_seqlens_kv = torch.arange(0, (grid_t + 1) * audio_tokens, audio_tokens, dtype=torch.int32)

        attn_out = phase.audio_attn.apply(
            q=q,
            k=encoder_k,
            v=encoder_v,
            cu_seqlens_q=self.audio_attn_cu_seqlens_q,
            cu_seqlens_kv=self.audio_attn_cu_seqlens_kv,
            max_seqlen_q=spatial_tokens,
            max_seqlen_kv=audio_tokens,
        )
        return phase.proj.apply(attn_out).view_as(x)

    def _apply_multi_human_audio_rope(self, q, encoder_k, x_ref_attn_map, grid_t):
        if x_ref_attn_map is None:
            return q, encoder_k

        max_values = x_ref_attn_map.max(1).values[:, None, None]
        min_values = x_ref_attn_map.min(1).values[:, None, None]
        max_min_values = torch.cat([max_values, min_values], dim=2)
        human1_max_value, human1_min_value = max_min_values[0, :, 0].max(), max_min_values[0, :, 1].min()
        human2_max_value, human2_min_value = max_min_values[1, :, 0].max(), max_min_values[1, :, 1].min()

        human1 = normalize_and_scale(x_ref_attn_map[0], (human1_min_value, human1_max_value), (self.rope_h1[0], self.rope_h1[1]))
        human2 = normalize_and_scale(x_ref_attn_map[1], (human2_min_value, human2_max_value), (self.rope_h2[0], self.rope_h2[1]))
        back = torch.full((x_ref_attn_map.size(1),), self.rope_bak, dtype=human1.dtype, device=human1.device)
        max_indices = x_ref_attn_map.argmax(dim=0)
        normalized_map = torch.stack([human1, human2, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices]

        q_rope = rearrange(q, "t s h d -> 1 h (t s) d")
        q_rope = self.rope_1d(q_rope, normalized_pos)
        q = rearrange(q_rope, "1 h (t s) d -> t s h d", t=grid_t)

        audio_tokens = encoder_k.shape[1]
        per_frame = torch.zeros(audio_tokens, dtype=encoder_k.dtype, device=encoder_k.device)
        per_frame[: audio_tokens // 2] = (self.rope_h1[0] + self.rope_h1[1]) / 2
        per_frame[audio_tokens // 2 :] = (self.rope_h2[0] + self.rope_h2[1]) / 2
        encoder_pos = torch.concat([per_frame] * grid_t, dim=0)
        k_rope = rearrange(encoder_k, "t s h d -> 1 h (t s) d")
        k_rope = self.rope_1d(k_rope, encoder_pos)
        encoder_k = rearrange(k_rope, "1 h (t s) d -> t s h d", t=grid_t)
        return q, encoder_k
