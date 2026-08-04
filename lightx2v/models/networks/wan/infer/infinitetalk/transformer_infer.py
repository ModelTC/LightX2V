import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import GET_DTYPE

from .rope import RotaryPositionalEmbedding1D


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    source_min, source_max = source_range
    new_min, new_max = target_range
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    return normalized * (new_max - new_min) + new_min


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

    def reset_infer_states(self, x, context):
        super().reset_infer_states(x, context)
        self.audio_attn_cu_seqlens_q = None
        self.audio_attn_cu_seqlens_kv = None

    def _seq_parallel_token_count(self, pre_infer_out):
        grid_t, grid_h, grid_w = pre_infer_out.grid_sizes.tuple
        return int(grid_t * grid_h * grid_w)

    def _seq_parallel_gather_tokens(self, x):
        # Reference attention is evaluated before Ulysses exchanges sequence
        # shards for head shards. Gather only the token dimension here and
        # keep every attention head resident on every SP rank.
        x = x.contiguous()
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
        if self.rope_positions is None:
            q, k = phase.rope.apply(q, k, cos_sin)
        else:
            q, k = phase.rope.apply(q, k, cos_sin, positions=self.rope_positions)

        x_ref_attn_map = None
        ref_target_masks = pre_infer_out.adapter_args.get("ref_target_masks")
        if pre_infer_out.adapter_args.get("human_num", 1) > 1 and ref_target_masks is not None:
            if self.config["seq_parallel"]:
                token_count = self._seq_parallel_token_count(pre_infer_out)
                map_q = self._seq_parallel_gather_tokens(q)[:token_count]
                map_k = self._seq_parallel_gather_tokens(k)[:token_count]
            else:
                map_q, map_k = q, k
            if map_q.shape[1] != self.num_heads or map_k.shape[1] != self.num_heads:
                raise RuntimeError(f"InfiniteTalk reference attention requires all heads on every SP rank; expected {self.num_heads}, got q={map_q.shape[1]} and k={map_k.shape[1]}.")
            x_ref_attn_map = self._get_attn_map_with_target(map_q.unsqueeze(0), map_k.unsqueeze(0), pre_infer_out.grid_sizes.tuple, ref_target_masks)

        img_qkv_len = q.shape[0]
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
                enable_head_parallel=self.seq_p_head_parallel,
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

    def _get_attn_map_with_target(self, visual_q, ref_k, shape, ref_target_masks):
        _, grid_h, grid_w = shape
        ref_seqlen = grid_h * grid_w
        device = visual_q.device
        visual_q = visual_q.to(dtype=GET_DTYPE())
        ref_k = ref_k[:, :ref_seqlen].to(device=device, dtype=GET_DTYPE())
        ref_target_masks = ref_target_masks.to(device=device, dtype=GET_DTYPE())
        _, seq_lens, heads, head_dim = visual_q.shape
        class_num, _ = ref_target_masks.shape
        x_ref_attn_maps = torch.zeros(class_num, seq_lens, device=device, dtype=GET_DTYPE())
        # InfiniteTalk's reference implementation always uses two equal head
        # groups. This is a local, sequential memory split and is unrelated to
        # the Ulysses SP world size; never shard these heads across SP ranks.
        split_num = 2
        if heads % split_num != 0:
            raise ValueError(f"Reference attention heads ({heads}) must be divisible by the fixed split_num={split_num}.")
        split_chunk = heads // split_num
        for split_idx in range(split_num):
            start = split_idx * split_chunk
            end = (split_idx + 1) * split_chunk
            maps = self._calculate_x_ref_attn_map(visual_q[:, :, start:end], ref_k[:, :, start:end], ref_target_masks, head_dim)
            x_ref_attn_maps += maps
        return x_ref_attn_maps / split_num

    @staticmethod
    def _calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, head_dim):
        scale = visual_q.new_tensor(head_dim**-0.5)
        visual_q = (visual_q * scale).transpose(1, 2).contiguous()
        ref_k = ref_k.transpose(1, 2).contiguous()

        batch_size, heads, ref_seqlen, value_dim = ref_k.shape
        class_num, mask_seqlen = ref_target_masks.shape
        if mask_seqlen != ref_seqlen:
            raise ValueError(f"Reference mask length ({mask_seqlen}) does not match reference K length ({ref_seqlen}).")
        if class_num > value_dim:
            raise ValueError(f"Reference mask count ({class_num}) exceeds attention head dimension ({value_dim}).")

        # The required map is softmax(QK^T) @ mask. Use each target mask as
        # a value channel so fused SDPA can compute it without materializing
        # the B x H x query_len x ref_len attention matrix. Pad V to the Q/K
        # head dimension for compatibility with fused CUDA kernels.
        mask_values = ref_k.new_zeros((batch_size, heads, ref_seqlen, value_dim))
        mask_values[..., :class_num] = ref_target_masks.transpose(0, 1)[None, None, :, :]

        # Q was scaled above to match InfiniteTalk's operation order, so SDPA
        # must not apply the default head_dim**-0.5 scale a second time. Keep
        # the math backend disabled: it may materialize the full attention map.
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            mask_attn = F.scaled_dot_product_attention(
                visual_q,
                ref_k,
                mask_values,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0,
            )

        mask_attn = mask_attn[..., :class_num]
        mask_sums = ref_target_masks.sum(-1).clamp_min(1.0)
        mask_attn = mask_attn / mask_sums[None, None, None, :]
        mask_attn = mask_attn.mean(dim=1)  # B, query_len, class_num
        return mask_attn.permute(2, 0, 1).reshape(class_num * batch_size, -1)

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
            q, encoder_k = self._apply_multi_human_audio_rope(
                phase,
                q,
                encoder_k,
                x_ref_attn_map,
                grid_t,
                human_num,
                pre_infer_out,
            )

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

    def _multi_human_rope_ranges(self, human_num, dtype, device):
        h1 = torch.tensor(self.rope_h1, dtype=dtype, device=device)
        h2 = torch.tensor(self.rope_h2, dtype=dtype, device=device)
        if human_num == 2:
            return torch.stack([h1, h2], dim=0)
        starts = torch.linspace(h1[0], h2[0], human_num, dtype=dtype, device=device)
        ends = torch.linspace(h1[1], h2[1], human_num, dtype=dtype, device=device)
        return torch.stack([starts, ends], dim=1)

    def _apply_multi_human_audio_rope(
        self,
        phase,
        q,
        encoder_k,
        x_ref_attn_map,
        grid_t,
        human_num,
        pre_infer_out,
    ):
        if x_ref_attn_map is None:
            return q, encoder_k

        human_map_count = min(int(human_num), max(0, x_ref_attn_map.shape[0] - 1))
        if human_map_count <= 0:
            return q, encoder_k

        rope_ranges = pre_infer_out.adapter_args.get("audio_rope_ranges")
        if rope_ranges is None:
            rope_ranges = self._multi_human_rope_ranges(human_map_count, x_ref_attn_map.dtype, x_ref_attn_map.device)
        human_positions = []
        for idx in range(human_map_count):
            human_map = x_ref_attn_map[idx]
            human_min_value = human_map.min()
            human_max_value = human_map.max()
            human_positions.append(
                normalize_and_scale(
                    human_map,
                    (human_min_value, human_max_value),
                    (rope_ranges[idx, 0], rope_ranges[idx, 1]),
                )
            )

        back = torch.full(
            (x_ref_attn_map.size(1),),
            self.rope_bak,
            dtype=x_ref_attn_map.dtype,
            device=x_ref_attn_map.device,
        )
        max_indices = x_ref_attn_map.argmax(dim=0)
        max_indices = torch.clamp(max_indices, max=human_map_count)
        normalized_map = torch.stack([*human_positions, back], dim=1)
        normalized_pos = normalized_map[range(x_ref_attn_map.size(1)), max_indices]

        q_rope = rearrange(q, "t s h d -> 1 h (t s) d")
        q_rope = self.rope_1d(phase.rope_1d, q_rope, normalized_pos)
        q = rearrange(q_rope, "1 h (t s) d -> t s h d", t=grid_t)

        k_rope = rearrange(encoder_k, "t s h d -> 1 h (t s) d")
        encoder_rope = pre_infer_out.adapter_args.get("audio_encoder_rope")
        if encoder_rope is None:
            audio_tokens = encoder_k.shape[1]
            per_frame = torch.zeros(audio_tokens, dtype=encoder_k.dtype, device=encoder_k.device)
            token_edges = torch.linspace(0, audio_tokens, human_map_count + 1, dtype=torch.int64, device=encoder_k.device)
            rope_centers = rope_ranges.mean(dim=1).to(dtype=encoder_k.dtype, device=encoder_k.device)
            for idx in range(human_map_count):
                per_frame[token_edges[idx] : token_edges[idx + 1]] = rope_centers[idx]
            encoder_pos = per_frame.repeat(grid_t)
            k_rope = self.rope_1d(phase.rope_1d, k_rope, encoder_pos)
        else:
            k_rope = self.rope_1d.apply_prepared(phase.rope_1d, k_rope, *encoder_rope)
        encoder_k = rearrange(k_rope, "1 h (t s) d -> t s h d", t=grid_t)
        return q, encoder_k
