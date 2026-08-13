import torch

from lightx2v.common.ops.attn.flex_attn import FlexAttnWeight
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


def rope_apply_source(x, grid_sizes, freqs, time_stride=1):
    """Apply Wan-Animate-2's float64-complex 3D RoPE to unbatched Q/K."""
    if x.ndim != 3:
        raise ValueError(f"Wan-Animate-2 RoPE expects [seq, heads, dim], got {tuple(x.shape)}")

    heads, complex_dim = x.size(1), x.size(2) // 2
    freqs_t, freqs_h, freqs_w = freqs.split(
        [complex_dim - 2 * (complex_dim // 3), complex_dim // 3, complex_dim // 3],
        dim=1,
    )
    frames, height, width = (int(value) for value in grid_sizes)
    valid_len = frames * height * width
    if valid_len > x.shape[0]:
        raise ValueError(f"RoPE grid {grid_sizes} requires {valid_len} tokens, got {x.shape[0]}.")

    rotary = torch.cat(
        [
            freqs_t[: frames * time_stride : time_stride].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            freqs_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            freqs_w[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(valid_len, 1, -1)
    valid = torch.view_as_complex(x[:valid_len].to(torch.float64).reshape(valid_len, heads, -1, 2))
    valid = torch.view_as_real(valid * rotary).flatten(2)
    return torch.cat([valid, x[valid_len:]], dim=0).float()


class WanAnimate2TransformerInfer(WanOffloadTransformerInfer):
    """Wan transformer with a static driving-reference branch per clip."""

    def __init__(self, config):
        super().__init__(config)
        if config.get("feature_caching", "NoCaching") != "NoCaching":
            raise NotImplementedError("Wan-Animate-2 does not support feature caching.")
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "phase":
            raise NotImplementedError("Wan-Animate-2 supports model/block offload, not phase offload.")
        if config.get("use_compile", False):
            raise NotImplementedError("Wan-Animate-2 block compilation is disabled; its FlexAttention kernel is compiled internally.")
        if self.seq_parallel and self.seq_p_attn_type != "ulysses":
            raise NotImplementedError("Wan-Animate-2 sequence parallelism currently requires Ulysses.")
        if self.seq_parallel and (self.seq_p_quant_scheme is not None or self.seq_p_head_parallel):
            raise NotImplementedError("Wan-Animate-2 source-parity mode does not support quantized SP communication or head-parallel SP.")

        self.flex_attn = FlexAttnWeight()
        self.log_scale = float(config.get("log_scale", 0.0))
        self.mode = "generation"
        self.reference_kv_cache = None

    @staticmethod
    def _layer_norm_fp32(norm, x):
        weight = norm._get_actual_weight()
        bias = norm._get_actual_bias()
        # Upstream computes LayerNorm statistics in FP32, then casts the
        # normalized value back to the residual stream dtype.  This cast is
        # observable in block 0, whose patch-embedding residual is BF16.
        return torch.nn.functional.layer_norm(
            x.float(),
            (x.shape[-1],),
            None if weight is None else weight.float(),
            None if bias is None else bias.float(),
            norm.eps,
        ).to(x.dtype)

    @staticmethod
    def _mm_fp32(module, value):
        value = value.float()
        weight = module._get_actual_weight().float()
        bias = module._get_actual_bias()
        output = torch.mm(value, weight) if bias is None else torch.addmm(bias.float(), value, weight)
        if getattr(module, "has_lora_branch", False):
            hidden = torch.mm(value, module.lora_down.float().t())
            lora = torch.mm(hidden, module.lora_up.float().t())
            output = output + float(module.lora_strength * module.lora_scale) * lora
        return output

    def pre_process(self, modulation, embed0):
        # The released block performs modulation and all residual gates in
        # float32 even though projections/attention run in BF16.
        values = (modulation.tensor.float() + embed0.float()).chunk(6, dim=1)
        return tuple(value.squeeze(1) for value in values)

    def _qkv(self, phase, x, shift_msa, scale_msa):
        norm1_out = self._layer_norm_fp32(phase.norm1, x).float()
        norm1_out = norm1_out * (1 + scale_msa.squeeze()) + shift_msa.squeeze()
        norm1_out = norm1_out.to(self.infer_dtype)

        seq_len = norm1_out.shape[0]
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(seq_len, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(seq_len, self.num_heads, self.head_dim)
        v = phase.self_attn_v.apply(norm1_out).view(seq_len, self.num_heads, self.head_dim)
        return q, k, v

    def _to_head_shard(self, q, k, v):
        if not self.seq_parallel:
            return q, k, v
        # The source keeps Q/K/V in a leading three-way batch while doing its
        # fused all-to-all.  The shared LightX2V helper is strictly 3D, so run
        # three equivalent collectives rather than concatenating their sequence
        # axes (which would interleave rank-local Q/K/V segments incorrectly).
        return tuple(all2all_seq2head(tensor, group=self.seq_p_group) for tensor in (q, k, v))

    def _to_sequence_shard(self, output):
        if not self.seq_parallel:
            return output
        return all2all_head2seq(output, group=self.seq_p_group)

    @staticmethod
    def _cu_seqlens(length):
        return torch.tensor([0, int(length)], dtype=torch.int32)

    def _reference_self_attention(self, phase, x, shift_msa, scale_msa, pre_infer_out):
        q, k, v = self._qkv(phase, x, shift_msa, scale_msa)
        q, k, v = self._to_head_shard(q, k, v)
        self.reference_kv_cache.store_kv(k, v, self.block_idx)

        grid = pre_infer_out.grid_sizes.tuple
        rope_table = pre_infer_out.adapter_args["rope_table"]
        stride = int(pre_infer_out.adapter_args["refer_stride"])
        q = rope_apply_source(q, grid, rope_table, stride)
        k = rope_apply_source(k, grid, rope_table, stride)
        valid_len = int(pre_infer_out.valid_token_len)
        cu_q = self._cu_seqlens(q.shape[0])
        cu_k = self._cu_seqlens(valid_len)
        output = phase.self_attn_1.apply(
            # Upstream's attention wrapper converts float32 RoPE Q/K to V's
            # half dtype for the kernel, then projects the half result.
            q=q.to(v.dtype),
            k=k[:valid_len].to(v.dtype),
            v=v[:valid_len],
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=valid_len,
        ).view(q.shape[0], q.shape[1], q.shape[2])
        output = self._to_sequence_shard(output).reshape(x.shape[0], -1)
        return phase.self_attn_o.apply(output.to(self.infer_dtype))

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        x = x.float() + y_out.float() * gate_msa.squeeze().float()
        norm3_out = self._layer_norm_fp32(phase.norm3, x).to(self.infer_dtype)

        context_img = context[:257]
        context_text = context[257:]
        context_img = context_img.to(self.infer_dtype)
        context_text = context_text.to(self.infer_dtype)
        heads, head_dim = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, heads, head_dim)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context_text)).view(-1, heads, head_dim)
        v = phase.cross_attn_v.apply(context_text).view(-1, heads, head_dim)
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )

        k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, heads, head_dim)
        v_img = phase.cross_attn_v_img.apply(context_img).view(-1, heads, head_dim)
        image_out = phase.cross_attn_2.apply(
            q=q,
            k=k_img,
            v=v_img,
            cu_seqlens_q=self.cross_attn_cu_seqlens_q,
            cu_seqlens_kv=self.cross_attn_cu_seqlens_kv_img,
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k_img.shape[0],
        )
        return x, phase.cross_attn_o.apply(attn_out + image_out)

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        del c_gate_msa
        # ``infer_block`` owns the residual tensor and passes it to
        # ``post_process`` after this helper returns.  Keep the cross-attention
        # residual on that same tensor; rebinding a local ``x`` here silently
        # drops cross-attention from every transformer block.
        x.add_(attn_out.float())
        norm2_out = self._layer_norm_fp32(phase.norm2, x)
        norm2_out = norm2_out * (1 + c_scale_msa.squeeze()) + c_shift_msa.squeeze()
        y = phase.ffn_0.apply(norm2_out.to(self.infer_dtype))
        y = torch.nn.functional.gelu(y, approximate="tanh")
        return phase.ffn_2.apply(y)

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        del pre_infer_out
        return x.float() + y.float() * c_gate_msa.squeeze().float()

    def infer_non_blocks(self, weights, x, embedding):
        modulation = weights.head_modulation.tensor.float()
        shift, scale = (modulation + embedding.float().unsqueeze(1)).chunk(2, dim=1)
        x = self._layer_norm_fp32(weights.norm, x)
        x = x * (1 + scale.squeeze(1)) + shift.squeeze(1)
        # Source unpatchifies under an outer BF16 autocast region.  Its einsum
        # therefore rounds the FP32 head output to BF16 before rearranging it.
        return self._mm_fp32(weights.head, x).to(self.infer_dtype)

    @staticmethod
    def _pack_stream(tensor, grid, frame_capacity, total_len):
        frames, height, width = (int(value) for value in grid)
        spatial = height * width
        valid_len = frames * spatial
        packed = tensor.new_zeros(total_len, tensor.shape[1], tensor.shape[2])
        source = tensor[:valid_len].view(frames, spatial, tensor.shape[1], tensor.shape[2])
        target = packed[: frames * frame_capacity].view(frames, frame_capacity, tensor.shape[1], tensor.shape[2])
        target[:, :spatial] = source
        return packed

    @staticmethod
    def _unpack_stream(tensor, grid, frame_capacity):
        frames, height, width = (int(value) for value in grid)
        spatial = height * width
        return tensor[: frames * frame_capacity].view(frames, frame_capacity, tensor.shape[1], tensor.shape[2])[:, :spatial].reshape(frames * spatial, tensor.shape[1], tensor.shape[2])

    def _generation_self_attention(self, phase, x, shift_msa, scale_msa, pre_infer_out):
        q, k, v = self._qkv(phase, x, shift_msa, scale_msa)
        q, k, v = self._to_head_shard(q, k, v)

        args = pre_infer_out.adapter_args
        grid = pre_infer_out.grid_sizes.tuple
        reference_grid = args["reference_grid"]
        q = rope_apply_source(q, grid, args["rope_table"])
        k = rope_apply_source(k, grid, args["rope_table"])

        if not self.reference_kv_cache.is_ready(self.block_idx):
            raise RuntimeError(f"Reference K/V for block {self.block_idx} was not prefetched.")
        reference_k = self.reference_kv_cache.k_cache(self.block_idx)
        reference_v = self.reference_kv_cache.v_cache(self.block_idx)
        reference_k = rope_apply_source(
            reference_k,
            reference_grid,
            args["reference_rope_table"],
            int(args["refer_stride"]),
        )

        _, q_total, reference_total, frame_capacity = self.flex_attn.mask_layout(args["origin_len"], args["origin_area"], q.device)
        q_padding = q[int(pre_infer_out.valid_token_len) :].clone()
        q_packed = self._pack_stream(q, grid, frame_capacity, q_total)
        k_packed = self._pack_stream(k, grid, frame_capacity, q_total + reference_total)
        v_packed = self._pack_stream(v, grid, frame_capacity, q_total + reference_total)

        reference_k_packed = self._pack_stream(reference_k, reference_grid, frame_capacity, reference_total)
        reference_v_packed = self._pack_stream(reference_v, reference_grid, frame_capacity, reference_total)
        k_packed[q_total:] = reference_k_packed
        v_packed[q_total:] = reference_v_packed

        output = self.flex_attn.apply(
            q_packed,
            k_packed,
            v_packed,
            origin_len=args["origin_len"],
            origin_area=args["origin_area"],
            log_scale=self.log_scale,
        )
        output = self._unpack_stream(output, grid, frame_capacity)
        output = torch.cat([output, q_padding], dim=0)
        output = self._to_sequence_shard(output).reshape(x.shape[0], -1)
        return phase.self_attn_o.apply(output.to(self.infer_dtype))

    def infer_block(self, block, x, pre_infer_out):
        if self.mode == "generation" and pre_infer_out.adapter_args["is_uncondition"] and self.block_idx == 9:
            return x

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.pre_process(
            block.compute_phases[0].modulation,
            pre_infer_out.embed0,
        )
        if self.mode == "reference":
            y_out = self._reference_self_attention(block.compute_phases[0], x, shift_msa, scale_msa, pre_infer_out)
        else:
            y_out = self._generation_self_attention(block.compute_phases[0], x, shift_msa, scale_msa, pre_infer_out)

        x, attn_out = self.infer_cross_attn(
            block.compute_phases[1],
            x,
            pre_infer_out.context,
            y_out,
            gate_msa,
        )
        y = self.infer_ffn(
            block.compute_phases[2],
            x,
            attn_out,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        )
        # The source wraps every Incontext_AttentionBlock with composable FSDP
        # and sets MixedPrecisionPolicy(output_dtype=BF16).  Consequently the
        # FP32 residual produced inside a block is rounded back to the model
        # dtype before it enters the next block (also when world_size == 1).
        return self.post_process(x, y, c_gate_msa, pre_infer_out).to(self.infer_dtype)

    @torch.no_grad()
    def infer_reference(self, weights, pre_infer_out):
        self.mode = "reference"
        self.reference_kv_cache = pre_infer_out.adapter_args["reference_kv_cache"]
        self.reference_kv_cache.reset()
        self.reset_infer_states(pre_infer_out.x, pre_infer_out.context)
        self.reset_attention_states(weights.blocks)
        self.infer_main_blocks(weights.blocks, pre_infer_out)
        if not self.reference_kv_cache.is_ready():
            raise RuntimeError("Reference K/V prefill did not populate every transformer block.")

    @torch.no_grad()
    def infer(self, weights, pre_infer_out):
        self.mode = "generation"
        self.reference_kv_cache = pre_infer_out.adapter_args["reference_kv_cache"]
        if not self.reference_kv_cache.is_ready():
            raise RuntimeError("Wan-Animate-2 generation requires a completed reference prefill.")
        self.reset_infer_states(pre_infer_out.x, pre_infer_out.context)
        self.reset_attention_states(weights.blocks)
        x = self.infer_main_blocks(weights.blocks, pre_infer_out)
        return self.infer_non_blocks(weights, x, pre_infer_out.embed)
