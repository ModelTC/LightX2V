import torch

from lightx2v.common.ops.attn.flex_attn import FlexAttnWeight
from lightx2v.common.ops.attn.utils.all2all import all2all_head2seq, all2all_seq2head
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


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
            raise NotImplementedError("Wan-Animate-2 does not support quantized SP communication or head-parallel SP.")

        self.flex_attn = FlexAttnWeight()
        self.log_scale = float(config.get("log_scale", 0.0))
        self.mode = "generation"
        self.reference_kv_cache = None

    def _apply_sensitive_norm(self, norm, x):
        # The selected LN backend owns its accumulation precision.  Keep its
        # input/output on the sensitive residual path; projection boundaries
        # below are the only places that downcast to the inference dtype.
        x = norm.apply(x.to(self.sensitive_layer_dtype))
        return x.to(self.sensitive_layer_dtype)

    def pre_process(self, modulation, embed0):
        modulation = modulation.tensor.to(self.sensitive_layer_dtype)
        embed0 = embed0.to(self.sensitive_layer_dtype)
        values = (modulation + embed0).chunk(6, dim=1)
        return tuple(value.squeeze(1) for value in values)

    def _qkv(self, phase, x, shift_msa, scale_msa):
        norm1_out = self._apply_sensitive_norm(phase.norm1, x)
        norm1_out = self.modulate_func(
            norm1_out.contiguous(),
            scale=scale_msa,
            shift=shift_msa,
        ).squeeze()
        norm1_out = norm1_out.to(self.infer_dtype)

        seq_len = norm1_out.shape[0]
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).to(self.infer_dtype).view(seq_len, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).to(self.infer_dtype).view(seq_len, self.num_heads, self.head_dim)
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
        rope_freqs = pre_infer_out.adapter_args["rope_freqs"]
        q, k = phase.rope.apply(q, k, rope_freqs)
        self.reference_kv_cache.store_kv(k, v, self.block_idx)
        valid_len = int(pre_infer_out.valid_token_len)
        cu_q = self._cu_seqlens(q.shape[0])
        cu_k = self._cu_seqlens(valid_len)
        output = phase.self_attn_1.apply(
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
        x = x.to(self.sensitive_layer_dtype)
        x.add_(y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze())
        norm3_out = self._apply_sensitive_norm(phase.norm3, x).to(self.infer_dtype)

        context_img = context[:257]
        context_text = context[257:]
        heads, head_dim = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).to(self.infer_dtype).view(-1, heads, head_dim)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context_text)).to(self.infer_dtype).view(-1, heads, head_dim)
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

        k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).to(self.infer_dtype).view(-1, heads, head_dim)
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
        return x, phase.cross_attn_o.apply((attn_out + image_out).to(self.infer_dtype))

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        del c_gate_msa
        x.add_(attn_out.to(self.sensitive_layer_dtype))
        norm2_out = self._apply_sensitive_norm(phase.norm2, x)
        norm2_out = self.modulate_func(
            norm2_out.contiguous(),
            scale=c_scale_msa,
            shift=c_shift_msa,
        ).squeeze()
        y = phase.ffn_0.apply(norm2_out.to(self.infer_dtype))
        y = torch.nn.functional.gelu(y, approximate="tanh")
        return phase.ffn_2.apply(y.to(self.infer_dtype))

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        del pre_infer_out
        x.add_(y.to(self.sensitive_layer_dtype) * c_gate_msa.squeeze())
        return x

    def infer_non_blocks(self, weights, x, embedding):
        modulation = weights.head_modulation.tensor.to(self.sensitive_layer_dtype)
        embedding = embedding.to(self.sensitive_layer_dtype)
        shift, scale = (modulation + embedding.unsqueeze(1)).chunk(2, dim=1)
        x = self._apply_sensitive_norm(weights.norm, x)
        x = self.modulate_func(
            x.contiguous(),
            scale=scale,
            shift=shift,
        ).squeeze()
        return weights.head.apply(x.to(self.infer_dtype))

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
        q, k = phase.rope.apply(q, k, args["rope_freqs"])

        if not self.reference_kv_cache.is_ready(self.block_idx):
            raise RuntimeError(f"Reference K/V for block {self.block_idx} was not prefetched.")
        reference_k = self.reference_kv_cache.k_cache(self.block_idx)
        reference_v = self.reference_kv_cache.v_cache(self.block_idx)

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
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)
        return x

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
