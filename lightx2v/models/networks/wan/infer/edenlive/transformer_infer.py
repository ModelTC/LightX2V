import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.common.ops.rope.torch_rope import TorchRealRope
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer


class EdenLiveTransformerInfer(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "phase":
            raise NotImplementedError("EdenLive phase offload needs its residual casts mapped; use block or model offload")
        self.cache_pools = {}
        self.rotary = TorchRealRope(layout="interleaved", compute_dtype=self.infer_dtype)

    def infer(self, weights, pre_infer_out):
        self.current_pre = pre_infer_out
        return super().infer(weights, pre_infer_out)

    def infer_block(self, block, x, pre_infer_out, self_attn_kwargs=None):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (block.compute_phases[0].modulation.tensor.float() + pre_infer_out.embed0.float()).chunk(6, dim=1)
        y = self.infer_self_attn(block.compute_phases[0], x, shift, scale, pre_infer_out.grid_sizes.tuple)
        x = (x.float() + y * gate.squeeze(1)).to(self.infer_dtype)
        attention = self.infer_cross_attn_output(block.compute_phases[1], x, pre_infer_out.context)
        x = x + attention
        phase = block.compute_phases[2]
        normalized = (phase.norm2.apply(x.float()) * (1 + ff_scale.squeeze(1)) + ff_shift.squeeze(1)).to(self.infer_dtype)
        y = phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(normalized), approximate="tanh"))
        return (x.float() + y.float() * ff_gate.squeeze(1)).to(self.infer_dtype)

    def infer_non_blocks(self, weights, x, embed):
        shift, scale = (weights.head_modulation.tensor + embed.unsqueeze(1)).unbind(1)
        x = (weights.norm.apply(x.float()) * (1 + scale) + shift).to(self.infer_dtype)
        return weights.head.apply(x)

    def infer_cross_attn_output(self, phase, x, context):
        segments = self.current_pre.adapter_args["text_segments"]
        if segments is None:
            return super().infer_cross_attn_output(phase, x, context)
        normalized = phase.norm3.apply(x).to(self.infer_dtype)
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(normalized)).view(-1, self.num_heads, self.head_dim)
        k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, self.num_heads, self.head_dim)
        v = phase.cross_attn_v.apply(context).view_as(k)
        tokens_per_frame = math.prod(self.current_pre.grid_sizes.tuple[1:])
        local_start = self.current_pre.adapter_args.get("seq_start", 0)
        attention = torch.zeros_like(q)
        cursor = 0
        for index, frames in enumerate(segments):
            end = cursor + frames * tokens_per_frame
            left, right = max(cursor - local_start, 0), min(end - local_start, q.shape[0])
            if right > left:
                text_start = index * self.config["text_len"]
                text_end = text_start + self.config["text_len"]
                output = phase.cross_attn_1.apply(
                    q=q[left:right],
                    k=k[text_start:text_end],
                    v=v[text_start:text_end],
                    cu_seqlens_q=torch.tensor([0, right - left], dtype=torch.int32),
                    cu_seqlens_kv=torch.tensor([0, self.config["text_len"]], dtype=torch.int32),
                    max_seqlen_q=right - left,
                    max_seqlen_kv=self.config["text_len"],
                )
                attention[left:right] = output.view(right - left, self.num_heads, self.head_dim)
            cursor = end
        return phase.cross_attn_o.apply(attention.flatten(1))

    def _freqs(self, frames, height, width, device):
        half = self.head_dim // 2
        widths = (half - 2 * (half // 3), half // 3, half // 3)
        axes = []
        for count, size in zip((frames, height, width), widths):
            angle = torch.outer(torch.arange(count, device=device, dtype=torch.float64), 10000 ** (-torch.arange(size, device=device, dtype=torch.float64) / size))
            axes.append(torch.polar(torch.ones_like(angle), angle).to(torch.complex64))
        return torch.cat(
            (axes[0][:, None, None].expand(-1, height, width, -1), axes[1][None, :, None].expand(frames, -1, width, -1), axes[2][None, None, :].expand(frames, height, -1, -1)), dim=-1
        ).reshape(-1, 1, half)

    def infer_self_attn(self, phase, x, shift_msa, scale_msa, grid_sizes=None, **kwargs):
        normalized = (phase.norm1.apply(x.float()) * (1 + scale_msa.squeeze(1)) + shift_msa.squeeze(1)).to(self.infer_dtype)
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(normalized)).view(-1, self.num_heads, self.head_dim)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(normalized)).view_as(q)
        v = phase.self_attn_v.apply(normalized).view_as(q)
        frames, height, width = grid_sizes
        current_tokens = math.prod(grid_sizes)
        if self.seq_parallel:
            gathered_k = [torch.empty_like(k) for _ in range(dist.get_world_size(self.seq_p_group))]
            gathered_v = [torch.empty_like(v) for _ in gathered_k]
            dist.all_gather(gathered_k, k, self.seq_p_group)
            dist.all_gather(gathered_v, v, self.seq_p_group)
            k, v = torch.cat(gathered_k)[:current_tokens], torch.cat(gathered_v)[:current_tokens]
        if self.config["model_cls"].endswith("_ar"):
            pool = self.cache_pools[self.scheduler.infer_condition]
            layer = self.block_idx
            if self.config["ar_config"].get("kv_offload", False):
                pool.begin_layer(layer)
            end = pool.get_local_end(layer)
            sink = self.config["ar_config"]["sink_size"] * height * width
            capacity = self.config["ar_config"]["local_attn_size"] * height * width
            evict = max(0, end + current_tokens - capacity)
            query_start = min(self.scheduler.chunk_start, self.config["ar_config"]["local_attn_size"] - self.config["ar_config"]["num_frame_per_chunk"])
            visible_capacity = (query_start + frames) * height * width
            visible_evict = max(0, end + current_tokens - visible_capacity)
            history_k = pool.k_cache(layer, 0, end)
            history_v = pool.v_cache(layer, 0, end)
            if visible_evict:
                history_k = torch.cat((history_k[:sink], history_k[sink + visible_evict :]))
                history_v = torch.cat((history_v[:sink], history_v[sink + visible_evict :]))
            # Materialize before commit: cache views may alias destination slots.
            keys, values = torch.cat((history_k, k)), torch.cat((history_v, v))
            if self.scheduler.cache_update:
                if pool.get_global_end(layer) != self.scheduler.chunk_start * height * width:
                    raise RuntimeError("EdenLive cache commit cursor does not match the chunk")
                if evict:
                    pool.roll_window(layer, sink, evict)
                end -= evict
                pool.store_kv(k, v, end, end + current_tokens, layer)
                pool.set_ends(layer, (self.scheduler.chunk_start + frames) * height * width, end + current_tokens)
            k, v = keys, values
        visible_frames = k.shape[0] // (height * width)
        key_freqs = self._freqs(visible_frames, height, width, q.device)
        query_freqs = key_freqs[-current_tokens:]
        if self.seq_parallel:
            start = self.current_pre.adapter_args["seq_start"]
            query_freqs = F.pad(query_freqs, (0, 0, 0, 0, 0, (-current_tokens) % dist.get_world_size(self.seq_p_group)))[start : start + q.shape[0]]
        q = self.rotary.apply_single(q, query_freqs)
        k = self.rotary.apply_single(k, key_freqs)
        attention = phase.self_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, k.shape[0]], dtype=torch.int32),
            max_seqlen_q=q.shape[0],
            max_seqlen_kv=k.shape[0],
        )
        if self.config["model_cls"].endswith("_ar") and self.config["ar_config"].get("kv_offload", False):
            pool.end_layer(layer, next_prefetch=None)
        return phase.self_attn_o.apply(attention)
