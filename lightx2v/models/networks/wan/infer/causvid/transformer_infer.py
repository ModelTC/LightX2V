import math

import torch

from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.utils.envs import *

from ..utils import apply_rotary_emb, compute_freqs_causvid


class WanTransformerInferCausVid(WanOffloadTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        self.num_frames = config["num_frames"]
        self.num_frame_per_block = config["num_frame_per_block"]
        self.frame_seq_length = config["frame_seq_length"]
        self.text_len = config["text_len"]
        self.kv_cache = None
        self.crossattn_cache = None

    def _init_kv_cache(self, dtype, device):
        kv_cache = []
        kv_size = self.num_frames * self.frame_seq_length

        for _ in range(self.blocks_num):
            kv_cache.append(
                {
                    "k": torch.zeros([kv_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([kv_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                }
            )

        self.kv_cache = kv_cache

    def _init_crossattn_cache(self, dtype, device):
        crossattn_cache = []

        for _ in range(self.blocks_num):
            crossattn_cache.append(
                {
                    "k": torch.zeros([self.text_len, self.num_heads, self.head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([self.text_len, self.num_heads, self.head_dim], dtype=dtype, device=device),
                    "is_init": False,
                }
            )

        self.crossattn_cache = crossattn_cache

    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        self.get_scheduler_values()
        self.reset_infer_states()
        return self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end)

    def _infer_with_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(
                    self.weights_stream_mgr.active_weights[0],
                    grid_sizes,
                    embed,
                    x,
                    embed0,
                    seq_lens,
                    freqs,
                    context,
                    block_idx,
                    kv_start,
                    kv_end,
                )

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)
            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, kv_start, kv_end):
        for block_idx in range(self.blocks_num):
            x = self.infer_block(
                weights.blocks[block_idx],
                grid_sizes,
                embed,
                x,
                embed0,
                seq_lens,
                freqs,
                context,
                block_idx,
                kv_start,
                kv_end,
            )
        return x

    def infer_self_attn(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx, kv_start, kv_end):
        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * (1 + embed0[1]) + embed0[0]).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs_causvid(q.size(2) // 2, grid_sizes, freqs, start_frame=kv_start // math.prod(grid_sizes[0][1:]).item())
        else:
            # TODO: Implement parallel attention for causvid inference
            raise NotImplementedError("Parallel attention is not implemented for causvid inference")

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        self.kv_cache[block_idx]["k"][kv_start:kv_end] = k
        self.kv_cache[block_idx]["v"][kv_start:kv_end] = v

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q=q, k_lens=torch.tensor([kv_end], dtype=torch.int32, device=k.device))

        if not self.parallel_attention:
            attn_out = weights.self_attn_1.apply(
                q=q,
                k=self.kv_cache[block_idx]["k"][:kv_end],
                v=self.kv_cache[block_idx]["v"][:kv_end],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
            )
        else:
            # TODO: Implement parallel attention for causvid inference
            raise NotImplementedError("Parallel attention is not implemented for causvid inference")

        y = weights.self_attn_o.apply(attn_out)

        x = x + y * embed0[2].squeeze(0)

        return x

    def infer_cross_attn(self, weights, x, context, block_idx):
        norm3_out = weights.norm3.apply(x)

        if self.task in ["i2v", "s2v"]:
            context_img = context[:257]
            context = context[257:]

        n, d = self.num_heads, self.head_dim
        q = weights.cross_attn_norm_q.apply(weights.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        if not self.crossattn_cache[block_idx]["is_init"]:
            k = weights.cross_attn_norm_k.apply(weights.cross_attn_k.apply(context)).view(-1, n, d)
            v = weights.cross_attn_v.apply(context).view(-1, n, d)
            self.crossattn_cache[block_idx]["k"] = k
            self.crossattn_cache[block_idx]["v"] = v
            self.crossattn_cache[block_idx]["is_init"] = True
        else:
            k = self.crossattn_cache[block_idx]["k"]
            v = self.crossattn_cache[block_idx]["v"]

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device))

        attn_out = weights.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        if self.task in ["i2v", "s2v"]:
            k_img = weights.cross_attn_norm_k_img.apply(weights.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = weights.cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )

            img_attn_out = weights.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )

            attn_out = attn_out + img_attn_out

        attn_out = weights.cross_attn_o.apply(attn_out)

        x = x + attn_out

        return x

    def infer_ffn(self, weights, x, embed0):
        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.ffn_0.apply(norm2_out * (1 + embed0[4].squeeze(0)) + embed0[3].squeeze(0))
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.ffn_2.apply(y)
        x = x + y * embed0[5].squeeze(0)

        return x

    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx, kv_start, kv_end):
        if embed0.dim() == 3:
            modulation = weights.compute_phases[0].modulation.tensor.unsqueeze(2)  # 1, 6, 1, dim
            embed0 = embed0.unsqueeze(0)  #
            embed0 = (modulation + embed0).chunk(6, dim=1)
            embed0 = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            embed0 = (weights.compute_phases[0].modulation.tensor + embed0).chunk(6, dim=1)

        x = self.infer_self_attn(weights.compute_phases[1], grid_sizes, embed, x, embed0, seq_lens, freqs, context, block_idx, kv_start, kv_end)
        x = self.infer_cross_attn(weights.compute_phases[2], x, context, block_idx)
        x = self.infer_ffn(weights.compute_phases[3], x, embed0)

        return x
