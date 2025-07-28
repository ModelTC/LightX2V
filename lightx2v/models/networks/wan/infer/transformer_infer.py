import torch
from .utils import compute_freqs, compute_freqs_dist, apply_rotary_emb
from lightx2v.common.offload.manager import WeightAsyncStreamManager, LazyWeightAsyncStreamManager
from lightx2v.utils.envs import *


class WanTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.blocks_num = config["num_layers"]
        self.phases_num = 3
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if self.config["cpu_offload"]:
            if "offload_ratio" in self.config:
                offload_ratio = self.config["offload_ratio"]
            else:
                offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self._infer_with_offload
            elif offload_granularity == "phase":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self._infer_with_phases_offload
                else:
                    self.infer_func = self._infer_with_phases_lazy_offload

            if not self.config.get("lazy_load", False):
                self.weights_stream_mgr = WeightAsyncStreamManager(
                    blocks_num=self.blocks_num,
                    offload_ratio=offload_ratio,
                    phases_num=self.phases_num,
                )
            else:
                self.weights_stream_mgr = LazyWeightAsyncStreamManager(
                    blocks_num=self.blocks_num,
                    offload_ratio=offload_ratio,
                    phases_num=self.phases_num,
                    num_disk_workers=self.config.get("num_disk_workers", 2),
                    max_memory=self.config.get("max_memory", 2),
                )
        else:
            self.infer_func = self._infer_without_offload

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def _calculate_q_k_len(self, q, k_lens):
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        import pdb; pdb.set_trace()
        return self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context)

    def _infer_with_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                self.weights_stream_mgr.active_weights[0] = weights.blocks[0]
                self.weights_stream_mgr.active_weights[0].to_cuda()

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)

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
                )

            self.weights_stream_mgr.swap_weights()

        return x

    def _infer_with_phases_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        for block_idx in range(weights.blocks_num):
            weights.blocks[block_idx].modulation.to_cuda()

            if embed0.dim() == 3:
                modulation = weights.blocks[block_idx].modulation.tensor.unsqueeze(2)
                current_embed0 = (modulation + embed0).chunk(6, dim=1)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in current_embed0]
            elif embed0.dim() == 2:
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.blocks[block_idx].modulation.tensor + embed0).chunk(6, dim=1)

            for phase_idx in range(self.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    phase = weights.blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    cur_phase_idx, cur_phase = self.weights_stream_mgr.active_weights[0]
                    if cur_phase_idx == 0:
                        x = self._infer_self_attn(
                            cur_phase,
                            x,
                            shift_msa,
                            scale_msa,
                            gate_msa,
                            grid_sizes,
                            freqs,
                            seq_lens,
                        )

                    elif cur_phase_idx == 1:
                        x = self._infer_cross_attn(cur_phase, x, context)

                    elif cur_phase_idx == 2:
                        x = self._infer_ffn(cur_phase, x, c_shift_msa, c_scale_msa, c_gate_msa)

                is_last_phase = block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1
                if not is_last_phase:
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            weights.blocks[block_idx].modulation.to_cpu()

        torch.cuda.empty_cache()

        return x

    def _infer_with_phases_lazy_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        self.weights_stream_mgr.prefetch_weights_from_disk(weights)

        for block_idx in range(weights.blocks_num):
            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                weights.blocks[block_idx].modulation.to_cuda()

            if embed0.dim() == 3:
                modulation = weights.blocks[block_idx].modulation.tensor.unsqueeze(2)
                current_embed0 = (modulation + embed0).chunk(6, dim=1)
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in current_embed0]
            elif embed0.dim() == 2:
                shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.blocks[block_idx].modulation.tensor + embed0).chunk(6, dim=1)

            for phase_idx in range(self.weights_stream_mgr.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    obj_key = (block_idx, phase_idx)
                    phase = self.weights_stream_mgr.pin_memory_buffer.get(obj_key)
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (obj_key, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    (
                        (
                            _,
                            cur_phase_idx,
                        ),
                        cur_phase,
                    ) = self.weights_stream_mgr.active_weights[0]

                    if cur_phase_idx == 0:
                        x = self._infer_self_attn(
                            cur_phase,
                            x,
                            shift_msa,
                            scale_msa,
                            gate_msa,
                            grid_sizes,
                            freqs,
                            seq_lens,
                        )
                    elif cur_phase_idx == 1:
                        x = self._infer_cross_attn(cur_phase, x, context)
                    elif cur_phase_idx == 2:
                        x = self._infer_ffn(cur_phase, x, c_shift_msa, c_scale_msa, c_gate_msa)

                if not (block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1):
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.weights_stream_mgr.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            weights.blocks[block_idx].modulation.to_cpu()
            self.weights_stream_mgr._async_prefetch_block(weights)

        torch.cuda.empty_cache()

        return x

    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
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
            )
        return x

    def _infer_self_attn(self, weights, x, shift_msa, scale_msa, gate_msa, grid_sizes, freqs, seq_lens):
        if hasattr(weights, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa) * weights.smooth_norm1_weight.tensor
            norm1_bias = shift_msa * weights.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa
            norm1_bias = shift_msa

        norm1_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        norm1_out = (norm1_out * norm1_weight + norm1_bias).squeeze(0)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim
        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)
        else:
            freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)

        q = apply_rotary_emb(q, freqs_i)
        k = apply_rotary_emb(k, freqs_i)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=seq_lens)

        if not self.parallel_attention:
            attn_out = weights.self_attn_1.apply(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k.size(0),
                model_cls=self.config["model_cls"],
            )
        else:
            attn_out = self.parallel_attention(
                attention_type=self.attention_type,
                q=q,
                k=k,
                v=v,
                img_qkv_len=q.shape[0],
                cu_seqlens_qkv=cu_seqlens_q,
            )

        y = weights.self_attn_o.apply(attn_out)
        x.add_(y * gate_msa.squeeze(0))
        return x

    def _infer_cross_attn(self, weights, x, context):
        norm3_out = weights.norm3.apply(x)

        if self.task == "i2v":
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        n, d = self.num_heads, self.head_dim
        q = weights.cross_attn_norm_q.apply(weights.cross_attn_q.apply(norm3_out)).view(-1, n, d)
        k = weights.cross_attn_norm_k.apply(weights.cross_attn_k.apply(context)).view(-1, n, d)
        v = weights.cross_attn_v.apply(context).view(-1, n, d)

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32, device=k.device),
        )

        attn_out = weights.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_k,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
            model_cls=self.config["model_cls"],
        )

        if self.task == "i2v" and context_img is not None:
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
                model_cls=self.config["model_cls"],
            )

            attn_out = attn_out + img_attn_out

        attn_out = weights.cross_attn_o.apply(attn_out)
        x.add_(attn_out)
        return x

    def _infer_ffn(self, weights, x, c_shift_msa, c_scale_msa, c_gate_msa):
        if hasattr(weights, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze(0)) * weights.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze(0) * weights.smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa.squeeze(0)
            norm2_bias = c_shift_msa.squeeze(0)

        norm2_out = torch.nn.functional.layer_norm(x, (x.shape[1],), None, None, 1e-6)
        y = weights.ffn_0.apply(norm2_out * norm2_weight + norm2_bias)
        y = torch.nn.functional.gelu(y, approximate="tanh")
        y = weights.ffn_2.apply(y)
        x.add_(y * c_gate_msa.squeeze(0))
        return x

    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        if embed0.dim() == 3:
            modulation = weights.modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.modulation.tensor + embed0).chunk(6, dim=1)

        x = self._infer_self_attn(
            weights.compute_phases[0],
            x,
            shift_msa,
            scale_msa,
            gate_msa,
            grid_sizes,
            freqs,
            seq_lens,
        )
        x = self._infer_cross_attn(weights.compute_phases[1], x, context)
        x = self._infer_ffn(weights.compute_phases[2], x, c_shift_msa, c_scale_msa, c_gate_msa)
        return x
