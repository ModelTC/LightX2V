import torch
from .utils import compute_freqs, compute_freqs_dist, compute_freqs_audio, compute_freqs_audio_dist, apply_rotary_emb, apply_rotary_emb_chunk
from lightx2v.common.offload.manager import (
    WeightAsyncStreamManager,
    LazyWeightAsyncStreamManager,
)
from lightx2v.common.transformer_infer.transformer_infer import BaseTransformerInfer
from lightx2v.utils.envs import *
from functools import partial


class WanTransformerInfer(BaseTransformerInfer):
    def __init__(self, config):
        self.config = config
        self.task = config["task"]
        self.attention_type = config.get("attention_type", "flash_attn2")
        self.blocks_num = config["num_layers"]
        self.phases_num = 4
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.window_size = config.get("window_size", (-1, -1))
        self.parallel_attention = None
        if config.get("rotary_chunk", False):
            chunk_size = config.get("rotary_chunk_size", 100)
            self.apply_rotary_emb_func = partial(apply_rotary_emb_chunk, chunk_size=chunk_size)
        else:
            self.apply_rotary_emb_func = apply_rotary_emb
        self.clean_cuda_cache = self.config.get("clean_cuda_cache", False)
        self.mask_map = None

        if self.config["cpu_offload"]:
            if torch.cuda.get_device_capability(0) == (9, 0):
                assert self.config["self_attn_1_type"] != "sage_attn2"
            if "offload_ratio" in self.config:
                offload_ratio = self.config["offload_ratio"]
            else:
                offload_ratio = 1
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                if not self.config.get("lazy_load", False):
                    self.infer_func = self._infer_with_offload
                else:
                    self.infer_func = self._infer_with_lazy_offload
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
                    offload_gra=offload_granularity,
                )
        else:
            self.infer_func = self._infer_without_offload

        self.infer_conditional = True

    def switch_status(self):
        self.infer_conditional = not self.infer_conditional

    def _calculate_q_k_len(self, q, k_lens):
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    @torch.compile(disable=not CHECK_ENABLE_GRAPH_MODE())
    def infer(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
        return self.infer_func(weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks)

    def _infer_with_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
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

    def _infer_with_lazy_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        self.weights_stream_mgr.prefetch_weights_from_disk(weights.blocks)

        for block_idx in range(self.blocks_num):
            if block_idx == 0:
                block = self.weights_stream_mgr.pin_memory_buffer.get(block_idx)
                block.to_cuda()
                self.weights_stream_mgr.active_weights[0] = (block_idx, block)

            if block_idx < self.blocks_num - 1:
                self.weights_stream_mgr.prefetch_weights(block_idx + 1, weights.blocks)

            with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                x = self.infer_block(
                    self.weights_stream_mgr.active_weights[0][1],
                    grid_sizes,
                    embed,
                    x,
                    embed0,
                    seq_lens,
                    freqs,
                    context,
                )

            self.weights_stream_mgr.swap_weights()

            if block_idx == self.blocks_num - 1:
                self.weights_stream_mgr.pin_memory_buffer.pop_front()

            self.weights_stream_mgr._async_prefetch_block(weights.blocks)

        if self.clean_cuda_cache:
            del grid_sizes, embed, embed0, seq_lens, freqs, context
            torch.cuda.empty_cache()

        return x

    def _infer_with_phases_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
        for block_idx in range(weights.blocks_num):
            for phase_idx in range(self.phases_num):
                if block_idx == 0 and phase_idx == 0:
                    phase = weights.blocks[block_idx].compute_phases[phase_idx]
                    phase.to_cuda()
                    self.weights_stream_mgr.active_weights[0] = (phase_idx, phase)

                with torch.cuda.stream(self.weights_stream_mgr.compute_stream):
                    cur_phase_idx, cur_phase = self.weights_stream_mgr.active_weights[0]
                    if cur_phase_idx == 0:
                        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(cur_phase, embed0)

                    elif cur_phase_idx == 1:
                        y_out = self.infer_self_attn(
                            cur_phase,
                            grid_sizes,
                            x,
                            seq_lens,
                            freqs,
                            shift_msa,
                            scale_msa,
                        )
                    elif cur_phase_idx == 2:
                        x, attn_out = self.infer_cross_attn(cur_phase, x, context, y_out, gate_msa)
                    elif cur_phase_idx == 3:
                        y = self.infer_ffn(cur_phase, x, attn_out, c_shift_msa, c_scale_msa)
                        x = self.post_process(x, y, c_gate_msa)

                is_last_phase = block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1
                if not is_last_phase:
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            if self.clean_cuda_cache:
                del attn_out, y_out, y
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            del shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa
            del grid_sizes, embed, embed0, seq_lens, freqs, context
            torch.cuda.empty_cache()

        return x

    def _infer_with_phases_lazy_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
        self.weights_stream_mgr.prefetch_weights_from_disk(weights.blocks)

        for block_idx in range(weights.blocks_num):
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
                        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(
                            cur_phase,
                            embed0,
                        )
                    elif cur_phase_idx == 1:
                        y_out = self.infer_self_attn(
                            cur_phase,
                            grid_sizes,
                            x,
                            seq_lens,
                            freqs,
                            shift_msa,
                            scale_msa,
                        )
                    elif cur_phase_idx == 2:
                        x, attn_out = self.infer_cross_attn(cur_phase, x, context, y_out, gate_msa)
                    elif cur_phase_idx == 3:
                        y = self.infer_ffn(cur_phase, x, attn_out, c_shift_msa, c_scale_msa)
                        x = self.post_process(x, y, c_gate_msa)

                if not (block_idx == weights.blocks_num - 1 and phase_idx == self.phases_num - 1):
                    next_block_idx = block_idx + 1 if phase_idx == self.phases_num - 1 else block_idx
                    next_phase_idx = (phase_idx + 1) % self.weights_stream_mgr.phases_num
                    self.weights_stream_mgr.prefetch_phase(next_block_idx, next_phase_idx, weights.blocks)

                self.weights_stream_mgr.swap_phases()

            self.weights_stream_mgr._async_prefetch_block(weights.blocks)

            if self.clean_cuda_cache:
                del attn_out, y_out, y
                torch.cuda.empty_cache()

        if self.clean_cuda_cache:
            del shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa
            del grid_sizes, embed, embed0, seq_lens, freqs, context
            torch.cuda.empty_cache()

        return x

    def zero_temporal_component_in_3DRoPE(self, valid_token_length, rotary_emb=None):
        if rotary_emb is None:
            return None
        self.use_real = False
        rope_t_dim = 44
        if self.use_real:
            freqs_cos, freqs_sin = rotary_emb
            freqs_cos[valid_token_length:, :, :rope_t_dim] = 0
            freqs_sin[valid_token_length:, :, :rope_t_dim] = 0
            return freqs_cos, freqs_sin
        else:
            freqs_cis = rotary_emb
            freqs_cis[valid_token_length:, :, : rope_t_dim // 2] = 0
            return freqs_cis

    def _infer_without_offload(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context, audio_dit_blocks=None):
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

            if audio_dit_blocks is not None and len(audio_dit_blocks) > 0:
                for ipa_out in audio_dit_blocks:
                    if block_idx in ipa_out:
                        cur_modify = ipa_out[block_idx]
                        x = cur_modify["modify_func"](x, grid_sizes, **cur_modify["kwargs"])
        return x

    def infer_block(self, weights, grid_sizes, embed, x, embed0, seq_lens, freqs, context):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self.infer_modulation(
            weights.compute_phases[0],
            embed0,
        )
        y_out = self.infer_self_attn(
            weights.compute_phases[1],
            grid_sizes,
            x,
            seq_lens,
            freqs,
            shift_msa,
            scale_msa,
        )
        x, attn_out = self.infer_cross_attn(weights.compute_phases[2], x, context, y_out, gate_msa)
        y = self.infer_ffn(weights.compute_phases[3], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa)
        return x

    def infer_modulation(self, weights, embed0):
        if embed0.dim() == 3:
            modulation = weights.modulation.tensor.unsqueeze(2)
            embed0 = (modulation + embed0).chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [ei.squeeze(1) for ei in embed0]
        elif embed0.dim() == 2:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (weights.modulation.tensor + embed0).chunk(6, dim=1)
        if self.clean_cuda_cache:
            del embed0
            torch.cuda.empty_cache()

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def infer_self_attn(self, weights, grid_sizes, x, seq_lens, freqs, shift_msa, scale_msa):
        if hasattr(weights, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze(0)) * weights.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze(0) * weights.smooth_norm1_bias.tensor
        else:
            norm1_weight = 1 + scale_msa.squeeze(0)
            norm1_bias = shift_msa.squeeze(0)

        norm1_out = weights.norm1.apply(x)

        if GET_DTYPE() != "BF16":
            norm1_out = norm1_out.float()

        norm1_out.mul_(norm1_weight).add_(norm1_bias)

        if GET_DTYPE() != "BF16":
            norm1_out = norm1_out.to(torch.bfloat16)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim

        q = weights.self_attn_norm_q.apply(weights.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = weights.self_attn_norm_k.apply(weights.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = weights.self_attn_v.apply(norm1_out).view(s, n, d)

        if not self.parallel_attention:
            if self.config.get("audio_sr", False):
                freqs_i = compute_freqs_audio(q.size(2) // 2, grid_sizes, freqs)
            else:
                freqs_i = compute_freqs(q.size(2) // 2, grid_sizes, freqs)
        else:
            if self.config.get("audio_sr", False):
                freqs_i = compute_freqs_audio_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)
            else:
                freqs_i = compute_freqs_dist(q.size(0), q.size(2) // 2, grid_sizes, freqs)

        freqs_i = self.zero_temporal_component_in_3DRoPE(seq_lens, freqs_i)

        q = self.apply_rotary_emb_func(q, freqs_i)
        k = self.apply_rotary_emb_func(k, freqs_i)

        k_lens = torch.empty_like(seq_lens).fill_(freqs_i.size(0))
        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)

        if self.clean_cuda_cache:
            del freqs_i, norm1_out, norm1_weight, norm1_bias
            torch.cuda.empty_cache()

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
                mask_map=self.mask_map,
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

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch.cuda.empty_cache()

        return y

    def infer_cross_attn(self, weights, x, context, y_out, gate_msa):
        if GET_DTYPE() != "BF16":
            x = x.float() + y_out.float() * gate_msa.squeeze(0)
        else:
            x.add_(y_out * gate_msa.squeeze(0))

        norm3_out = weights.norm3.apply(x)
        if self.task == "i2v":
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if GET_DTYPE() != "BF16":
            context = context.to(torch.bfloat16)
            if self.task == "i2v":
                context_img = context_img.to(torch.bfloat16)

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
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch.cuda.empty_cache()

        attn_out = weights.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch.cuda.empty_cache()
        return x, attn_out

    def infer_ffn(self, weights, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch.cuda.empty_cache()

        if hasattr(weights, "smooth_norm2_weight"):
            norm2_weight = (1 + c_scale_msa.squeeze(0)) * weights.smooth_norm2_weight.tensor
            norm2_bias = c_shift_msa.squeeze(0) * weights.smooth_norm2_bias.tensor
        else:
            norm2_weight = 1 + c_scale_msa.squeeze(0)
            norm2_bias = c_shift_msa.squeeze(0)

        norm2_out = weights.norm2.apply(x)
        if GET_DTYPE() != "BF16":
            norm2_out = norm2_out.float()
        norm2_out.mul_(norm2_weight).add_(norm2_bias)
        if GET_DTYPE() != "BF16":
            norm2_out = norm2_out.to(torch.bfloat16)

        y = weights.ffn_0.apply(norm2_out)
        if self.clean_cuda_cache:
            del norm2_out, x, norm2_weight, norm2_bias
            torch.cuda.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()
        y = weights.ffn_2.apply(y)

        return y

    def post_process(self, x, y, c_gate_msa):
        if GET_DTYPE() != "BF16":
            x = x.float() + y.float() * c_gate_msa.squeeze(0)
        else:
            x.add_(y * c_gate_msa.squeeze(0))

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch.cuda.empty_cache()
        return x
