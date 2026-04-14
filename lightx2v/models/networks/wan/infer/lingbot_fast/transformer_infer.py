import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import (
    WanSFTransformerInfer,
    causal_rope_apply,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanLingbotFastTransformerInfer(WanSFTransformerInfer):
    """Fast (autoregressive) transformer infer with lingbot camera injection and KV cache.

    Fixes over base WanSFTransformerInfer:
    - KV/cross-attn cache uses actual num_heads/head_dim (not hardcoded 12/128)
    - causal_rope_apply works with sequence parallelism
    - KV cache indexing uses actual token count (not hardcoded frame_seq_length)
    """

    def __init__(self, config):
        super().__init__(config)
        self._text_len = config.get("text_len", 512)

    def _sp_world_size(self):
        if self.config.get("seq_parallel", False) and dist.is_initialized():
            return dist.get_world_size(self.seq_p_group)
        return 1

    def reinit_caches(self, frame_seq_length, num_output_frames, text_len=None):
        self.frame_seq_length = frame_seq_length
        self._kv_size = frame_seq_length * num_output_frames
        if text_len is not None:
            self._text_len = text_len
        ws = self._sp_world_size()
        cfg_max = self.config.get("sf_config", {}).get("max_attention_size", None)
        if cfg_max is not None:
            self.max_attention_size = cfg_max // ws
        elif self.local_attn_size == -1:
            self.max_attention_size = self._kv_size // ws
        else:
            self.max_attention_size = self.local_attn_size * frame_seq_length // ws

        self._initialize_kv_cache(self.dtype, self.device)
        self._initialize_crossattn_cache(self.dtype, self.device)

    def _initialize_kv_cache(self, dtype, device):
        if not hasattr(self, "_kv_size"):
            return
        kv_cache1 = []
        ws = self._sp_world_size()
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length // ws
        else:
            kv_cache_size = self._kv_size // ws
        self.kv_cache_size = kv_cache_size

        n, d = self.num_heads, self.head_dim
        if self.kv_quant_config is not None:
            k_bit = self.kv_quant_config["k_bit"]
            v_bit = self.kv_quant_config["v_bit"]
            self.k_cache_dtype = torch.float8_e4m3fn if k_bit == "e4m3" else torch.float8_e5m2
            self.v_cache_dtype = torch.float8_e4m3fn if v_bit == "e4m3" else torch.float8_e5m2
        else:
            self.k_cache_dtype = None
            self.v_cache_dtype = None

        for _ in range(self.config["num_layers"]):
            if self.k_cache_dtype is not None:
                entry = {
                    "k": torch.zeros((self.kv_cache_size, n, d), dtype=self.k_cache_dtype, device=self.device),
                    "v": torch.zeros((self.kv_cache_size, n, d), dtype=self.v_cache_dtype, device=self.device),
                    "k_scales": torch.zeros((self.kv_cache_size, n, 1), dtype=GET_DTYPE(), device=self.device),
                    "v_scales": torch.zeros((self.kv_cache_size, n, 1), dtype=GET_DTYPE(), device=self.device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=self.device),
                }
            else:
                entry = {
                    "k": torch.zeros((self.kv_cache_size, n, d)).to(dtype).to(device),
                    "v": torch.zeros((self.kv_cache_size, n, d)).to(dtype).to(device),
                    "global_end_index": torch.tensor([0], dtype=torch.long).to(device),
                    "local_end_index": torch.tensor([0], dtype=torch.long).to(device),
                }
            kv_cache1.append(entry)

        self.kv_cache1_default = kv_cache1

    def _initialize_crossattn_cache(self, dtype, device):
        if not hasattr(self, "_kv_size"):
            return
        crossattn_cache = []
        n, d = self.num_heads, self.head_dim
        # Align with source: cross_kv_shape = [batch, max_sequence_length, num_heads, head_dim]
        text_len = self._text_len
        for _ in range(self.config["num_layers"]):
            crossattn_cache.append(
                {
                    "k": torch.zeros((text_len, n, d)).to(dtype).to(device),
                    "v": torch.zeros((text_len, n, d)).to(dtype).to(device),
                }
            )
        self.crossattn_cache_default = crossattn_cache

    def _apply_rope_sp(self, q, k, grid_sizes, freqs, start_frame):
        """Apply causal RoPE correctly when tokens are split across GPUs."""
        f, h, w = grid_sizes[0].tolist()
        full_seq_len = f * h * w
        c = q.size(-1) // 2

        freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        pos_freqs = torch.cat(
            [
                freqs_split[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(full_seq_len, 1, -1)

        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        multiple = world_size * f
        padding_size = (multiple - (full_seq_len % multiple)) % multiple
        if padding_size > 0:
            pos_freqs = F.pad(pos_freqs, (0, 0, 0, 0, 0, padding_size))
        pos_freqs = torch.chunk(pos_freqs, world_size, dim=0)[cur_rank]

        actual_len = q.size(0)
        pos_freqs = pos_freqs[:actual_len]

        n = q.size(1)
        q_c = torch.view_as_complex(q.to(torch.float64).reshape(actual_len, n, -1, 2))
        k_c = torch.view_as_complex(k.to(torch.float64).reshape(actual_len, n, -1, 2))
        q = torch.view_as_real(q_c * pos_freqs).flatten(2).type_as(q)
        k = torch.view_as_real(k_c * pos_freqs).flatten(2).type_as(k)
        return q, k

    @staticmethod
    def _a2a_seq_to_heads(x, world_size, shard_heads, group):
        """[local_seq, all_heads, dim] -> [full_seq, shard_heads, dim]"""
        local_seq, _, dim = x.shape
        x = x.reshape(local_seq, world_size, shard_heads, dim)
        x = x.permute(1, 0, 2, 3).contiguous()  # [world_size, local_seq, shard_heads, dim]
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        # out[i] = rank i's local tokens for this rank's head shard — keep contiguous
        return out.reshape(local_seq * world_size, shard_heads, dim)

    @staticmethod
    def _a2a_heads_to_seq(x, world_size, shard_heads, group):
        """[full_seq, shard_heads, dim] -> [local_seq, all_heads, dim]"""
        full_seq, _, dim = x.shape
        local_seq = full_seq // world_size
        x = x.reshape(world_size, local_seq, shard_heads, dim).contiguous()
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x, group=group)
        return out.permute(1, 0, 2, 3).reshape(local_seq, world_size * shard_heads, dim)

    def _sp_kvcache_attn(self, q, k_cache, v_cache, phase):
        """Self-attention with KV cache under sequence parallelism.

        The standard Ulysses all-to-all assumes Q/K/V have the same "image"
        length, but with KV cache K/V is longer than Q.  We do separate
        all-to-all for Q and K/V so the full history is properly assembled.
        """
        world_size = dist.get_world_size(self.seq_p_group)
        shard_heads = self.num_heads // world_size
        d = self.head_dim

        full_q = self._a2a_seq_to_heads(q, world_size, shard_heads, self.seq_p_group)
        full_k = self._a2a_seq_to_heads(k_cache, world_size, shard_heads, self.seq_p_group)
        full_v = self._a2a_seq_to_heads(v_cache, world_size, shard_heads, self.seq_p_group)

        q_lens = torch.tensor([full_q.size(0)], dtype=torch.int32, device=full_q.device)
        k_lens = torch.tensor([full_k.size(0)], dtype=torch.int32, device=full_k.device)
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)

        attn_out = phase.self_attn_1.apply(
            q=full_q,
            k=full_k,
            v=full_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_k,
            max_seqlen_q=full_q.size(0),
            max_seqlen_kv=full_k.size(0),
        )

        # flash_attn returns 2D [seq, shard_heads*dim]; reshape to 3D for all-to-all
        attn_out = attn_out.view(full_q.size(0), shard_heads, d)
        attn_out = self._a2a_heads_to_seq(attn_out, world_size, shard_heads, self.seq_p_group)
        # flatten back to 2D [local_seq, all_heads*dim]
        return attn_out.reshape(q.size(0), self.num_heads * d)

    # ---- Override self-attention to fix RoPE and KV cache indexing ----

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

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        s, n, d = *norm1_out.shape[:1], self.num_heads, self.head_dim

        q0 = phase.self_attn_q.apply(norm1_out)
        k0 = phase.self_attn_k.apply(norm1_out)

        q = phase.self_attn_norm_q.apply(q0).view(s, n, d)
        k = phase.self_attn_norm_k.apply(k0).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        seg_index = int(self.scheduler.seg_index)
        frame_seqlen = grid_sizes[0][1:].prod().item()
        current_start_frame = seg_index * self.num_frame_per_block

        if self.config.get("seq_parallel", False):
            q, k = self._apply_rope_sp(q, k, grid_sizes, freqs, current_start_frame)
        else:
            q = causal_rope_apply(q.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
            k = causal_rope_apply(k.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]

        num_new_tokens = int(q.size(0))
        # Use num_new_tokens for KV cache positioning — it already adapts to SP
        # (with SP each rank holds total_tokens/world_size per segment).
        # Using frame_seqlen (full spatial) would leave gaps in per-rank caches.
        current_start = seg_index * num_new_tokens
        current_end = current_start + num_new_tokens
        kv_cache = self.kv_cache1[self.block_idx]
        local_per_frame = num_new_tokens // self.num_frame_per_block if self.num_frame_per_block > 0 else 0
        sink_tokens = self.sink_size * local_per_frame

        global_end = int(kv_cache["global_end_index"].item())
        local_end = int(kv_cache["local_end_index"].item())

        if self.local_attn_size != -1 and (current_end > global_end) and (num_new_tokens + local_end > self.kv_cache_size):
            num_evicted_tokens = num_new_tokens + local_end - self.kv_cache_size
            num_rolled_tokens = local_end - num_evicted_tokens - sink_tokens
            src_start = sink_tokens + num_evicted_tokens
            src_end = src_start + num_rolled_tokens
            dst_start = sink_tokens
            dst_end = dst_start + num_rolled_tokens
            kv_cache["k"][dst_start:dst_end] = kv_cache["k"][src_start:src_end].clone()
            kv_cache["v"][dst_start:dst_end] = kv_cache["v"][src_start:src_end].clone()
            if self.kv_quant_config is not None:
                kv_cache["k_scales"][dst_start:dst_end] = kv_cache["k_scales"][src_start:src_end].clone()
                kv_cache["v_scales"][dst_start:dst_end] = kv_cache["v_scales"][src_start:src_end].clone()
            local_end_index = local_end + current_end - global_end - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
        else:
            local_end_index = local_end + current_end - global_end
            local_start_index = local_end_index - num_new_tokens

        if self.kv_quant_config is not None:
            s0, s1, s2 = k.shape
            k_2d = k.view(s0 * s1, s2)
            v_2d = v.view(s0 * s1, s2)
            k_q, k_scales = self.quant_fp8_vllm(k_2d)
            v_q, v_scales = self.quant_fp8_vllm(v_2d)
            kv_cache["k"][local_start_index:local_end_index] = k_q.view(s0, s1, s2)
            kv_cache["v"][local_start_index:local_end_index] = v_q.view(s0, s1, s2)
            kv_cache["k_scales"][local_start_index:local_end_index] = k_scales.view(s0, s1, 1)
            kv_cache["v_scales"][local_start_index:local_end_index] = v_scales.view(s0, s1, 1)
        else:
            kv_cache["k"][local_start_index:local_end_index] = k
            kv_cache["v"][local_start_index:local_end_index] = v

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        attn_start = max(0, local_end_index - self.max_attention_size)
        if self.kv_quant_config is not None:
            k_fp8 = kv_cache["k"][attn_start:local_end_index]
            v_fp8 = kv_cache["v"][attn_start:local_end_index]
            k_sc = kv_cache["k_scales"][attn_start:local_end_index]
            v_sc = kv_cache["v_scales"][attn_start:local_end_index]
            attn_k = self.dequant_fp8_vllm(k_fp8, k_sc, self.dtype)
            attn_v = self.dequant_fp8_vllm(v_fp8, v_sc, self.dtype)
        else:
            attn_k = kv_cache["k"][attn_start:local_end_index]
            attn_v = kv_cache["v"][attn_start:local_end_index]

        if self.clean_cuda_cache:
            del norm1_out, norm1_weight, norm1_bias
            torch_device_module.empty_cache()

        if self.config.get("seq_parallel", False):
            attn_out = self._sp_kvcache_attn(q, attn_k, attn_v, phase)
        else:
            k_lens = torch.empty_like(seq_lens).fill_(attn_k.size(0))
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(q, k_lens=k_lens)
            attn_out = phase.self_attn_1.apply(
                q=q,
                k=attn_k,
                v=attn_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=attn_k.size(0),
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out, attn_k, attn_v
            torch_device_module.empty_cache()

        return y

    def infer_block_with_kvcache(self, block, x, pre_infer_out):
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
            block=block,
            conditional_dict=pre_infer_out.conditional_dict,
        )

        y = self.infer_ffn(block.compute_phases[2], x, attn_out, c_shift_msa, c_scale_msa)
        x = self.post_process(x, y, c_gate_msa, pre_infer_out)

        if self.has_post_adapter:
            x = self.infer_post_adapter(block.compute_phases[3], x, pre_infer_out)
        # print(x, x.shape)
        # exit()
        return x

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa, block=None, conditional_dict=None):
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // gate_msa.shape[0]
        seg_index = self.scheduler.seg_index

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))

        if conditional_dict and "c2ws_plucker_emb" in conditional_dict and block is not None:
            cam = conditional_dict["c2ws_plucker_emb"]
            if cam.dim() == 3:
                cam = cam.squeeze(0)
            if cam.shape[0] < x.shape[0]:
                cam = torch.nn.functional.pad(cam, (0, 0, 0, x.shape[0] - cam.shape[0]))
            elif cam.shape[0] > x.shape[0]:
                cam = cam[: x.shape[0]]
            cam = cam.to(dtype=x.dtype, device=x.device)
            cam_hidden = block.cam_injector_layer2.apply(torch.nn.functional.silu(block.cam_injector_layer1.apply(cam)))
            cam_hidden = cam_hidden + cam
            cam_scale = block.cam_scale_layer.apply(cam_hidden)
            cam_shift = block.cam_shift_layer.apply(cam_hidden)
            x = (1.0 + cam_scale) * x + cam_shift

        norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim

        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        if seg_index == 0:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)
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
        )

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True) and context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)

            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32, device=k.device),
            )
            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_k,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

            if self.clean_cuda_cache:
                del k_img, v_img, img_attn_out
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out
