import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

from lightx2v.common.kvcache.quant import CalibRollingKVCachePool
from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.wan.infer.lingbot.transformer_infer import WanLingbotTransformerInfer
from lightx2v.models.networks.wan.infer.self_forcing.transformer_infer import causal_rope_apply
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanLingbotFastTransformerInfer(WanLingbotTransformerInfer):
    def __init__(self, config):
        super().__init__(config)
        ar = config.get("ar_config", {})
        self.num_frame_per_chunk = ar.get("num_frame_per_chunk", 3)
        self._ar_kv_offload: bool = bool(ar.get("kv_offload", False))
        if self._ar_kv_offload:
            self.infer_block_func = self.infer_block_with_kvoffload
        else:
            self.infer_block_func = self.infer_block_with_kvcache

        # Weight CPU↔GPU block streaming (WeightAsyncStreamManager) — independent of
        # ``infer_block_func`` (KV cache CPU offload vs on-GPU).
        self._weight_offload_block_compute = False
        cpu_off = self.config.get("cpu_offload", False)
        gran = self.config.get("offload_granularity", "block")
        if cpu_off and gran == "block":
            self.offload_manager = WeightAsyncStreamManager(offload_granularity="block")
            self.lazy_load = self.config.get("lazy_load", False)
            if self.lazy_load:
                self.offload_manager.init_lazy_load(
                    num_workers=self.config.get("num_disk_workers", 4),
                )
            self.infer_func = self.infer_with_kvcache_blocks_offload
            self._weight_offload_block_compute = True
        elif cpu_off:
            logger.warning(
                "[WanLingbotFastTransformerInfer] cpu_offload with offload_granularity={!r} does not use "
                "block weight streaming; falling back to infer_with_kvcache. Use offload_granularity='block' "
                "to enable infer_with_kvcache_blocks_offload (WeightAsyncStreamManager).",
                gran,
            )
            self.infer_func = self.infer_with_kvcache
        else:
            self.infer_func = self.infer_with_kvcache

    def _apply_rope_sp(self, q, k, grid_sizes, freqs, start_frame):
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
        pos_freqs = torch.chunk(pos_freqs, world_size, dim=0)[cur_rank][: q.size(0)]

        n = q.size(1)
        q_c = torch.view_as_complex(q.to(torch.float64).reshape(q.size(0), n, -1, 2))
        k_c = torch.view_as_complex(k.to(torch.float64).reshape(k.size(0), n, -1, 2))
        q = torch.view_as_real(q_c * pos_freqs).flatten(2).type_as(q)
        k = torch.view_as_real(k_c * pos_freqs).flatten(2).type_as(k)
        return q, k

    # def _sp_kvcache_attn(self, q, k_cache, v_cache, phase):
    #     kv_cache = self.kv_cache_manager.self_attn_kv_cache
    #     return kv_cache.sp_kvcache_attn(
    #         q=q,
    #         k_cache=k_cache,
    #         v_cache=v_cache,
    #         attention_module=phase.self_attn_1,
    #         seq_p_group=self.seq_p_group,
    #         num_heads=self.num_heads,
    #         head_dim=self.head_dim,
    #     )

    def _calculate_q_k_len(self, q, k_lens):
        q_lens = torch.tensor([q.size(0)], dtype=torch.int32, device=q.device)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        return cu_seqlens_q, cu_seqlens_k

    def infer_with_kvcache(self, blocks, x, pre_infer_out):
        mgr = self.kv_cache_manager
        self.kv_cache_size = mgr.kv_size
        self.max_attention_size = mgr.max_attention_size
        self._kv_offload = self._ar_kv_offload
        kv_cache = mgr.self_attn_kv_cache
        num_blocks = len(blocks)

        if self._kv_offload:
            kv_cache.prefetch_initial(list(range(min(2, num_blocks))))

        for block_idx in range(num_blocks):
            self.block_idx = block_idx
            if self._kv_offload:
                self._next_prefetch = block_idx + 2 if block_idx + 2 < num_blocks else None
            x = self.infer_block_func(blocks[block_idx], x, pre_infer_out)

        if self._kv_offload:
            comp = getattr(kv_cache, "compute_stream", None)
            if comp is not None:
                comp.synchronize()
            kv_cache.sync_all()
        return x

    def infer_with_kvcache_blocks_offload(self, blocks, x, pre_infer_out):
        mgr = self.kv_cache_manager
        self.kv_cache_size = mgr.kv_size
        self.max_attention_size = mgr.max_attention_size
        self._kv_offload = self._ar_kv_offload
        kv_cache = mgr.self_attn_kv_cache
        num_blocks = len(blocks)

        if self._kv_offload:
            kv_cache.prefetch_initial(list(range(min(2, num_blocks))))

        for block_idx in range(num_blocks):
            self.block_idx = block_idx
            if self._kv_offload:
                self._next_prefetch = block_idx + 2 if block_idx + 2 < num_blocks else None

            if self.offload_manager.need_init_first_buffer:
                self.offload_manager.init_first_buffer(blocks)
    
            self.offload_manager.prefetch_weights((block_idx + 1) % num_blocks, blocks)
            gpu_block = self.offload_manager.cuda_buffers[0]
            if AI_DEVICE == "xpu":
                x = self.infer_block_func(gpu_block, x, pre_infer_out)
            else:
                with torch_device_module.stream(self.offload_manager.compute_stream):
                    x = self.infer_block_func(gpu_block, x, pre_infer_out)

            self.offload_manager.swap_blocks()

        if self.clean_cuda_cache:
            del pre_infer_out.embed0, pre_infer_out.context
            torch_device_module.empty_cache()

        if self._kv_offload:
            if self._weight_offload_block_compute and AI_DEVICE == "cuda":
                self.offload_manager.compute_stream.synchronize()
            else:
                comp = getattr(kv_cache, "compute_stream", None)
                if comp is not None:
                    comp.synchronize()
            kv_cache.sync_all()
        return x

    def infer_block_with_kvoffload(self, block, x, pre_infer_out):
        """``begin_layer`` + optional ``kvcache.compute_stream`` around :meth:`infer_block_with_kvcache`.

        When weight block offload is active, compute already runs on ``offload_manager.compute_stream``;
        do not nest ``kvcache.compute_stream`` so ``begin_layer`` / attention share that stream.
        """
        kv_cache = self.kv_cache_manager.self_attn_kv_cache
        if self._weight_offload_block_compute:
            kv_cache.begin_layer(self.block_idx)
            return self.infer_block_with_kvcache(block, x, pre_infer_out)
        comp = getattr(kv_cache, "compute_stream", None)
        if comp is not None:
            with torch_device_module.stream(comp):
                kv_cache.begin_layer(self.block_idx)
                return self.infer_block_with_kvcache(block, x, pre_infer_out)
        kv_cache.begin_layer(self.block_idx)
        return self.infer_block_with_kvcache(block, x, pre_infer_out)

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

        return x

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
        q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(s, n, d)
        k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(s, n, d)
        v = phase.self_attn_v.apply(norm1_out).view(s, n, d)

        seg_index = int(self.scheduler.seg_index)
        current_start_frame = seg_index * self.num_frame_per_chunk

        if self.config.get("seq_parallel", False):
            q, k = self._apply_rope_sp(q, k, grid_sizes, freqs, current_start_frame)
        else:
            q = causal_rope_apply(q.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]
            k = causal_rope_apply(k.unsqueeze(0), grid_sizes, freqs, start_frame=current_start_frame).type_as(v)[0]

        kv_cache = self.kv_cache_manager.self_attn_kv_cache

        num_new = int(q.size(0))
        current_start = seg_index * num_new
        current_end = current_start + num_new
        global_end = kv_cache.get_global_end(self.block_idx)
        local_end = kv_cache.get_local_end(self.block_idx)
        local_per_frame = num_new // self.num_frame_per_chunk if self.num_frame_per_chunk > 0 else 0
        sink_tokens = self.kv_cache_manager.sink_size * local_per_frame

        if self.kv_cache_manager.local_attn_size != -1 and current_end > global_end and num_new + local_end > self.kv_cache_size:
            num_evicted = num_new + local_end - self.kv_cache_size
            kv_cache.roll_window(self.block_idx, sink_tokens, num_evicted)
            local_end_idx = local_end + current_end - global_end - num_evicted
        else:
            local_end_idx = local_end + current_end - global_end
        local_start_idx = local_end_idx - num_new

        kv_cache.store_kv(k, v, local_start_idx, local_end_idx, self.block_idx)
        kv_cache.set_ends(self.block_idx, current_end, local_end_idx)
        attn_start = max(0, local_end_idx - self.max_attention_size)

        if self.clean_cuda_cache:
            del norm1_out, norm1_weight, norm1_bias
            torch_device_module.empty_cache()

        if self.config.get("seq_parallel", False):
            attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
            attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)
            attn_out = kv_cache.sp_kvcache_attn(
                q=q,
                k_cache=attn_k,
                v_cache=attn_v,
                attention_module=phase.self_attn_1,
                seq_p_group=self.seq_p_group,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                attn_start=attn_start,
                local_end=local_end_idx,
            )
        else:
            attn_k = kv_cache.k_cache(self.block_idx, attn_start, local_end_idx)
            attn_v = kv_cache.v_cache(self.block_idx, attn_start, local_end_idx)

            if isinstance(kv_cache, CalibRollingKVCachePool) and not getattr(
                self.scheduler,
                "is_rerun",
                False,
            ):
                kv_cache.capture_attn(self.block_idx, attn_start, local_end_idx)
            if isinstance(attn_k, tuple):
                k_lens = torch.empty_like(seq_lens).fill_(attn_k[0].size(0))
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
                max_seqlen_kv=attn_k.size(0) if not isinstance(attn_k, tuple) else attn_k[0].size(0),
            )

        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, attn_out
            torch_device_module.empty_cache()
        if self._kv_offload:
            self.kv_cache_manager.self_attn_kv_cache.end_layer(
                self.block_idx,
                next_prefetch=self._next_prefetch,
            )
        return y

    def infer_cross_attn_with_kvcache(self, phase, x, context, y_out, gate_msa, block=None, conditional_dict=None):
        num_frames = gate_msa.shape[0]
        frame_seqlen = x.shape[0] // num_frames
        seg_index = self.scheduler.seg_index

        x.add_((y_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen)) * gate_msa).flatten(0, 1))

        if conditional_dict and "c2ws_plucker_emb" in conditional_dict and block is not None:
            cam = conditional_dict["c2ws_plucker_emb"]
            if cam.dim() == 3:
                cam = cam.squeeze(0)
            if cam.shape[0] < x.shape[0]:
                cam = F.pad(cam, (0, 0, 0, x.shape[0] - cam.shape[0]))
            elif cam.shape[0] > x.shape[0]:
                cam = cam[: x.shape[0]]
            cam = cam.to(dtype=x.dtype, device=x.device)
            cam_hidden = block.cam_injector_layer2.apply(F.silu(block.cam_injector_layer1.apply(cam))) + cam
            x = (1.0 + block.cam_scale_layer.apply(cam_hidden)) * x + block.cam_shift_layer.apply(cam_hidden)

        norm3_out = phase.norm3.apply(x)

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True):
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(self.infer_dtype)

        n, d = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, n, d)

        cross_kv_cache = self.kv_cache_manager.cross_attn_kv_cache

        if seg_index == 0:
            k = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, n, d)
            v = phase.cross_attn_v.apply(context).view(-1, n, d)
            cross_kv_cache.store_kv(k, v, self.block_idx)
            self._cross_kv_len = k.size(0)
        else:
            L = self._cross_kv_len
            k = cross_kv_cache.k_cache(self.block_idx)[:L]
            v = cross_kv_cache.v_cache(self.block_idx)[:L]

        cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
            q,
            k_lens=torch.tensor([k.size(0)], dtype=torch.int32),
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

        if context_img is not None:
            k_img = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, n, d)
            v_img = phase.cross_attn_v_img.apply(context_img).view(-1, n, d)
            cu_seqlens_q, cu_seqlens_k = self._calculate_q_k_len(
                q,
                k_lens=torch.tensor([k_img.size(0)], dtype=torch.int32),
            )
            attn_out.add_(
                phase.cross_attn_2.apply(
                    q=q,
                    k=k_img,
                    v=v_img,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_k,
                    max_seqlen_q=q.size(0),
                    max_seqlen_kv=k_img.size(0),
                )
            )

            if self.clean_cuda_cache:
                del k_img, v_img
                torch_device_module.empty_cache()

        attn_out = phase.cross_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del q, k, v, norm3_out, context, context_img
            torch_device_module.empty_cache()
        return x, attn_out

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa):
        x.add_(attn_out)

        if self.clean_cuda_cache:
            del attn_out
            torch.cuda.empty_cache()

        num_frames = c_shift_msa.shape[0]
        frame_seqlen = x.shape[0] // c_shift_msa.shape[0]

        norm2_weight = 1 + c_scale_msa
        norm2_bias = c_shift_msa

        norm2_out = phase.norm2.apply(x)
        norm2_out = norm2_out.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        norm2_out.mul_(norm2_weight).add_(norm2_bias)
        norm2_out = norm2_out.flatten(0, 1)

        y = phase.ffn_0.apply(norm2_out)
        if self.clean_cuda_cache:
            del norm2_out, x, norm2_weight, norm2_bias
            torch.cuda.empty_cache()
        y = torch.nn.functional.gelu(y, approximate="tanh")
        if self.clean_cuda_cache:
            torch.cuda.empty_cache()
        y = phase.ffn_2.apply(y)

        return y

    def post_process(self, x, y, c_gate_msa, pre_infer_out=None):
        num_frames = c_gate_msa.shape[0]
        frame_seqlen = x.shape[0] // c_gate_msa.shape[0]
        y = y.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))
        x.add_(y * c_gate_msa)
        x = x.flatten(0, 1)

        if self.clean_cuda_cache:
            del y, c_gate_msa
            torch.cuda.empty_cache()
        return x

    def infer_non_blocks(self, weights, x, e):
        num_frames = e.shape[0]
        frame_seqlen = x.shape[0] // e.shape[0]

        x = weights.norm.apply(x)
        x = x.unflatten(dim=0, sizes=(num_frames, frame_seqlen))

        t = self.scheduler.timestep_input
        e = e.unflatten(dim=0, sizes=t.shape).unsqueeze(2)
        modulation = weights.head_modulation.tensor
        e = (modulation.unsqueeze(1) + e).chunk(2, dim=2)

        x.mul_(1 + e[1][0]).add_(e[0][0])
        x = x.flatten(0, 1)
        x = weights.head.apply(x)

        if self.clean_cuda_cache:
            del e
            torch.cuda.empty_cache()
        return x
