import torch

from lightx2v.models.networks.wan.infer.mxfp8_fuse import scaled_mxfp8_modulate_quant
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class WanTransformerInferCausVid(WanOffloadTransformerInfer):
    """Wan inference with persistent causal self/cross-attention caches."""

    def __init__(self, config):
        super().__init__(config)
        self.scheduler = None
        self.num_frames = config["num_frames"]
        self.frame_seq_length = config["frame_seq_length"]
        self.kv_cache = None
        self.crossattn_cache = None
        self._cache_meta = None
        self._cu_seqlens_cache = {}
        self._reset_request_cache()

    def set_scheduler(self, scheduler):
        super().set_scheduler(scheduler)
        self._reset_request_cache()
        self._kv_cache_request_id = scheduler.rope_request_id

    def _init_kv_cache(self, dtype, device, kv_size=None):
        kv_size = kv_size or self.num_frames * self.frame_seq_length
        self.kv_cache = [
            {
                "k": torch.zeros((kv_size, self.num_heads, self.head_dim), dtype=dtype, device=device),
                "v": torch.zeros((kv_size, self.num_heads, self.head_dim), dtype=dtype, device=device),
            }
            for _ in range(self.blocks_num)
        ]
        self.crossattn_cache = [
            {
                "k": None,
                "v": None,
                "k_img": None,
                "v_img": None,
                "is_init": False,
            }
            for _ in range(self.blocks_num)
        ]
        self._cache_meta = (dtype, torch.device(device), kv_size)

    def _ensure_attention_caches(self, x):
        configured_size = self.num_frames * self.frame_seq_length
        required_capacity = max(configured_size, self._kv_end, self._valid_kv_end)
        if self._cache_meta is None or self.kv_cache is None:
            self._init_kv_cache(x.dtype, x.device, required_capacity)
            return

        cached_dtype, cached_device, cached_capacity = self._cache_meta
        if cached_dtype == x.dtype and cached_device == x.device and cached_capacity >= required_capacity:
            return

        old_kv_cache = self.kv_cache
        valid_kv_end = self._valid_kv_end
        self._init_kv_cache(x.dtype, x.device, required_capacity)
        if valid_kv_end == 0:
            return

        for old_block_cache, new_block_cache in zip(old_kv_cache, self.kv_cache, strict=True):
            new_block_cache["k"][:valid_kv_end].copy_(old_block_cache["k"][:valid_kv_end])
            new_block_cache["v"][:valid_kv_end].copy_(old_block_cache["v"][:valid_kv_end])

    def _reset_request_cache(self):
        self._kv_cache_request_id = None
        self._kv_start = 0
        self._kv_end = 0
        self._valid_kv_end = 0
        self._cu_seqlens_cache.clear()
        if self.crossattn_cache is None:
            return
        for block_cache in self.crossattn_cache:
            block_cache["k"] = None
            block_cache["v"] = None
            block_cache["k_img"] = None
            block_cache["v_img"] = None
            block_cache["is_init"] = False

    def _sync_request_cache(self):
        request_id = self.scheduler.rope_request_id
        if request_id == self._kv_cache_request_id:
            return request_id

        self._reset_request_cache()
        self._kv_cache_request_id = request_id
        return request_id

    def _get_cu_seqlens(self, q_len, kv_len):
        key = (int(q_len), int(kv_len))
        cached = self._cu_seqlens_cache.get(key)
        if cached is None:
            cached = (
                torch.tensor([0, q_len], dtype=torch.int32),
                torch.tensor([0, kv_len], dtype=torch.int32),
            )
            self._cu_seqlens_cache[key] = cached
        return cached

    def reset_infer_states(self, x, context):
        query_len = x.shape[0]
        has_image_context = self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True)
        context_len = context.shape[0] - (257 if has_image_context else 0)

        self.self_attn_cu_seqlens_qkv, _ = self._get_cu_seqlens(query_len, query_len)
        self.cross_attn_cu_seqlens_q, self.cross_attn_cu_seqlens_kv = self._get_cu_seqlens(query_len, context_len)
        if has_image_context:
            _, self.cross_attn_cu_seqlens_kv_img = self._get_cu_seqlens(query_len, 257)

        if self.has_post_adapter:
            self.reset_post_adapter_states()

    @torch.no_grad()
    def infer(self, weights, pre_infer_out, kv_start, kv_end):
        if self.config["seq_parallel"]:
            raise NotImplementedError("Sequence parallel inference is not implemented for CausVid.")
        if self.scheduler is None:
            raise RuntimeError("WanTransformerInferCausVid scheduler is not initialized.")

        request_id = self._sync_request_cache()

        kv_start = int(kv_start)
        kv_end = int(kv_end)
        query_len = pre_infer_out.x.shape[0]
        if kv_start < 0 or kv_end <= kv_start:
            raise ValueError(f"Invalid CausVid KV range: [{kv_start}, {kv_end}).")
        if self._valid_kv_end == 0 and kv_start != 0:
            raise ValueError(f"CausVid request_id={request_id} must start at kv_start=0; got kv_start={kv_start}, kv_end={kv_end}.")
        if kv_start > self._valid_kv_end:
            raise ValueError(f"CausVid request_id={request_id} leaves an uninitialized KV gap: valid_kv_end={self._valid_kv_end}, kv_start={kv_start}, kv_end={kv_end}.")
        if kv_end - kv_start != query_len:
            raise ValueError(f"CausVid query length must match its KV cache range: query_len={query_len}, range=[{kv_start}, {kv_end}).")

        self._kv_start = kv_start
        self._kv_end = kv_end
        self._ensure_attention_caches(pre_infer_out.x)
        output = super().infer(weights, pre_infer_out)
        self._valid_kv_end = max(self._valid_kv_end, kv_end)
        return output

    def infer_self_attn(self, phase, x, shift_msa, scale_msa, grid_sizes=None):
        norm1_quant = None
        norm1_scale = None
        if hasattr(phase, "smooth_norm1_weight"):
            norm1_weight = (1 + scale_msa.squeeze()) * phase.smooth_norm1_weight.tensor
            norm1_bias = shift_msa.squeeze() * phase.smooth_norm1_bias.tensor
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            norm1_out.mul_(norm1_weight).add_(norm1_bias)
        else:
            norm1_out = phase.norm1.apply(x)
            if self.sensitive_layer_dtype != self.infer_dtype:
                norm1_out = norm1_out.to(self.sensitive_layer_dtype)
            if self._use_mxfp8_quant_fuse():
                self._ensure_mxfp8_quant_fuse_ready(
                    phase,
                    norm1_out,
                    scale_msa,
                    shift_msa,
                    module_names=("self_attn_q", "self_attn_k", "self_attn_v"),
                )
            if self._can_reuse_self_attn_mxfp8_quant(phase, norm1_out, scale_msa, shift_msa):
                norm1_quant, norm1_scale = scaled_mxfp8_modulate_quant(norm1_out, scale_msa, shift_msa)
            else:
                norm1_out = self.modulate_func(norm1_out, scale=scale_msa, shift=shift_msa).squeeze()

        if self.sensitive_layer_dtype != self.infer_dtype:
            norm1_out = norm1_out.to(self.infer_dtype)

        seq_len, num_heads, head_dim = norm1_out.shape[0], self.num_heads, self.head_dim
        if norm1_quant is not None:
            q = phase.self_attn_norm_q.apply(self._mxfp8_apply_quantized(phase.self_attn_q, norm1_quant, norm1_scale)).view(seq_len, num_heads, head_dim)
            k = phase.self_attn_norm_k.apply(self._mxfp8_apply_quantized(phase.self_attn_k, norm1_quant, norm1_scale)).view(seq_len, num_heads, head_dim)
            v = self._mxfp8_apply_quantized(phase.self_attn_v, norm1_quant, norm1_scale).view(seq_len, num_heads, head_dim)
        else:
            q = phase.self_attn_norm_q.apply(phase.self_attn_q.apply(norm1_out)).view(seq_len, num_heads, head_dim)
            k = phase.self_attn_norm_k.apply(phase.self_attn_k.apply(norm1_out)).view(seq_len, num_heads, head_dim)
            v = phase.self_attn_v.apply(norm1_out).view(seq_len, num_heads, head_dim)

        if self.rope_positions is None:
            q, k = phase.rope.apply(q, k, self.cos_sin)
        else:
            q, k = phase.rope.apply(q, k, self.cos_sin, positions=self.rope_positions)
        block_cache = self.kv_cache[self.block_idx]
        block_cache["k"][self._kv_start : self._kv_end].copy_(k)
        block_cache["v"][self._kv_start : self._kv_end].copy_(v)

        cu_seqlens_q, cu_seqlens_kv = self._get_cu_seqlens(seq_len, self._kv_end)
        attn_out = phase.self_attn_1.apply(
            q=q,
            k=block_cache["k"][: self._kv_end],
            v=block_cache["v"][: self._kv_end],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=seq_len,
            max_seqlen_kv=self._kv_end,
            block_idx=self.block_idx,
            scheduler=self.scheduler,
            grid_sizes=grid_sizes,
        )
        y = phase.self_attn_o.apply(attn_out)

        if self.clean_cuda_cache:
            del norm1_out, q, k, v, attn_out, shift_msa, scale_msa
            if norm1_quant is not None:
                del norm1_quant, norm1_scale
            torch_device_module.empty_cache()
        return y

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        if self.sensitive_layer_dtype != self.infer_dtype:
            x = x.to(self.sensitive_layer_dtype) + y_out.to(self.sensitive_layer_dtype) * gate_msa.squeeze()
        else:
            x.add_(y_out * gate_msa.squeeze())

        norm3_out = phase.norm3.apply(x)
        has_image_context = self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"] and self.config.get("use_image_encoder", True)
        if has_image_context:
            context_img = context[:257]
            context = context[257:]
        else:
            context_img = None

        if self.sensitive_layer_dtype != self.infer_dtype:
            context = context.to(self.infer_dtype)
            if context_img is not None:
                context_img = context_img.to(self.infer_dtype)

        num_heads, head_dim = self.num_heads, self.head_dim
        q = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(norm3_out)).view(-1, num_heads, head_dim)
        block_cache = self.crossattn_cache[self.block_idx]
        if not block_cache["is_init"]:
            block_cache["k"] = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, num_heads, head_dim)
            block_cache["v"] = phase.cross_attn_v.apply(context).view(-1, num_heads, head_dim)
            if context_img is not None:
                block_cache["k_img"] = phase.cross_attn_norm_k_img.apply(phase.cross_attn_k_img.apply(context_img)).view(-1, num_heads, head_dim)
                block_cache["v_img"] = phase.cross_attn_v_img.apply(context_img).view(-1, num_heads, head_dim)
            block_cache["is_init"] = True

        k = block_cache["k"]
        v = block_cache["v"]
        cu_seqlens_q, cu_seqlens_kv = self._get_cu_seqlens(q.size(0), k.size(0))
        attn_out = phase.cross_attn_1.apply(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=q.size(0),
            max_seqlen_kv=k.size(0),
        )

        k_img = block_cache["k_img"]
        v_img = block_cache["v_img"]
        if k_img is not None:
            cu_seqlens_q, cu_seqlens_kv_img = self._get_cu_seqlens(q.size(0), k_img.size(0))
            img_attn_out = phase.cross_attn_2.apply(
                q=q,
                k=k_img,
                v=v_img,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv_img,
                max_seqlen_q=q.size(0),
                max_seqlen_kv=k_img.size(0),
            )
            attn_out.add_(img_attn_out)

        attn_out = phase.cross_attn_o.apply(attn_out)
        if self.clean_cuda_cache:
            del q, norm3_out, context, context_img, y_out, gate_msa
            torch_device_module.empty_cache()
        return x, attn_out
