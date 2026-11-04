import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from lightx2v.models.networks.wan.infer.module_io import GridOutput, WanPreInferModuleOutput
from lightx2v.models.networks.wan.infer.offload.transformer_infer import WanOffloadTransformerInfer
from lightx2v.models.networks.wan.infer.post_infer import WanPostInfer
from lightx2v.models.networks.wan.infer.transformer_infer import modulate

from .attention import SpatialAttentionSpec, get_local_attention_ops
from .compile import RelayCompileMixin, mark_sequence_dynamic, native_sdpa
from .rope import apply_rope


class RelayWanPreInfer:
    def __init__(self, config, geometry):
        self.config = config
        self.prepare_lr = self.native_x2_input if geometry.scale == 2 else self.native_x4_input

    @staticmethod
    def native_x2_input(lr):
        return F.pixel_unshuffle(lr, 8)

    @staticmethod
    def native_x4_input(lr):
        return F.pixel_unshuffle(F.interpolate(lr, scale_factor=4, mode="bilinear", align_corners=False), 16)

    def conditioning(self, weights):
        dim = self.config["freq_dim"]
        exponent = -math.log(10000) * torch.arange(dim // 2, device=weights.prompt.tensor.device, dtype=torch.float32) / (dim // 2)
        angles = 1000 * torch.exp(exponent)
        timestep = torch.cat((angles.cos(), angles.sin())).unsqueeze(0).to(weights.prompt.tensor.dtype)
        temb = weights.time2.apply(F.silu(weights.time1.apply(timestep)))
        modulation = weights.time_proj.apply(F.silu(temb)).view(6, -1)
        context = weights.text2.apply(F.gelu(weights.text1.apply(weights.prompt.tensor[0]), approximate="tanh"))
        return temb, modulation, context

    def infer(self, weights, latents, lr):
        hidden = weights.patch.apply(latents).flatten(2).transpose(1, 2)[0]
        lq = self.prepare_lr(lr)
        lq = F.silu(weights.lq_norm1.apply(weights.lq_conv1.apply(lq)))
        lq = F.silu(weights.lq_norm2.apply(weights.lq_conv2.apply(lq)))
        # Flattening also preserves nn.Linear's leading stride for one token.
        lq = lq.flatten(2).transpose(1, 2).flatten(0, 1)
        lq = weights.lq_out.apply(lq) if lq.is_contiguous() else weights.lq_out_unbiased.apply(lq) + weights.lq_out.bias
        return hidden + lq


class RelayWanTransformerInfer(RelayCompileMixin, WanOffloadTransformerInfer):
    """Wan execution/offload/FFN with RelayVSR's causal reference attention.

    Keep the shared block pipeline, offload, normal-size FFN and final residuals.
    Relay-specific math is limited to input preparation, local self attention,
    fixed-context KV, and the checkpoint's FP32 normalization boundaries.
    """

    def __init__(self, config, deployment, parallel, weights):
        family_config = {
            **deployment,
            "task": "sr",
            "seq_parallel": parallel.size > 1,
            "num_layers": config["num_layers"],
            "num_heads": config["num_attention_heads"],
            "dim": config["num_attention_heads"] * config["attention_head_dim"],
            "modulate_type": "torch",
            "mxfp8_fuse_enable": False,
            "clean_cuda_cache": False,
        }
        super().__init__(family_config)
        self.relay_config, self.parallel = config, parallel
        backend = deployment.get("attention_backend", "triton")
        self.local_attention = get_local_attention_ops(backend)[1]
        self.use_native_sdpa_leaf = backend == "torch_sdpa" and self.use_compile and deployment.get("cross_attention_backend", "torch_sdpa") == "torch_sdpa"
        # Norms and modulation run in FP32, but residuals round to the inference
        # dtype after every sublayer, matching the official distilled checkpoint.
        self.sensitive_layer_dtype = self.infer_dtype
        self.modulate_func = self.modulate_and_cast
        self.cross_cache = []
        if weights.offload:
            self.offload_manager.init_cuda_buffer(blocks_cuda_buffer=weights.offload_block_cuda_buffers)

    def modulate_and_cast(self, x, scale, shift):
        return modulate(x, scale, shift).to(self.infer_dtype)

    def prepare_cross_attention(self, blocks, context):
        self.cross_cache = []
        pre = WanPreInferModuleOutput(embed=None, grid_sizes=None, x=None, embed0=None, context=context, adapter_args={"prepare_cross_kv": True})
        # Use the same family traversal/offload path for initialization as for
        # inference. Each staged block's KV remains resident after it is evicted.
        self.infer_func(blocks, None, pre)
        return self.cross_cache

    def run_block(self, block_idx, block, x, pre_infer_out):
        if pre_infer_out.adapter_args.get("prepare_cross_kv", False):
            return self.infer_block(block, x, pre_infer_out)
        inputs = pre_infer_out.adapter_args
        cache = (inputs["cache"].k_cache(block_idx), inputs["cache"].v_cache(block_idx))
        attention = {"q_rope": inputs["q_rope"], "k_rope": inputs["k_rope"], "cache": cache, "cache_slot": inputs["frame"] % self.relay_config["temporal_context_frames"]}
        pre_infer_out = replace(pre_infer_out, context=self.cross_cache[block_idx])
        if self.use_compile:
            mark_sequence_dynamic(x, *cache, *inputs["q_rope"], *inputs["k_rope"])
        return super().run_block(block_idx, block, x, pre_infer_out, attention)

    def infer_block(self, block, x, pre_infer_out, self_attn_kwargs=None):
        if pre_infer_out.adapter_args.get("prepare_cross_kv", False):
            phase, context = block.compute_phases[1], pre_infer_out.context
            key = phase.cross_attn_norm_k.apply(phase.cross_attn_k.apply(context)).view(-1, self.num_heads, self.head_dim)
            value = phase.cross_attn_v.apply(context).view(-1, self.num_heads, self.head_dim)
            self.cross_cache.append((key, value))
            return x
        return super().infer_block(block, x, pre_infer_out, self_attn_kwargs=self_attn_kwargs)

    def infer_cross_attn(self, phase, x, context, y_out, gate_msa):
        # RelayVSR has one fixed prompt and no image/CFG branch. Keep its
        # persistent context KV here without changing the Wan family contract.
        x.add_(y_out * gate_msa.squeeze())
        query = phase.cross_attn_norm_q.apply(phase.cross_attn_q.apply(phase.norm3.apply(x))).view(-1, self.num_heads, self.head_dim)
        key, value = context
        if self.use_native_sdpa_leaf:
            attended = native_sdpa(query, key, value)
        else:
            attended = phase.cross_attn_1.apply(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=self.cross_attn_cu_seqlens_q,
                cu_seqlens_kv=self.cross_attn_cu_seqlens_kv,
                max_seqlen_q=query.size(0),
                max_seqlen_kv=key.size(0),
            )
        return x, phase.cross_attn_o.apply(attended)

    def infer_ffn(self, phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa=None):
        if x.shape[0] != 1:
            return super().infer_ffn(phase, x, attn_out, c_shift_msa, c_scale_msa, c_gate_msa)
        # A minimal padded LR image has a single Wan token. The base FFN squeezes
        # away that axis; handle only this boundary case in the specialization.
        x.add_(attn_out)
        normalized = self.modulate_func(phase.norm2.apply(x), scale=c_scale_msa, shift=c_shift_msa).reshape(1, -1)
        return phase.ffn_2.apply(F.gelu(phase.ffn_0.apply(normalized), approximate="tanh"))

    def infer_self_attn(self, phase, x, shift_msa, scale_msa, grid_sizes=None, *, q_rope, k_rope, cache, cache_slot):
        _, height, width = grid_sizes
        normalized = self.modulate_func(phase.norm1.apply(x), scale_msa, shift_msa)
        tokens = q_rope[0].shape[0]
        query = self.parallel.to_heads(phase.self_attn_norm_q.apply(phase.self_attn_q.apply(normalized)).view(-1, self.num_heads, self.head_dim), tokens)
        key = self.parallel.to_heads(phase.self_attn_norm_k.apply(phase.self_attn_k.apply(normalized)).view(-1, self.num_heads, self.head_dim), tokens)
        value = self.parallel.to_heads(phase.self_attn_v.apply(normalized).view(-1, self.num_heads, self.head_dim), tokens)
        start = cache_slot * tokens
        cache[0][start : start + tokens].copy_(key)
        cache[1][start : start + tokens].copy_(value)
        # Derive both views inside the graph. Independent overlapping inputs
        # can lose the write-before-read dependency during functionalization.
        key, value = cache[0][: k_rope[0].shape[0]], cache[1][: k_rope[0].shape[0]]
        query = apply_rope(query, q_rope)
        key = apply_rope(key, k_rope)
        visible = key.shape[0] // tokens
        wh, ww = self.relay_config["spatial_window_size"]
        if height > wh or width > ww:
            spec = SpatialAttentionSpec(1, height, width, wh, ww, self.relay_config["temporal_context_frames"], key_frames=visible, query_frame_offset=visible - 1)
            attended = self.local_attention(query[None], key[None], value[None], spec)[0]
        else:
            attended = phase.self_attn_1.apply(query, key, value).view(tokens, -1, self.head_dim)
        return phase.self_attn_o.apply(self.parallel.to_sequence(attended))

    def prepare_input(self, x, temb, modulation, context, rope, cache, frame, height, width):
        window = self.relay_config["temporal_context_frames"]
        visible = min(frame + 1, window)
        origin = max(0, frame - self.relay_config["temporal_rope"]["training_frames"] + 1)
        positions = tuple(frame - ((frame - slot) % window) - origin for slot in range(visible))
        grid = (1, height, width)
        return WanPreInferModuleOutput(
            embed=temb,
            grid_sizes=GridOutput(tensor=torch.tensor(grid), tuple=grid),
            x=x,
            embed0=modulation.float().unsqueeze(0),
            context=context,
            adapter_args={
                "q_rope": rope.for_positions(height, width, (frame - origin,)),
                "k_rope": rope.for_positions(height, width, positions),
                "cache": cache,
                "frame": frame,
            },
        )

    def infer_non_blocks(self, weights, x, e):
        # The official head modulates FP32-normalized activations with BF16
        # coefficients, then casts before projection. Gather SP tokens first.
        x = self.parallel.gather(x, self._output_tokens)
        shift, scale = (weights.post.modulation.tensor[0] + e).unbind(0)
        x = (weights.post.norm.apply(x) * (1 + scale) + shift).to(x.dtype)
        return weights.post.projection.apply(x)

    def infer_main_blocks(self, blocks, pre_infer_out):
        _, height, width = pre_infer_out.grid_sizes.tuple
        self._output_tokens = height * width
        return super().infer_main_blocks(blocks, pre_infer_out)


class RelayWanPostInfer(WanPostInfer):
    def infer(self, x, pre_infer_out):
        # Reuse Wan's unpatchify; the one-step scheduler expects a batched BF16
        # tensor instead of the base post-infer's list of FP32 videos.
        return self.unpatchify(x, pre_infer_out.grid_sizes.tuple)[0].unsqueeze(0)
