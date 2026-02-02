from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from lightx2v.common.offload.manager import WeightAsyncStreamManager
from lightx2v.models.networks.hunyuan_video.infer.module_io import (
    HunyuanVideo15ImgBranchOutput,
    HunyuanVideo15TxtBranchOutput,
)
from lightx2v.models.networks.hunyuan_video.infer.transformer_infer import (
    HunyuanVideo15TransformerInfer,
    apply_gate,
)
from lightx2v.models.networks.worldplay.prope.camera_rope import prope_qkv
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class KVCache:
    """
    KV Cache for autoregressive generation.

    Stores key and value tensors for each transformer block to enable
    efficient autoregressive generation without recomputing past tokens.

    Structure matches HY-WorldPlay original implementation:
    - k_txt, v_txt: Text KV cache (set once at the beginning)
    - k_vision, v_vision: Vision KV cache (accumulated for context frames)
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._cache: List[Dict[str, Optional[torch.Tensor]]] = []
        self._initialized = False

    def initialize(self):
        """Initialize cache structure for each block."""
        if self._initialized:
            return

        self._cache = []
        for _ in range(self.num_blocks):
            self._cache.append(
                {
                    "k_txt": None,
                    "v_txt": None,
                    "k_vision": None,
                    "v_vision": None,
                }
            )
        self._initialized = True

    def set_txt_cache(self, block_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Set text KV cache (called once at the beginning)."""
        self._cache[block_idx]["k_txt"] = k
        self._cache[block_idx]["v_txt"] = v

    def get_txt_cache(self, block_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get text KV cache."""
        return self._cache[block_idx]["k_txt"], self._cache[block_idx]["v_txt"]

    def set_vision_cache(self, block_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Set or append vision KV cache for context frames."""
        if self._cache[block_idx]["k_vision"] is None:
            self._cache[block_idx]["k_vision"] = k
            self._cache[block_idx]["v_vision"] = v
        else:
            self._cache[block_idx]["k_vision"] = torch.cat([self._cache[block_idx]["k_vision"], k], dim=2)
            self._cache[block_idx]["v_vision"] = torch.cat([self._cache[block_idx]["v_vision"], v], dim=2)

    def get_vision_cache(self, block_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get vision KV cache."""
        return self._cache[block_idx]["k_vision"], self._cache[block_idx]["v_vision"]

    def clear_vision_cache(self):
        """Clear only vision cache (keep text cache)."""
        for block_cache in self._cache:
            block_cache["k_vision"] = None
            block_cache["v_vision"] = None

    def clear(self):
        """Clear all cache."""
        self._cache = []
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized


def modulate_per_token(x, scale, shift):
    """Modulate with per-token scale and shift (no unsqueeze).

    Args:
        x: [L, C] tensor
        scale: [B, L, C] or [B, C] tensor
        shift: [B, L, C] or [B, C] tensor

    Returns:
        Modulated tensor [L, C]
    """
    if scale.dim() == 3:
        scale = scale.squeeze(0)
        shift = shift.squeeze(0)
        return x * (1 + scale) + shift
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WorldPlayARTransformerInfer(HunyuanVideo15TransformerInfer):
    """
    Transformer inference for WorldPlay AR (Autoregressive) model.

    Key differences from WorldPlayTransformerInfer (Distill):
    - Uses KV cache for efficient autoregressive generation
    - Supports separate text and vision inference phases
    - Text KV is cached once, vision KV is accumulated for context frames

    AR inference flow (matching HY-WorldPlay):
    1. infer_txt(): Cache text KV (called once at generation start)
    2. For each chunk:
       a. infer_vision(cache_vision=True): Cache context frame KV
       b. For each denoising step:
          infer_vision(cache_vision=False): Generate using cached KV
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_prope = config.get("use_prope", True)
        self.chunk_latent_frames = config.get("chunk_latent_frames", 4)

        # KV Cache - new structure matching HY-WorldPlay
        self._kv_cache: Optional[KVCache] = None

        # Causal attention - use torch_sdpa with causal=True
        self.causal_attention = ATTN_WEIGHT_REGISTER.get("torch_sdpa", None)
        if self.causal_attention is not None:
            self.causal_attention = self.causal_attention()

        # Setup offload if enabled
        if self.config.get("cpu_offload", False):
            offload_granularity = self.config.get("offload_granularity", "block")
            if offload_granularity == "block":
                self.infer_func = self.infer_with_blocks_offload
            elif offload_granularity == "model":
                self.infer_func = self.infer_without_offload
            else:
                raise NotImplementedError
            if offload_granularity != "model":
                self.offload_manager = WeightAsyncStreamManager(offload_granularity=offload_granularity)

    @property
    def _vec_is_per_token(self):
        """Check if vec is per-token from scheduler flag."""
        return getattr(self.scheduler, "vec_is_per_token", False)

    def init_kv_cache(self):
        """
        Initialize KV cache for autoregressive generation.
        Structure matches HY-WorldPlay original implementation.
        """
        self._kv_cache = KVCache(num_blocks=self.double_blocks_num)
        self._kv_cache.initialize()
        return self._kv_cache

    def clear_kv_cache(self):
        """Clear KV cache after generation."""
        if self._kv_cache is not None:
            self._kv_cache.clear()
            self._kv_cache = None

    def clear_vision_cache(self):
        """Clear only vision cache (keep text cache for next chunk)."""
        if self._kv_cache is not None:
            self._kv_cache.clear_vision_cache()

    @torch.no_grad()
    def _infer_img_branch_before_attn(self, weights, infer_module_out):
        """
        Override to handle per-token vec modulation for WorldPlay AR.

        Returns:
            Tuple of (img_q, img_k, img_v, img_q_pre_rope, img_k_pre_rope, img_branch_out)
        """
        vec = infer_module_out.vec
        img_seqlen = infer_module_out.img.shape[1]

        if self._vec_is_per_token and vec.dim() == 3:
            B, L, C = vec.shape
            vec_flat = vec.reshape(B * L, C)
            mod_output_flat = weights.img_branch.img_mod.apply(vec_flat)
            mod_output = mod_output_flat.reshape(B, L, -1)

            img_mod = mod_output[:, :img_seqlen, :]
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = img_mod.chunk(6, dim=-1)

            img_modulated = weights.img_branch.img_norm1.apply(infer_module_out.img.squeeze(0))
            img_modulated = img_modulated * (1 + img_mod1_scale.squeeze(0)) + img_mod1_shift.squeeze(0)
        else:
            mod_output = weights.img_branch.img_mod.apply(vec)
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = mod_output.chunk(6, dim=-1)
            img_modulated = weights.img_branch.img_norm1.apply(infer_module_out.img.squeeze(0))
            img_modulated = self.modulate_func(img_modulated, scale=img_mod1_scale, shift=img_mod1_shift).squeeze(0)

        img_q = weights.img_branch.img_attn_q.apply(img_modulated)
        img_k = weights.img_branch.img_attn_k.apply(img_modulated)
        img_v = weights.img_branch.img_attn_v.apply(img_modulated)
        img_q = rearrange(img_q, "L (H D) -> L H D", H=self.heads_num)
        img_k = rearrange(img_k, "L (H D) -> L H D", H=self.heads_num)
        img_v = rearrange(img_v, "L (H D) -> L H D", H=self.heads_num)
        img_q = weights.img_branch.img_attn_q_norm.apply(img_q)
        img_k = weights.img_branch.img_attn_k_norm.apply(img_k)

        # Save pre-RoPE Q/K for PRoPE branch
        img_q_pre_rope = img_q.unsqueeze(0)
        img_k_pre_rope = img_k.unsqueeze(0)

        # Apply RoPE for standard attention branch
        img_q, img_k = self.apply_rope_func(img_q.unsqueeze(0), img_k.unsqueeze(0), cos_sin_cache=self.scheduler.cos_sin)

        return (
            img_q,
            img_k,
            img_v.unsqueeze(0),
            img_q_pre_rope,
            img_k_pre_rope,
            HunyuanVideo15ImgBranchOutput(
                img_mod1_gate=img_mod1_gate,
                img_mod2_shift=img_mod2_shift,
                img_mod2_scale=img_mod2_scale,
                img_mod2_gate=img_mod2_gate,
            ),
        )

    @torch.no_grad()
    def _infer_txt_branch_before_attn(self, weights, infer_module_out):
        """Override to handle per-token vec modulation for text branch."""
        vec = infer_module_out.vec

        if self._vec_is_per_token and vec.dim() == 3:
            vec_for_txt = vec.mean(dim=1)
        else:
            vec_for_txt = vec

        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = weights.txt_branch.txt_mod.apply(vec_for_txt).chunk(6, dim=-1)

        txt_modulated = weights.txt_branch.txt_norm1.apply(infer_module_out.txt.squeeze(0))
        txt_modulated = self.modulate_func(txt_modulated, scale=txt_mod1_scale, shift=txt_mod1_shift).squeeze(0)

        txt_q = weights.txt_branch.txt_attn_q.apply(txt_modulated)
        txt_k = weights.txt_branch.txt_attn_k.apply(txt_modulated)
        txt_v = weights.txt_branch.txt_attn_v.apply(txt_modulated)
        txt_q = rearrange(txt_q, "L (H D) -> L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "L (H D) -> L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "L (H D) -> L H D", H=self.heads_num)
        txt_q = weights.txt_branch.txt_attn_q_norm.apply(txt_q).to(txt_v)
        txt_k = weights.txt_branch.txt_attn_k_norm.apply(txt_k).to(txt_v)

        return (
            txt_q.unsqueeze(0),
            txt_k.unsqueeze(0),
            txt_v.unsqueeze(0),
            HunyuanVideo15TxtBranchOutput(
                txt_mod1_gate=txt_mod1_gate,
                txt_mod2_shift=txt_mod2_shift,
                txt_mod2_scale=txt_mod2_scale,
                txt_mod2_gate=txt_mod2_gate,
            ),
        )

    @torch.no_grad()
    def _infer_causal_attn(self, weights, img_q, img_k, img_v, txt_q, txt_k, txt_v, block_idx=None, use_kv_cache=True):
        """
        Perform attention for AR model.

        Note: When running full-video inference (not chunk-by-chunk AR generation),
        we use bidirectional attention like the distill model. Causal attention
        is only needed for true autoregressive chunk-by-chunk generation.

        Args:
            weights: Block weights
            img_q/k/v: Image Q/K/V tensors
            txt_q/k/v: Text Q/K/V tensors
            block_idx: Block index for KV cache
            use_kv_cache: Whether to use KV cache

        Returns:
            Tuple of (img_attn, txt_attn)
        """
        img_seqlen = img_q.shape[1]

        # For standard inference, don't use KV cache
        cached_img_k, cached_img_v = img_k, img_v
        txt_cache_k, txt_cache_v = txt_k, txt_v

        # Concatenate image and text for joint attention
        query = torch.cat([img_q, txt_q], dim=1)
        key = torch.cat([cached_img_k, txt_cache_k], dim=1)
        value = torch.cat([cached_img_v, txt_cache_v], dim=1)

        # Use bidirectional attention for full-video inference
        # (causal=False, same as distill model)
        seqlen = query.shape[1]
        cu_seqlens_qkv = torch.tensor([0, seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)
        attn_out = weights.self_attention.apply(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_seqlens_qkv,
            cu_seqlens_kv=cu_seqlens_qkv,
            max_seqlen_q=seqlen,
            max_seqlen_kv=seqlen,
        )

        # Split output back to image and text
        img_attn = attn_out[:img_seqlen]
        txt_attn = attn_out[img_seqlen:]

        return img_attn, txt_attn

    @torch.no_grad()
    def _infer_img_branch_after_attn(self, weights, img_attn, img, img_branch_out, img_attn_prope=None):
        """Override to handle per-token modulation in post-attention."""
        if self._vec_is_per_token and img_branch_out.img_mod2_scale.dim() == 3:
            img_seqlen = img.shape[1]
            img_mod2_scale = img_branch_out.img_mod2_scale[:, :img_seqlen, :]
            img_mod2_shift = img_branch_out.img_mod2_shift[:, :img_seqlen, :]
            img_mod1_gate = img_branch_out.img_mod1_gate[:, :img_seqlen, :]
            img_mod2_gate = img_branch_out.img_mod2_gate[:, :img_seqlen, :]

            attn_proj = weights.img_branch.img_attn_proj.apply(img_attn)
            if img_attn_prope is not None:
                attn_proj = attn_proj + img_attn_prope

            gated_attn = attn_proj * img_mod1_gate.squeeze(0)
            img = img + gated_attn.unsqueeze(0)

            img_squeezed = img.squeeze(0)
            normed = weights.img_branch.img_norm2.apply(img_squeezed)
            modulated = normed * (1 + img_mod2_scale.squeeze(0)) + img_mod2_shift.squeeze(0)
            out = weights.img_branch.img_mlp_fc1.apply(modulated)
            out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
            gated_out = out * img_mod2_gate.squeeze(0)
            img = img + gated_out.unsqueeze(0)
        else:
            attn_proj = weights.img_branch.img_attn_proj.apply(img_attn).unsqueeze(0)
            if img_attn_prope is not None:
                attn_proj = attn_proj + img_attn_prope.unsqueeze(0)
            img = img + apply_gate(attn_proj, gate=img_branch_out.img_mod1_gate)
            out = weights.img_branch.img_mlp_fc1.apply(
                self.modulate_func(weights.img_branch.img_norm2.apply(img.squeeze(0)), scale=img_branch_out.img_mod2_scale, shift=img_branch_out.img_mod2_shift).squeeze(0)
            )
            out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
            img = img + apply_gate(out.unsqueeze(0), gate=img_branch_out.img_mod2_gate)
        return img

    @torch.no_grad()
    def _infer_txt_branch_after_attn(self, weights, txt_attn, txt, txt_branch_out):
        """Standard text branch post-attention processing."""
        txt = txt + apply_gate(weights.txt_branch.txt_attn_proj.apply(txt_attn).unsqueeze(0), gate=txt_branch_out.txt_mod1_gate)
        out = weights.txt_branch.txt_mlp_fc1.apply(
            self.modulate_func(weights.txt_branch.txt_norm2.apply(txt.squeeze(0)), scale=txt_branch_out.txt_mod2_scale, shift=txt_branch_out.txt_mod2_shift).squeeze(0)
        )
        out = weights.txt_branch.txt_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
        txt = txt + apply_gate(out.unsqueeze(0), gate=txt_branch_out.txt_mod2_gate)
        return txt

    def _apply_prope(self, q, k, v, viewmats, Ks, grid_sizes):
        """
        Apply Projective Positional Encoding to Q, K, V.

        Args:
            q: Query tensor [B, L, H, D] or [B, H, L, D]
            k: Key tensor
            v: Value tensor
            viewmats: Camera view matrices [B, T, 4, 4]
            Ks: Camera intrinsics [B, T, 3, 3]
            grid_sizes: Tuple of (T, H, W) grid sizes

        Returns:
            Tuple of (q, k, v, apply_fn_o)
        """
        grid_t, grid_h, grid_w = grid_sizes

        B = q.shape[0]
        if q.dim() == 4 and q.shape[2] == self.heads_num:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            need_transpose_back = True
        else:
            need_transpose_back = False

        if viewmats.dim() == 3:
            viewmats = viewmats.unsqueeze(0)
        if Ks is not None and Ks.dim() == 3:
            Ks = Ks.unsqueeze(0)

        q_prope, k_prope, v_prope, apply_fn_o = prope_qkv(
            q,
            k,
            v,
            viewmats=viewmats.to(q.dtype),
            Ks=Ks.to(q.dtype) if Ks is not None else None,
            patches_x=grid_w,
            patches_y=grid_h,
        )

        if need_transpose_back:
            q_prope = q_prope.transpose(1, 2)
            k_prope = k_prope.transpose(1, 2)
            v_prope = v_prope.transpose(1, 2)

        return q_prope, k_prope, v_prope, apply_fn_o

    @torch.no_grad()
    def infer_double_block(self, weights, infer_module_out, block_idx=None):
        """
        Infer a double stream block with causal attention and optional ProPE.

        Args:
            weights: Block weights
            infer_module_out: Module output containing img, txt, vec
            block_idx: Block index for KV cache and ProPE projection

        Returns:
            Tuple of (img, txt) tensors
        """
        # Get Q/K/V for both branches
        (img_q, img_k, img_v, img_q_pre_rope, img_k_pre_rope, img_branch_out) = self._infer_img_branch_before_attn(weights, infer_module_out)
        txt_q, txt_k, txt_v, txt_branch_out = self._infer_txt_branch_before_attn(weights, infer_module_out)

        # Apply ProPE if camera parameters are available
        img_attn_prope = None
        if self.use_prope and hasattr(self.scheduler, "viewmats") and self.scheduler.viewmats is not None:
            img_q_prope, img_k_prope, img_v_prope, apply_fn_o = self._apply_prope(img_q_pre_rope, img_k_pre_rope, img_v, self.scheduler.viewmats, self.scheduler.Ks, infer_module_out.grid_sizes)

            # First attention: original Q/K/V with RoPE (causal)
            img_attn, txt_attn = self._infer_causal_attn(weights, img_q, img_k, img_v, txt_q, txt_k, txt_v, block_idx=block_idx, use_kv_cache=True)

            # Second attention: PRoPE transformed Q/K/V (causal)
            img_attn_prope, _ = self._infer_causal_attn(weights, img_q_prope, img_k_prope, img_v_prope, txt_q, txt_k, txt_v, block_idx=None, use_kv_cache=False)

            # Apply ProPE output transform and projection
            if apply_fn_o is not None and block_idx is not None:
                prope_proj_weight = getattr(self.action_weights, f"img_attn_prope_proj_{block_idx}", None)
                if prope_proj_weight is not None:
                    L, C = img_attn_prope.shape
                    head_dim = C // self.heads_num
                    img_attn_prope_4d = img_attn_prope.reshape(1, L, self.heads_num, head_dim).transpose(1, 2)
                    img_attn_prope_transformed = apply_fn_o(img_attn_prope_4d)
                    img_attn_prope = img_attn_prope_transformed.transpose(1, 2).reshape(L, C)
                    img_attn_prope = prope_proj_weight.apply(img_attn_prope)
        else:
            img_attn, txt_attn = self._infer_causal_attn(weights, img_q, img_k, img_v, txt_q, txt_k, txt_v, block_idx=block_idx, use_kv_cache=True)

        img = self._infer_img_branch_after_attn(weights, img_attn, infer_module_out.img, img_branch_out, img_attn_prope)
        txt = self._infer_txt_branch_after_attn(weights, txt_attn, infer_module_out.txt, txt_branch_out)
        return img, txt

    @torch.no_grad()
    def infer_final_layer(self, weights, infer_module_out):
        """Override to handle per-token vec in final layer."""
        x = torch.cat((infer_module_out.img, infer_module_out.txt), 1)
        img = x[:, : infer_module_out.img.shape[1], ...]
        img_seqlen = img.shape[1]

        vec = infer_module_out.vec

        if self._vec_is_per_token and vec.dim() == 3:
            vec_for_final = vec[:, :img_seqlen, :]
            B, L, C = vec_for_final.shape
            vec_flat = vec_for_final.reshape(B * L, C)
            mod_output_flat = weights.final_layer.adaLN_modulation.apply(vec_flat)
            mod_output = mod_output_flat.reshape(B, L, -1)
            shift, scale = mod_output.chunk(2, dim=-1)

            normed = weights.final_layer.norm_final.apply(img.squeeze(0))
            img = normed * (1 + scale.squeeze(0)) + shift.squeeze(0)
        else:
            shift, scale = weights.final_layer.adaLN_modulation.apply(vec).chunk(2, dim=1)
            img = self.modulate_func(weights.final_layer.norm_final.apply(img.squeeze(0)), scale=scale, shift=shift).squeeze(0)

        img = weights.final_layer.linear.apply(img)
        return img.unsqueeze(0)

    @torch.no_grad()
    def infer_without_offload(self, weights, infer_module_out):
        """Override to pass block index for KV cache and ProPE projection."""
        for i in range(self.double_blocks_num):
            infer_module_out.img, infer_module_out.txt = self.infer_double_block(weights.double_blocks[i], infer_module_out, block_idx=i)

    @torch.no_grad()
    def infer_with_blocks_offload(self, weights, infer_module_out):
        """Inference with block-level CPU offload."""
        for block_idx in range(self.double_blocks_num):
            self.block_idx = block_idx
            if block_idx == 0:
                self.offload_manager.init_first_buffer(weights.double_blocks)
            if block_idx < self.double_blocks_num - 1:
                self.offload_manager.prefetch_weights(block_idx + 1, weights.double_blocks)
            with torch_device_module.stream(self.offload_manager.compute_stream):
                infer_module_out.img, infer_module_out.txt = self.infer_double_block(self.offload_manager.cuda_buffers[0], infer_module_out, block_idx=block_idx)
            self.offload_manager.swap_blocks()

    def set_action_weights(self, action_weights):
        """Set action weights for ProPE projection access."""
        self.action_weights = action_weights

    # ========== AR-specific inference methods ==========

    @torch.no_grad()
    def infer_txt(self, weights, infer_module_out, cache_txt=True):
        """
        Text KV caching inference, corresponding to original forward_txt().
        Called once at the beginning of generation to cache text KV.

        Args:
            weights: Transformer weights
            infer_module_out: Module output containing txt, vec
            cache_txt: Whether to cache text KV (default True)

        Returns:
            KV cache reference
        """
        for block_idx in range(self.double_blocks_num):
            block_weights = weights.double_blocks[block_idx]

            # Compute text Q/K/V
            txt_q, txt_k, txt_v, txt_branch_out = self._infer_txt_branch_before_attn(block_weights, infer_module_out)

            # Text self-attention (is_causal=False)
            txt_seqlen = txt_q.shape[1]
            cu_seqlens_qkv = torch.tensor([0, txt_seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)
            txt_attn = block_weights.self_attention.apply(
                q=txt_q,
                k=txt_k,
                v=txt_v,
                cu_seqlens_q=cu_seqlens_qkv,
                cu_seqlens_kv=cu_seqlens_qkv,
                max_seqlen_q=txt_seqlen,
                max_seqlen_kv=txt_seqlen,
            )

            # Cache text K/V in [B, H, L, D] format
            if cache_txt and self._kv_cache is not None:
                # txt_k/v are [B, L, H, D], transpose to [B, H, L, D]
                self._kv_cache.set_txt_cache(block_idx, txt_k.transpose(1, 2), txt_v.transpose(1, 2))

            # Update text representation
            infer_module_out.txt = self._infer_txt_branch_after_attn(block_weights, txt_attn, infer_module_out.txt, txt_branch_out)

        return self._kv_cache

    @torch.no_grad()
    def infer_vision(self, weights, infer_module_out, cache_vision=False):
        """
        Vision inference using cached text KV.
        Corresponds to original forward_vision().

        Implements the dual-stream pattern from HY-WorldPlay:
        - Normal stream: RoPE-transformed Q/K/V
        - ProPE stream: ProPE-transformed Q/K/V
        - Both streams share text KV (repeated with R=2)
        - Vision KV is cached with both streams concatenated along batch dim

        Args:
            weights: Transformer weights
            infer_module_out: Module output containing img, txt, vec, grid_sizes
            cache_vision: Whether to cache vision KV for context frames

        Returns:
            If cache_vision=True: KV cache reference
            If cache_vision=False: Final layer output (noise prediction)
        """
        # Check if ProPE is enabled
        use_prope = self.use_prope and hasattr(self.scheduler, "viewmats") and self.scheduler.viewmats is not None

        for block_idx in range(self.double_blocks_num):
            block_weights = weights.double_blocks[block_idx]

            # Compute image Q/K/V (includes RoPE and pre-RoPE for ProPE)
            (img_q, img_k, img_v, img_q_pre_rope, img_k_pre_rope, img_branch_out) = self._infer_img_branch_before_attn(block_weights, infer_module_out)

            # Get cached text K/V [B, H, L, D]
            txt_k_cached, txt_v_cached = self._kv_cache.get_txt_cache(block_idx)

            # Get cached vision K/V (may contain both normal and prope if use_prope)
            vision_k_cached, vision_v_cached = self._kv_cache.get_vision_cache(block_idx)

            img_attn_prope = None
            apply_fn_o = None

            if use_prope:
                # Apply ProPE transformation to get prope Q/K/V
                img_q_prope, img_k_prope, img_v_prope, apply_fn_o = self._apply_prope(img_q_pre_rope, img_k_pre_rope, img_v, self.scheduler.viewmats, self.scheduler.Ks, infer_module_out.grid_sizes)

                # Concatenate normal and prope along batch dimension (matching HY-WorldPlay)
                # query: [B, L, H, D] -> [2B, L, H, D]
                query = torch.cat([img_q, img_q_prope], dim=0)
                # key/value: [B, L, H, D] -> [2B, L, H, D]
                key_current = torch.cat([img_k, img_k_prope], dim=0)
                value_current = torch.cat([img_v, img_v_prope], dim=0)

                # Transpose to [2B, H, L, D] for KV cache operations
                key_current_t = key_current.transpose(1, 2)
                value_current_t = value_current.transpose(1, 2)

                # Cache current vision K/V if requested (stores both normal and prope)
                if cache_vision:
                    self._kv_cache.set_vision_cache(block_idx, key_current_t, value_current_t)

                # Repeat text KV with R=2 to match doubled batch
                # txt_k_cached: [B, H, L, D] -> [2B, H, L, D]
                txt_k_repeated = txt_k_cached.repeat(2, 1, 1, 1)
                txt_v_repeated = txt_v_cached.repeat(2, 1, 1, 1)

                # Build K/V for attention: [text, cached_vision, current_vision]
                if vision_k_cached is not None and not cache_vision:
                    # During generation: use cached context + current
                    # vision_k_cached is already [2B, H, L, D] from previous caching
                    key_full = torch.cat([txt_k_repeated, vision_k_cached, key_current_t], dim=2)
                    value_full = torch.cat([txt_v_repeated, vision_v_cached, value_current_t], dim=2)
                else:
                    # During context caching or first chunk: just text + current
                    key_full = torch.cat([txt_k_repeated, key_current_t], dim=2)
                    value_full = torch.cat([txt_v_repeated, value_current_t], dim=2)

                # Transpose back to [2B, L, H, D] for attention
                key_full = key_full.transpose(1, 2)
                value_full = value_full.transpose(1, 2)

                # Attention computation with doubled batch
                img_seqlen = query.shape[1]
                kv_seqlen = key_full.shape[1]

                # cu_seqlens for 2 sequences (normal and prope)
                cu_seqlens_q = torch.tensor([0, img_seqlen, 2 * img_seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)
                cu_seqlens_kv = torch.tensor([0, kv_seqlen, 2 * kv_seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)

                # Single attention call for both streams
                attn_out = block_weights.self_attention.apply(
                    q=query,
                    k=key_full,
                    v=value_full,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=img_seqlen,
                    max_seqlen_kv=kv_seqlen,
                )

                # Split output back to normal and prope
                # attn_out shape: [2*L, C] (flattened from [2B, L, H, D])
                total_len = attn_out.shape[0]
                img_attn = attn_out[: total_len // 2]
                img_attn_prope = attn_out[total_len // 2 :]

                # Apply ProPE output transform and projection
                if apply_fn_o is not None:
                    prope_proj_weight = getattr(self.action_weights, f"img_attn_prope_proj_{block_idx}", None)
                    if prope_proj_weight is not None:
                        L, C = img_attn_prope.shape
                        head_dim = C // self.heads_num
                        img_attn_prope_4d = img_attn_prope.reshape(1, L, self.heads_num, head_dim).transpose(1, 2)
                        img_attn_prope_transformed = apply_fn_o(img_attn_prope_4d)
                        img_attn_prope = img_attn_prope_transformed.transpose(1, 2).reshape(L, C)
                        img_attn_prope = prope_proj_weight.apply(img_attn_prope)

            else:
                # No ProPE - simple single-stream attention
                # Convert img K/V to [B, H, L, D] format
                img_k_t = img_k.transpose(1, 2)
                img_v_t = img_v.transpose(1, 2)

                # Cache current vision K/V if requested
                if cache_vision:
                    self._kv_cache.set_vision_cache(block_idx, img_k_t, img_v_t)

                # Build K/V for attention: [text, cached_vision, current_vision]
                if vision_k_cached is not None and not cache_vision:
                    key = torch.cat([txt_k_cached, vision_k_cached, img_k_t], dim=2)
                    value = torch.cat([txt_v_cached, vision_v_cached, img_v_t], dim=2)
                else:
                    key = torch.cat([txt_k_cached, img_k_t], dim=2)
                    value = torch.cat([txt_v_cached, img_v_t], dim=2)

                # Transpose back to [B, L, H, D] for attention
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                # Attention computation
                img_seqlen = img_q.shape[1]
                kv_seqlen = key.shape[1]
                cu_seqlens_q = torch.tensor([0, img_seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)
                cu_seqlens_kv = torch.tensor([0, kv_seqlen], dtype=torch.int32, device="cpu").to(AI_DEVICE, non_blocking=True)

                img_attn = block_weights.self_attention.apply(
                    q=img_q,
                    k=key,
                    v=value,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    max_seqlen_q=img_seqlen,
                    max_seqlen_kv=kv_seqlen,
                )

            # Update image representation
            infer_module_out.img = self._infer_img_branch_after_attn(block_weights, img_attn, infer_module_out.img, img_branch_out, img_attn_prope)

        if cache_vision:
            return self._kv_cache
        else:
            return self.infer_final_layer(weights, infer_module_out)
