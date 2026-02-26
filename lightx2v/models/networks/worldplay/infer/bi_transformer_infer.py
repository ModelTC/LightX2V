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
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


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
        # Per-token: scale/shift are [B, L, C], squeeze to [L, C]
        scale = scale.squeeze(0)
        shift = shift.squeeze(0)
        return x * (1 + scale) + shift
    else:
        # Global: scale/shift are [B, C], broadcast to [L, C]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WorldPlayBITransformerInfer(HunyuanVideo15TransformerInfer):
    """
    Transformer inference for WorldPlay BI (Bidirectional) model.

    Key differences from AR transformer:
    - Uses bidirectional attention (not causal)
    - No KV cache management
    - Supports context frame concatenation for chunk-based generation

    Key differences from Distill transformer:
    - Standard 50-step inference
    - Supports classifier-free guidance

    Extends HunyuanVideo15TransformerInfer with:
    - ProPE (Projective Positional Encoding) for camera pose conditioning
    - Per-token vec modulation for per-frame action conditioning
    - Context frame handling for chunk-based generation
    - CPU offload support
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_prope = config.get("use_prope", True)

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

    @torch.no_grad()
    def _infer_img_branch_before_attn(self, weights, infer_module_out):
        """
        Override to handle per-token vec modulation for WorldPlay BI.

        When vec is [B, T*H*W, C] (per-frame action conditioning), we need to
        extract the img portion and apply per-token modulation.

        Returns:
            Tuple of (img_q, img_k, img_v, img_q_pre_rope, img_k_pre_rope, img_branch_out)
            - img_q, img_k: Q/K with RoPE applied (for standard attention branch)
            - img_q_pre_rope, img_k_pre_rope: Q/K without RoPE (for PRoPE branch)
            - img_v: V tensor (shared by both branches)
            - img_branch_out: modulation parameters
        """
        vec = infer_module_out.vec
        img_seqlen = infer_module_out.img.shape[1]

        if self._vec_is_per_token and vec.dim() == 3:
            # Per-token vec: reshape for linear layer [B, L, C] -> [B*L, C]
            B, L, C = vec.shape
            vec_flat = vec.reshape(B * L, C)
            mod_output_flat = weights.img_branch.img_mod.apply(vec_flat)
            mod_output = mod_output_flat.reshape(B, L, -1)

            # Extract img portion
            img_mod = mod_output[:, :img_seqlen, :]  # [B, img_seqlen, 6*C]
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = img_mod.chunk(6, dim=-1)

            # Apply per-token modulation
            img_modulated = weights.img_branch.img_norm1.apply(infer_module_out.img.squeeze(0))
            # img_modulated: [L, C], img_mod1_scale/shift: [B, L, C]
            img_modulated = img_modulated * (1 + img_mod1_scale.squeeze(0)) + img_mod1_shift.squeeze(0)
        else:
            # Global vec: standard modulation
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

        # Save pre-RoPE Q/K for PRoPE branch (PRoPE should be applied without RoPE)
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
        """
        Override to handle per-token vec modulation for WorldPlay BI.

        For text branch, we use global vec (first token or mean) since text
        doesn't have per-frame structure.
        """
        vec = infer_module_out.vec

        if self._vec_is_per_token and vec.dim() == 3:
            # Use mean of vec for text modulation (text is not per-frame)
            vec_for_txt = vec.mean(dim=1)  # [B, C]
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
    def _infer_img_branch_after_attn(self, weights, img_attn, img, img_branch_out, img_attn_prope=None):
        """Override to handle per-token modulation in post-attention.

        Args:
            weights: Block weights
            img_attn: Original attention output [L, C]
            img: Input image tensor [B, L, C]
            img_branch_out: Branch output containing modulation parameters
            img_attn_prope: PRoPE attention output [L, C], already projected through prope_proj
        """
        if self._vec_is_per_token and img_branch_out.img_mod2_scale.dim() == 3:
            # Per-token modulation
            img_seqlen = img.shape[1]
            img_mod2_scale = img_branch_out.img_mod2_scale[:, :img_seqlen, :]  # [B, L, C]
            img_mod2_shift = img_branch_out.img_mod2_shift[:, :img_seqlen, :]  # [B, L, C]
            img_mod1_gate = img_branch_out.img_mod1_gate[:, :img_seqlen, :]  # [B, L, C]
            img_mod2_gate = img_branch_out.img_mod2_gate[:, :img_seqlen, :]  # [B, L, C]

            # img_attn is [L, C], project through img_attn_proj
            attn_proj = weights.img_branch.img_attn_proj.apply(img_attn)  # [L, C]
            # Add PRoPE attention output if available (already projected through prope_proj)
            if img_attn_prope is not None:
                attn_proj = attn_proj + img_attn_prope  # [L, C]
            # Apply per-token gate: [L, C] * [B, L, C] -> squeeze gate to [L, C]
            gated_attn = attn_proj * img_mod1_gate.squeeze(0)  # [L, C]
            img = img + gated_attn.unsqueeze(0)  # [B, L, C]

            # img is now [B, L, C], squeeze to [L, C] for norm
            img_squeezed = img.squeeze(0)  # [L, C]
            normed = weights.img_branch.img_norm2.apply(img_squeezed)
            # Per-token modulation: normed [L, C], scale/shift [B, L, C] -> squeeze to [L, C]
            modulated = normed * (1 + img_mod2_scale.squeeze(0)) + img_mod2_shift.squeeze(0)
            out = weights.img_branch.img_mlp_fc1.apply(modulated)
            out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
            # Apply per-token gate
            gated_out = out * img_mod2_gate.squeeze(0)  # [L, C]
            img = img + gated_out.unsqueeze(0)  # [B, L, C]
        else:
            # Standard modulation
            # Project original attention through img_attn_proj
            attn_proj = weights.img_branch.img_attn_proj.apply(img_attn).unsqueeze(0)  # [1, L, C]
            # Add PRoPE attention output if available (already projected through prope_proj)
            if img_attn_prope is not None:
                attn_proj = attn_proj + img_attn_prope.unsqueeze(0)  # [1, L, C]
            img = img + apply_gate(attn_proj, gate=img_branch_out.img_mod1_gate)
            out = weights.img_branch.img_mlp_fc1.apply(
                self.modulate_func(weights.img_branch.img_norm2.apply(img.squeeze(0)), scale=img_branch_out.img_mod2_scale, shift=img_branch_out.img_mod2_shift).squeeze(0)
            )
            out = weights.img_branch.img_mlp_fc2.apply(F.gelu(out, approximate="tanh"))
            img = img + apply_gate(out.unsqueeze(0), gate=img_branch_out.img_mod2_gate)
        return img

    @torch.no_grad()
    def infer_final_layer(self, weights, infer_module_out):
        """Override to handle per-token vec in final layer."""
        x = torch.cat((infer_module_out.img, infer_module_out.txt), 1)
        img = x[:, : infer_module_out.img.shape[1], ...]
        img_seqlen = img.shape[1]

        vec = infer_module_out.vec

        if self._vec_is_per_token and vec.dim() == 3:
            # Per-token vec: extract img portion for final layer modulation
            vec_for_final = vec[:, :img_seqlen, :]  # [B, img_seqlen, C]
            B, L, C = vec_for_final.shape
            vec_flat = vec_for_final.reshape(B * L, C)
            mod_output_flat = weights.final_layer.adaLN_modulation.apply(vec_flat)
            mod_output = mod_output_flat.reshape(B, L, -1)
            shift, scale = mod_output.chunk(2, dim=-1)  # [B, img_seqlen, C] each

            # Apply per-token modulation
            normed = weights.final_layer.norm_final.apply(img.squeeze(0))
            img = normed * (1 + scale.squeeze(0)) + shift.squeeze(0)
        else:
            # Global vec: standard modulation
            shift, scale = weights.final_layer.adaLN_modulation.apply(vec).chunk(2, dim=1)
            img = self.modulate_func(weights.final_layer.norm_final.apply(img.squeeze(0)), scale=scale, shift=shift).squeeze(0)

        img = weights.final_layer.linear.apply(img)
        return img.unsqueeze(0)

    @torch.no_grad()
    def infer_double_block(self, weights, infer_module_out, block_idx=None):
        """
        Infer a double stream block with optional ProPE.

        For BI model, uses bidirectional attention (not causal).

        Args:
            weights: Block weights
            infer_module_out: Module output containing img, txt, vec
            block_idx: Block index for accessing ProPE projection weights

        Returns:
            Tuple of (img, txt) tensors
        """
        img_q, img_k, img_v, img_q_pre_rope, img_k_pre_rope, img_branch_out = self._infer_img_branch_before_attn(weights, infer_module_out)
        txt_q, txt_k, txt_v, txt_branch_out = self._infer_txt_branch_before_attn(weights, infer_module_out)

        # Apply ProPE if camera parameters are available
        if self.use_prope and hasattr(self.scheduler, "viewmats") and self.scheduler.viewmats is not None:
            # Apply PRoPE transform to pre-RoPE Q/K/V (PRoPE should NOT include RoPE)
            img_q_prope, img_k_prope, img_v_prope, apply_fn_o = self._apply_prope(img_q_pre_rope, img_k_pre_rope, img_v, self.scheduler.viewmats, self.scheduler.Ks, infer_module_out.grid_sizes)

            # First attention: original Q/K/V with RoPE (standard attention)
            # BI model uses bidirectional attention
            img_attn, txt_attn = self._infer_attn(weights, img_q, img_k, img_v, txt_q, txt_k, txt_v)

            # Second attention: PRoPE transformed Q/K/V (PRoPE only, no RoPE)
            img_attn_prope, _ = self._infer_attn(weights, img_q_prope, img_k_prope, img_v_prope, txt_q, txt_k, txt_v)

            # Apply ProPE output transform and projection
            if apply_fn_o is not None and block_idx is not None:
                # Get the prope projection weight for this block
                prope_proj_weight = getattr(self.action_weights, f"img_attn_prope_proj_{block_idx}", None)
                if prope_proj_weight is not None:
                    # img_attn_prope shape: [L, C] where L = img_seqlen, C = hidden_size
                    # Need to reshape to [B, H, L, D] for ProPE output transform
                    L, C = img_attn_prope.shape
                    head_dim = C // self.heads_num
                    # Reshape: [L, C] -> [1, L, H, D] -> [1, H, L, D]
                    img_attn_prope_4d = img_attn_prope.reshape(1, L, self.heads_num, head_dim).transpose(1, 2)
                    # Apply ProPE output transform
                    img_attn_prope_transformed = apply_fn_o(img_attn_prope_4d)
                    # Reshape back: [1, H, L, D] -> [1, L, H, D] -> [L, C]
                    img_attn_prope = img_attn_prope_transformed.transpose(1, 2).reshape(L, C)
                    # Project PRoPE attention output
                    img_attn_prope = prope_proj_weight.apply(img_attn_prope)
        else:
            img_attn, txt_attn = self._infer_attn(weights, img_q, img_k, img_v, txt_q, txt_k, txt_v)
            img_attn_prope = None

        img = self._infer_img_branch_after_attn(weights, img_attn, infer_module_out.img, img_branch_out, img_attn_prope)
        txt = self._infer_txt_branch_after_attn(weights, txt_attn, infer_module_out.txt, txt_branch_out)
        return img, txt

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

        # Reshape for ProPE: need [B, H, L, D] format
        B = q.shape[0]
        if q.dim() == 4 and q.shape[2] == self.heads_num:
            # Shape is [B, L, H, D], transpose to [B, H, L, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            need_transpose_back = True
        else:
            need_transpose_back = False

        # Ensure viewmats and Ks have correct shape
        if viewmats.dim() == 3:
            viewmats = viewmats.unsqueeze(0)
        if Ks is not None and Ks.dim() == 3:
            Ks = Ks.unsqueeze(0)

        # Apply ProPE
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
    def infer_without_offload(self, weights, infer_module_out):
        """Override to pass block index for ProPE projection."""
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
