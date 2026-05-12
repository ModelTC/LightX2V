"""
Pre-transformer inference for Lyra2WanDiT.

Implements the Lyra2 framepack patchify path plus time/text embedding.
Source reference: lyra_2/_src/networks/wan2pt1_lyra2.py

Covered logic (L900-971 of Lyra2WanModel.forward, plus helpers _patchify_lyra2
L555-803, _patchify_linear L515-534, _pixelshuffle_tokens L536-553,
_pad_for_linear_patch L499-513, and VideoRopePosition3DEmb L94-190):

  # Original Lyra2WanModel.forward (framepack path):
  #   camera = x[:, -384:]  ; x = x[:, :-384]
  #   camera_5d = rearrange(camera, "b c t (h h2) (w w2) -> b (c h2 w2) t h w", h2=2, w2=2)
  #   x_tokens, freqs_tokens, camera_tokens, buffer_tokens, gen_range, gen_grid = _patchify_lyra2(...)
  #   e_B_D  = time_embedding(sinusoidal_embedding_1d(freq_dim, t_B))
  #   e0_B_6_D = time_projection(e_B_D).unflatten(1, (6, dim))
  #   context  = text_embedding(crossattn_emb)
  #   if frame_cond: context = cat([img_emb(frame_cond), context], dim=1)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lightx2v.models.networks.lyra2.infer.module_io import Lyra2PreInferOutput


# ---------------------------------------------------------------------------
# Pure-math helpers (no learnable params – replicate Lyra2 source)
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Original: lyra_2/_src/networks/wan2pt1.py L191-201
      sinusoid = outer(position, 10000^(-arange/half))
      x = cat([cos(sinusoid), sin(sinusoid)], dim=1)
    """
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)),
    )
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def _generate_rope_freqs(
    T: int,
    H: int,
    W: int,
    head_dim: int,
    h_ntk_factor: float,
    w_ntk_factor: float,
    t_ntk_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Replicate VideoRopePosition3DEmb.generate_embeddings
    (lyra_2/_src/networks/wan2pt1.py L135-189).

    Returns: [T*H*W, 1, 1, head_dim] float32 tensor.
    """
    dim = head_dim
    dim_h = dim // 6 * 2
    dim_w = dim_h
    dim_t = dim - 2 * dim_h

    h_theta = 10000.0 * h_ntk_factor
    w_theta = 10000.0 * w_ntk_factor
    t_theta = 10000.0 * t_ntk_factor

    dim_spatial_range = torch.arange(0, dim_h, 2, device=device)[: (dim_h // 2)].float() / dim_h
    dim_temporal_range = torch.arange(0, dim_t, 2, device=device)[: (dim_t // 2)].float() / dim_t

    seq = torch.arange(max(H, W, T), device=device).float()
    h_spatial_freqs = 1.0 / (h_theta ** dim_spatial_range)
    w_spatial_freqs = 1.0 / (w_theta ** dim_spatial_range)
    temporal_freqs = 1.0 / (t_theta ** dim_temporal_range)

    freqs_h = torch.outer(seq[:H], h_spatial_freqs)  # [H, dim_h/2]
    freqs_w = torch.outer(seq[:W], w_spatial_freqs)  # [W, dim_w/2]
    freqs_t = torch.outer(seq[:T], temporal_freqs)   # [T, dim_t/2]

    freqs_T_H_W_D = torch.cat(
        [
            repeat(freqs_t, "t d -> t h w d", h=H, w=W),
            repeat(freqs_h, "h d -> t h w d", t=T, w=W),
            repeat(freqs_w, "w d -> t h w d", t=T, h=H),
        ],
        dim=-1,
    )  # [T, H, W, head_dim]
    return rearrange(freqs_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


def _pad_for_linear_patch(x: torch.Tensor, kernel: Tuple[int, int, int]) -> torch.Tensor:
    """
    Original: Lyra2WanModel._pad_for_linear_patch (wan2pt1_lyra2.py L499-513)
      Pad so T/H/W divisible by kernel (kt, kh, kw), replicate padding.
    """
    _, _, t, h, w = x.shape
    kt, kh, kw = kernel
    pad_t = (kt - (t % kt)) % kt
    pad_h = (kh - (h % kh)) % kh
    pad_w = (kw - (w % kw)) % kw
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
    return x


def _patchify_linear(
    x: torch.Tensor,
    patch: Tuple[int, int, int],
    weight_module,  # MM_WEIGHT_REGISTER instance
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Original: Lyra2WanModel._patchify_linear (wan2pt1_lyra2.py L515-534)
      x [B, C, T, H, W] -> tokens [B, f*h*w, dim], (f, h, w)
    Uses weight_module.apply() instead of nn.Linear forward.
    """
    pt, ph, pw = patch
    x = _pad_for_linear_patch(x, patch)
    b, c, t, h, w = x.shape
    f, hh, ww = t // pt, h // ph, w // pw
    # Merge patch dims into channel dim, then flatten spatial batch
    x = rearrange(
        x,
        "b c (f pt) (hh ph) (ww pw) -> (b f hh ww) (c pt ph pw)",
        f=f, pt=pt, hh=hh, ph=ph, ww=ww, pw=pw,
    )
    x = weight_module.apply(x)           # [(b*f*h*w), dim]
    x = x.reshape(b, f * hh * ww, -1)   # [B, f*h*w, dim]
    return x, (f, hh, ww)


def _pixelshuffle_tokens(
    x: torch.Tensor,
    patch: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Original: Lyra2WanModel._pixelshuffle_tokens (wan2pt1_lyra2.py L536-553)
      Rearrange spatial patches into channels, no linear projection.
    """
    pt, ph, pw = patch
    x = _pad_for_linear_patch(x, patch)
    b, c, t, h, w = x.shape
    f, hh, ww = t // pt, h // ph, w // pw
    x = rearrange(
        x,
        "b c (f pt) (hh ph) (ww pw) -> b (f hh ww) (c pt ph pw)",
        f=f, pt=pt, hh=hh, ph=ph, ww=ww, pw=pw,
    )
    return x, (f, hh, ww)


# ---------------------------------------------------------------------------
# Pre-infer class
# ---------------------------------------------------------------------------

class Lyra2PreInfer:
    """
    Implements the pre-transformer stage of Lyra2WanDiT.

    Covers: camera extraction, _patchify_lyra2, time embedding, text embedding.
    Source: Lyra2WanModel.forward L806-971.
    """

    def __init__(self, config: dict):
        self.config = config
        self.dim = config["dim"]
        self.freq_dim = config.get("freq_dim", 256)
        self.num_heads = config["num_heads"]
        self.head_dim = self.dim // self.num_heads
        self.patch_size: Tuple[int, int, int] = tuple(config.get("patch_size", [1, 2, 2]))
        self.use_plucker = config.get("use_plucker_condition", True)
        self.use_correspondence = config.get("use_correspondence", True)
        self.buffer_pixelshuffle = config.get("buffer_pixelshuffle", True)
        self.buffer_in_dim = config.get("buffer_in_dim", 0)
        self.buffer_sincos_multires = config.get("buffer_sincos_multires", 2)
        self.inject_kq_only = config.get("inject_kq_only", True)
        self.has_img_emb = config.get("model_type", "i2v") in ("i2v", "flf2v")

        # NTK extrapolation factors for RoPE (defaults matching Lyra2WanModel)
        self._h_ntk = 1.0
        self._w_ntk = 1.0
        self._t_ntk = 1.0

        # Will hold (kernel_sizes, kernel_types) after init_clean_patch_embeddings
        self._clean_kernel_sizes: Optional[List[int]] = None
        self._clean_kernel_types: Optional[List[str]] = None

    def init_clean_kernels(self, kernel_sizes: List[int], kernel_types: List[str]):
        """Called once after the model is loaded to configure framepack kernels."""
        self._clean_kernel_sizes = list(kernel_sizes)
        self._clean_kernel_types = list(kernel_types)

    # ------------------------------------------------------------------
    # _patchify_lyra2 reimplemented using weight modules
    # Original: Lyra2WanModel._patchify_lyra2 (wan2pt1_lyra2.py L555-803)
    # ------------------------------------------------------------------
    def _patchify_lyra2(
        self,
        weights,           # Lyra2PreWeights
        x: torch.Tensor,   # [B, C, T, H, W]  (camera channels already stripped)
        framepack_indices: torch.Tensor,
        framepack_splits: List[int],
        framepack_kernel_ids: List[int],
        framepack_kernel_types: Optional[List[str]],
        camera: Optional[torch.Tensor],          # [B, (c*h2*w2), T, H, W]
        buffer_B_C_T_H_W: Optional[torch.Tensor],
    ):
        """
        Returns:
            x_tokens     [B, L, dim]
            freqs_tokens [L, 1, 1, head_dim]
            camera_tokens [B, L, cam_dim] or None
            buffer_tokens [B, L, buf_dim] or None
            (gen_start, gen_end)
            (f_gen, h_gen, w_gen)
        """
        assert self._clean_kernel_sizes is not None, "Call init_clean_kernels() first"
        pt, ph, pw = self.patch_size

        xs = x[:, :, framepack_indices].split(framepack_splits, dim=2)
        inds = framepack_indices.split(framepack_splits)

        _, _, T_total, H, W = x.shape
        f_base = T_total // pt
        h_base = H // ph
        w_base = W // pw

        # Precompute base RoPE frequencies once
        freqs_base = _generate_rope_freqs(
            f_base, h_base, w_base, self.head_dim,
            self._h_ntk, self._w_ntk, self._t_ntk, x.device,
        )  # [f*h*w, 1, 1, head_dim]
        freqs_base_5d = rearrange(
            freqs_base, "(f h w) 1 1 d -> 1 d f h w", f=f_base, h=h_base, w=w_base
        )  # [1, head_dim, f_base, h_base, w_base]

        token_chunks: List[torch.Tensor] = []
        freq_chunks: List[torch.Tensor] = []
        cam_chunks: List[torch.Tensor] = []
        use_buf = self.use_correspondence and buffer_B_C_T_H_W is not None
        buf_chunks: List[torch.Tensor] = []
        buf_validity_chunks: List[torch.Tensor] = []

        buf_splits = None
        buffer_full_match = False
        if buffer_B_C_T_H_W is not None:
            buffer_full_match = int(buffer_B_C_T_H_W.shape[2]) == int(x.shape[2])
            if buffer_full_match:
                buf_splits = buffer_B_C_T_H_W[:, :, framepack_indices].split(framepack_splits, dim=2)

        gen_start = gen_end = None
        total_tokens = 0
        gen_grid = (0, 0, 0)

        # Determine buffer token channel dim
        if use_buf:
            if self.buffer_pixelshuffle:
                buffer_token_dim = self.buffer_in_dim * pt * ph * pw
            else:
                buffer_token_dim = self.dim
        else:
            buffer_token_dim = None

        for i, x_chunk in enumerate(xs):
            kid = framepack_kernel_ids[i]
            ktype = framepack_kernel_types[i] if framepack_kernel_types and i < len(framepack_kernel_types) else None

            if kid == -1:
                # ---- Generated segment – base patch embedding ----
                x_tokens, (f, h, w) = _patchify_linear(x_chunk, self.patch_size, weights.patch_embedding)
                token_chunks.append(x_tokens)
                if gen_start is None:
                    gen_start = total_tokens
                total_tokens += x_tokens.shape[1]
                gen_end = total_tokens
                gen_grid = (f, h, w)

                # Buffer tokens for generated segment
                if use_buf:
                    buf = buf_splits[i] if buffer_full_match else buffer_B_C_T_H_W
                    buf = buf.to(dtype=x_chunk.dtype, device=x_chunk.device)
                    if not self.buffer_pixelshuffle:
                        raise ValueError("use_correspondence requires buffer_pixelshuffle=True")
                    buf_tokens, _ = _pixelshuffle_tokens(buf, self.patch_size)
                    buf_chunks.append(buf_tokens)
                    buf_validity_chunks.append(
                        torch.ones(buf_tokens.shape[0], buf_tokens.shape[1], 1,
                                   device=buf_tokens.device, dtype=buf_tokens.dtype)
                    )

                # RoPE freqs – slice base grid along T
                t_idx = inds[i].to(device=freqs_base_5d.device, dtype=torch.long)
                if f > t_idx.numel():
                    pad_t = f - t_idx.numel()
                    t_idx = torch.cat([t_idx, t_idx[-1:].repeat(pad_t)], dim=0)
                freqs_sel = freqs_base_5d[:, :, t_idx, :, :]
                freq_chunks.append(rearrange(freqs_sel[0], "d f h w -> (f h w) 1 1 d"))

                # Camera tokens for generated segment
                if camera is not None:
                    cam_t_idx = t_idx.to(device=camera.device, dtype=torch.long)
                    cam_sel = camera[:, :, cam_t_idx, :, :]
                    cam_chunks.append(rearrange(cam_sel, "b d f h w -> b (f h w) d").type_as(x_tokens))
            else:
                # ---- History/clean segment – enlarged patch embedding ----
                kernel_factor = int(self._clean_kernel_sizes[kid])
                clean_lin = weights._clean_emb_modules[kid]
                clean_ktype = self._clean_kernel_types[kid] if self._clean_kernel_types else "k"
                effective_ktype = ktype if ktype is not None else clean_ktype

                if effective_ktype == "s":
                    enlarged_patch = (pt, ph * kernel_factor, pw * kernel_factor)
                    pool_k = (1, kernel_factor, kernel_factor)
                else:
                    enlarged_patch = (pt * kernel_factor, ph * kernel_factor, pw * kernel_factor)
                    pool_k = (kernel_factor, kernel_factor, kernel_factor)

                x_tokens, (f, h, w) = _patchify_linear(x_chunk, enlarged_patch, clean_lin)
                token_chunks.append(x_tokens)
                total_tokens += x_tokens.shape[1]

                # Buffer tokens for history segment
                if use_buf:
                    if buffer_full_match:
                        buf = buf_splits[i].to(dtype=x_chunk.dtype, device=x_chunk.device)
                        if effective_ktype == "s":
                            if kernel_factor <= 1:
                                buf_tokens, _ = _pixelshuffle_tokens(buf, self.patch_size)
                            else:
                                pk = (1, pool_k[1], pool_k[2])
                                buf_pooled = F.avg_pool3d(buf.float(), kernel_size=pk, stride=pk)
                                buf_tokens, _ = _pixelshuffle_tokens(buf_pooled.type_as(buf), self.patch_size)
                            buf_is_real = True
                        else:
                            buf_tokens = torch.full(
                                (x_tokens.shape[0], x_tokens.shape[1], int(buffer_token_dim)),
                                -1.0, device=x_tokens.device, dtype=x_tokens.dtype,
                            )
                            buf_is_real = False
                    else:
                        fill_val = -1.0 if effective_ktype != "s" else 0.0
                        buf_tokens = torch.full(
                            (x_tokens.shape[0], x_tokens.shape[1], int(buffer_token_dim)),
                            fill_val, device=x_tokens.device, dtype=x_tokens.dtype,
                        )
                        buf_is_real = False
                    buf_chunks.append(buf_tokens)
                    validity_val = 1.0 if buf_is_real else 0.0
                    buf_validity_chunks.append(torch.full(
                        (buf_tokens.shape[0], buf_tokens.shape[1], 1),
                        validity_val, device=buf_tokens.device, dtype=buf_tokens.dtype,
                    ))

                # RoPE freqs – avg-pool base freqs to match enlarged patch grid
                t_idx = inds[i].to(device=freqs_base_5d.device, dtype=torch.long)
                if pool_k[0] > 1:
                    pad_t = (-t_idx.numel()) % pool_k[0]
                    if pad_t:
                        t_idx = torch.cat([t_idx, t_idx[-1:].repeat(pad_t)], dim=0)
                freqs_sel = freqs_base_5d[:, :, t_idx, :, :]
                pad_h = (-h_base) % pool_k[1]
                pad_w = (-w_base) % pool_k[2]
                if pad_h or pad_w:
                    freqs_sel = F.pad(freqs_sel, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                freqs_pooled = F.avg_pool3d(
                    freqs_sel.float(),
                    kernel_size=(pool_k[0], pool_k[1], pool_k[2]),
                    stride=(pool_k[0], pool_k[1], pool_k[2]),
                )
                freq_chunks.append(rearrange(freqs_pooled[0], "d f h w -> (f h w) 1 1 d"))

                # Camera pooling mirrors RoPE pooling
                if camera is not None:
                    cam_t_idx = t_idx.to(device=camera.device, dtype=torch.long)
                    cam_sel = camera[:, :, cam_t_idx, :, :]
                    if pad_h or pad_w:
                        cam_sel = F.pad(cam_sel, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
                    cam_pooled = F.avg_pool3d(cam_sel.float(), kernel_size=(pool_k[0], pool_k[1], pool_k[2]),
                                              stride=(pool_k[0], pool_k[1], pool_k[2]))
                    cam_chunks.append(rearrange(cam_pooled, "b d f h w -> b (f h w) d").type_as(x_tokens))

        x_tokens = torch.cat(token_chunks, dim=1)
        freqs_tokens = torch.cat(freq_chunks, dim=0)
        camera_tokens = torch.cat(cam_chunks, dim=1) if camera is not None else None
        buffer_tokens = torch.cat(buf_chunks, dim=1) if use_buf else None

        # Append per-token validity flag when inject_kq_only
        # Original: wan2pt1_lyra2.py L797-803
        if self.inject_kq_only and buffer_tokens is not None:
            validity_tokens = torch.cat(buf_validity_chunks, dim=1)
            buffer_tokens = torch.cat([buffer_tokens, validity_tokens], dim=-1)

        return x_tokens, freqs_tokens, camera_tokens, buffer_tokens, (gen_start, gen_end), gen_grid

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def infer(
        self,
        weights,                          # Lyra2PreWeights
        x_B_C_T_H_W: torch.Tensor,
        t_B: torch.Tensor,                # [B]
        crossattn_emb: torch.Tensor,      # [B, L, text_dim] or [B, 1, L, text_dim]
        framepack_indices: torch.Tensor,
        framepack_splits: List[int],
        framepack_kernel_ids: List[int],
        framepack_kernel_types: Optional[List[str]],
        y_buffer_B_C_T_H_W: Optional[torch.Tensor] = None,
        frame_cond_crossattn_emb: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Lyra2PreInferOutput:
        """
        Corresponds to the pre-block section of Lyra2WanModel.forward (framepack path)
        L900-971 in wan2pt1_lyra2.py:
          1. Extract camera Plücker channels from x.
          2. _patchify_lyra2 → x_tokens, freqs, camera_tokens, buffer_tokens.
          3. Time embedding + projection.
          4. Text embedding.
          5. Optional image conditioning (MLPProj).
        """
        # ------ 1. Camera extraction (L901-912) ------
        # Original:
        #   camera_ch = 384
        #   camera = x[:, -384:]
        #   x = x[:, :-384]
        #   camera_5d = rearrange(camera, "b c t (h h2) (w w2) -> b (c h2 w2) t h w", h2=2, w2=2)
        camera_5d = None
        if self.use_plucker:
            camera_ch = 384
            camera = x_B_C_T_H_W[:, -camera_ch:]
            x_B_C_T_H_W = x_B_C_T_H_W[:, :-camera_ch]
            camera_5d = rearrange(camera, "b c t (h h2) (w w2) -> b (c h2 w2) t h w", h2=2, w2=2)

        # ------ Padding mask concat (L914-919) ------
        # Original:
        #   if self.concat_padding_mask and padding_mask is not None:
        #       padding_mask_resized = resize(padding_mask, ...)
        #       x = cat([x, padding_mask.unsqueeze(1).repeat(...)], dim=1)
        # (concat_padding_mask defaults to False for ZoomGS, skipped)

        # ------ 2. Patchify (L929-937) ------
        x_tokens, freqs_tokens, camera_tokens, buffer_tokens, gen_range, gen_grid = self._patchify_lyra2(
            weights,
            x_B_C_T_H_W,
            framepack_indices,
            framepack_splits,
            framepack_kernel_ids,
            framepack_kernel_types,
            camera=camera_5d,
            buffer_B_C_T_H_W=y_buffer_B_C_T_H_W,
        )

        # ------ 3. Time embedding (L962-967) ------
        # Original uses amp.autocast fp32 context; weights are bfloat16 and autocast
        # upcasts them transparently via F.linear.  mm_weight.apply uses torch.addmm
        # directly, which requires matching dtypes.  We cast sin_emb to the token dtype
        # (= weight dtype, typically bfloat16) to avoid RuntimeError.
        model_dtype = x_tokens.dtype  # bfloat16 (matches weight dtype after to_cuda)
        sin_emb = sinusoidal_embedding_1d(self.freq_dim, t_B).to(dtype=model_dtype, device=x_tokens.device)
        # time_embedding: Linear → SiLU → Linear
        e_B_D = weights.time_emb_0.apply(sin_emb)   # [B, dim]
        e_B_D = F.silu(e_B_D)
        e_B_D = weights.time_emb_2.apply(e_B_D)     # [B, dim]
        # time_projection: SiLU → Linear
        e0 = weights.time_proj_1.apply(F.silu(e_B_D))   # [B, dim*6]
        e0_B_6_D = e0.unflatten(1, (6, self.dim))       # [B, 6, dim]

        # ------ 4. Text embedding (L968-971) ------
        # Original:
        #   if crossattn_emb.dim() == 4: crossattn_emb = crossattn_emb.squeeze(1)
        #   context_B_L_D = text_embedding(crossattn_emb)
        if crossattn_emb.dim() == 4:
            crossattn_emb = crossattn_emb.squeeze(1)  # [B, L, text_dim]
        crossattn_emb = crossattn_emb.to(dtype=model_dtype, device=x_tokens.device)

        B, L_ctx, _ = crossattn_emb.shape
        ctx = crossattn_emb.reshape(B * L_ctx, -1)      # [B*L, text_dim]
        # text_embedding: Linear → GELU → Linear
        ctx = weights.text_emb_0.apply(ctx)              # [B*L, dim]
        ctx = F.gelu(ctx, approximate="tanh")
        ctx = weights.text_emb_2.apply(ctx)              # [B*L, dim]
        context_B_L_D = ctx.reshape(B, L_ctx, self.dim)  # [B, L, dim]

        # ------ 5. Image conditioning (MLPProj) (L969-971) ------
        # Original:
        #   if frame_cond_crossattn_emb_B_L_D is not None:
        #       context_clip = img_emb(frame_cond_crossattn_emb_B_L_D)
        #       context_B_L_D = cat([context_clip, context_B_L_D], dim=1)
        if frame_cond_crossattn_emb is not None and self.has_img_emb:
            frame_cond_crossattn_emb = frame_cond_crossattn_emb.to(dtype=model_dtype, device=x_tokens.device)
            B2, L2, _ = frame_cond_crossattn_emb.shape
            fc = frame_cond_crossattn_emb.reshape(B2 * L2, -1)
            # MLPProj.proj: LayerNorm → Linear → GELU → Linear → LayerNorm
            fc = weights.img_emb_ln0.apply(fc)
            fc = F.gelu(weights.img_emb_fc1.apply(fc), approximate="tanh")
            fc = weights.img_emb_fc3.apply(fc)
            fc = weights.img_emb_ln4.apply(fc)
            context_clip = fc.reshape(B2, L2, self.dim)
            context_B_L_D = torch.cat([context_clip, context_B_L_D], dim=1)

        return Lyra2PreInferOutput(
            x_tokens=x_tokens,
            freqs_tokens=freqs_tokens,
            context=context_B_L_D,
            e0_B_6_D=e0_B_6_D,
            e_B_D=e_B_D,
            camera_tokens=camera_tokens,
            buffer_tokens=buffer_tokens,
            gen_range=gen_range,
            gen_grid=gen_grid,
        )
