"""
Transformer-block inference for Lyra2WanDiT.

Implements Lyra2AttentionBlock.forward (wan2pt1_lyra2.py L150-212) and the
transformer loop from Lyra2WanModel.forward (L997-1001) using LightX2V weight
modules for all linear operations.

Original Lyra2AttentionBlock.forward summary:
  e = (self.modulation + e).chunk(6, dim=1)     # [B, 1, dim] x6

  # Self-attention (with optional camera / buffer injection)
  cam_emb = cam_encoder(camera) if camera else 0
  buf_emb = buffer_encoder(sincos_embed(buffer)) if buffer else 0
  y = norm1(x) * (1 + e[1]) + e[0]              # shift/scale, float32
  if inject_kq_only:
      kq_bias = cam_emb + buf_emb
      y = self_attn(y, ..., kq_bias=kq_bias)
  else:
      y = self_attn(y + cam_emb + buf_emb, ...)
  x = x + y * e[2]                              # gate

  # Cross-attention
  x = x + cross_attn(norm3(x), context, context_lens)   # e[3] not used here

  # FFN
  y = ffn(norm2(x) * (1 + e[4]) + e[3])
  x = x + y * e[5]

RoPE application (from WanSelfAttention.forward L325-357, rope_apply L203-232):
  Uses flash_attn.layers.rotary.apply_rotary_emb for in-place rotary embedding.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn.functional as F
from einops import rearrange

from lightx2v.models.networks.lyra2.infer.module_io import Lyra2PreInferOutput

try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
    _HAS_FLASH_ROTARY = True
except ImportError:
    _HAS_FLASH_ROTARY = False


# ---------------------------------------------------------------------------
# RoPE helper  (wan2pt1.py L203-232)
# ---------------------------------------------------------------------------

def rope_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding to x.

    Original: rope_apply (wan2pt1.py L203-232).
      Uses flash_apply_rotary_emb for efficiency.

    Args:
        x      [B, L, n_heads, head_dim]
        freqs  [L, 1, 1, head_dim // 2]   (output of generate_rope_freqs)
    Returns:
        [B, L, n_heads, head_dim]  same dtype as input
    """
    seq_len = x.shape[1]
    freqs = freqs.view(seq_len, freqs.shape[-1])  # [L, head_dim // 2] (complex half)
    cos = torch.cos(freqs).to(torch.float32)
    sin = torch.sin(freqs).to(torch.float32)

    if _HAS_FLASH_ROTARY:
        rotated = flash_apply_rotary_emb(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)
    else:
        # Fallback: manual rotation
        x_f = x.to(torch.float32)
        x1, x2 = x_f[..., :freqs.shape[-1]], x_f[..., freqs.shape[-1]:]
        cos_e = cos.unsqueeze(0).unsqueeze(2)   # [1, L, 1, d/2]
        sin_e = sin.unsqueeze(0).unsqueeze(2)
        # Interleaved (even/odd): rebuild by rotating pairs
        x1r = x1 * cos_e - x2 * sin_e
        x2r = x1 * sin_e + x2 * cos_e
        rotated = torch.cat([x1r, x2r], dim=-1)

    return rotated.to(x.dtype)


# ---------------------------------------------------------------------------
# Block-level sincos helper (Lyra2AttentionBlock._sincos_embed L135-147)
# ---------------------------------------------------------------------------

def _sincos_embed(x: torch.Tensor, multires: int) -> torch.Tensor:
    """
    Original: Lyra2AttentionBlock._sincos_embed (wan2pt1_lyra2.py L135-147)
      Positional sinusoidal embedding applied to buffer features.
    """
    if multires <= 0:
        return x
    x_float = x.float()
    embeds = []
    for i in range(multires):
        freq = (2.0 ** i) * math.pi
        embeds.append(torch.sin(x_float * freq))
        embeds.append(torch.cos(x_float * freq))
    return torch.cat(embeds, dim=-1).type_as(x)


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------

def _scaled_dot_product_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Flash-attention-2 style (no mask).  Inputs [B, L, n, d], output [B, L, n*d].

    Uses torch.nn.functional.scaled_dot_product_attention (fused kernel when
    available, otherwise falls back to eager math attention).

    Original: SelfAttnOp / DotProductAttention in wan2pt1.py – calls
    TransformerEngine's DotProductAttention via attn_op.
    """
    B, L, n, d = q.shape
    # SDPA expects [B, n, L, d]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
    # Back to [B, L, n*d]
    return out.permute(0, 2, 1, 3).reshape(B, L, n * d)


# ---------------------------------------------------------------------------
# Single-block inference
# ---------------------------------------------------------------------------

def infer_block(
    x: torch.Tensor,
    block_weights,          # Lyra2BlockWeights
    e0_B_6_D: torch.Tensor,
    freqs_tokens: torch.Tensor,
    context: torch.Tensor,
    num_heads: int,
    head_dim: int,
    buffer_sincos_multires: int,
    inject_kq_only: bool,
    cross_attn_type: str = "i2v_cross_attn",
    camera: Optional[torch.Tensor] = None,
    buffer: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    One Lyra2AttentionBlock forward pass.

    Original: Lyra2AttentionBlock.forward (wan2pt1_lyra2.py L150-212)
    """
    B, L, D = x.shape
    n, d = num_heads, head_dim

    # --- Modulation (float32) ---
    # Original: e = (self.modulation + e).chunk(6, dim=1)   [B, 1, dim] x6
    with amp.autocast("cuda", dtype=torch.float32):
        modulation = block_weights.modulation.tensor   # [1, 6, dim]
        e = (modulation + e0_B_6_D).chunk(6, dim=1)   # list of [B, 1, dim]

    # --- Camera embedding ---
    # Original: cam_emb = cam_encoder(camera) if camera else 0
    # camera shape: [B, L, 1536]  (cam_dim = 1536 when use_plucker_condition)
    if camera is not None:
        cam_2d = camera.reshape(B * L, -1)   # [B*L, 1536]
        cam_emb = block_weights.cam_encoder.apply(cam_2d).reshape(B, L, D)
    else:
        cam_emb = None

    # --- Buffer embedding ---
    # Original: buf_emb = buffer_encoder(sincos_embed(buffer)) if buffer else 0
    if buffer is not None:
        if inject_kq_only:
            validity = buffer[..., -1:]         # [B, L, 1]
            buf_feat = buffer[..., :-1]         # [B, L, buf_dim]
        else:
            buf_feat = buffer
            validity = None
        if buffer_sincos_multires > 0:
            buf_feat = _sincos_embed(buf_feat, buffer_sincos_multires)
        # buffer_encoder: Sequential(Linear, Linear)
        buf_2d = buf_feat.reshape(B * L, -1)
        buf_emb = block_weights.buffer_enc_0.apply(buf_2d)
        if hasattr(block_weights, "buffer_enc_1"):
            buf_emb = F.gelu(buf_emb)  # activation between squeeze layers
            buf_emb = block_weights.buffer_enc_1.apply(buf_emb)
        buf_emb = buf_emb.reshape(B, L, D)
        if inject_kq_only and validity is not None:
            buf_emb = buf_emb * validity
    else:
        buf_emb = None

    # --- norm1 + modulation (float32) ---
    # Original: y = (norm1(x).float() * (1 + e[1]) + e[0]).type_as(x)
    with amp.autocast("cuda", dtype=torch.float32):
        x_norm1 = F.layer_norm(x.float(), (D,), weight=None, bias=None, eps=eps)
        y = (x_norm1 * (1.0 + e[1]) + e[0]).to(x.dtype)

    # --- Self-attention ---
    # Original WanSelfAttention.forward (L325-357):
    #   x_kq = y if kq_bias is None else y + kq_bias
    #   q = norm_q(q_proj(x_kq)).view(B, L, n, d)
    #   k = norm_k(k_proj(x_kq)).view(B, L, n, d)
    #   v = v_proj(y).view(B, L, n, d)
    #   attn_out = attn_op(rope_apply(q), rope_apply(k), v)
    #   out = o_proj(attn_out.flatten(2))
    if inject_kq_only:
        kq_add = 0
        if cam_emb is not None:
            kq_add = kq_add + cam_emb
        if buf_emb is not None:
            kq_add = kq_add + buf_emb
        y_kq = y + kq_add if not isinstance(kq_add, int) else y
    else:
        y_kq = y
        if cam_emb is not None:
            y_kq = y_kq + cam_emb
        if buf_emb is not None:
            y_kq = y_kq + buf_emb

    y_kq_2d = y_kq.reshape(B * L, D)
    y_2d = y.reshape(B * L, D)
    q = block_weights.self_attn.norm_q.apply(block_weights.self_attn.q.apply(y_kq_2d)).reshape(B, L, n, d)
    k = block_weights.self_attn.norm_k.apply(block_weights.self_attn.k.apply(y_kq_2d)).reshape(B, L, n, d)
    v = block_weights.self_attn.v.apply(y_2d).reshape(B, L, n, d)

    q = rope_apply(q, freqs_tokens)
    k = rope_apply(k, freqs_tokens)

    attn_out = _scaled_dot_product_attn(q, k, v)   # [B, L, n*d]
    attn_out = block_weights.self_attn.o.apply(attn_out.reshape(B * L, n * d)).reshape(B, L, D)

    # --- Residual + gate ---
    # Original: x = x + y * e[2].type_as(x)  (in float32)
    with amp.autocast("cuda", dtype=torch.float32):
        x = x + attn_out * e[2].to(x.dtype)

    # --- Cross-attention ---
    # Original cross_attn_ffn block (L205-208):
    #   x = x + cross_attn(norm3(x), context, context_lens)
    # norm3 is Identity for ZoomGS (cross_attn_norm=False) → just pass x.
    x_for_cross = x    # norm3 = Identity in our case
    B_c, L_c, _ = context.shape
    cx_2d = x_for_cross.reshape(B * L, D)
    ctx_2d = context.reshape(B_c * L_c, -1)

    # WanI2VCrossAttention splits context into image prefix + text suffix
    # (wan2pt1.py L410-445):
    #   image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER (=512)
    T5_LEN = 512
    if cross_attn_type == "i2v_cross_attn":
        img_ctx_len = L_c - T5_LEN
        if img_ctx_len > 0:
            context_img = context[:, :img_ctx_len]
            context_txt = context[:, img_ctx_len:]
        else:
            context_img = None
            context_txt = context
    else:
        context_img = None
        context_txt = context

    # Text cross-attention Q/K/V
    q_c = block_weights.cross_attn.norm_q.apply(block_weights.cross_attn.q.apply(cx_2d)).reshape(B, L, n, d)
    L_txt = context_txt.shape[1]
    k_c = block_weights.cross_attn.norm_k.apply(
        block_weights.cross_attn.k.apply(context_txt.reshape(B * L_txt, -1))
    ).reshape(B, L_txt, n, d)
    v_c = block_weights.cross_attn.v.apply(
        context_txt.reshape(B * L_txt, -1)
    ).reshape(B, L_txt, n, d)
    cross_out = _scaled_dot_product_attn(q_c, k_c, v_c)  # [B, L, n*d]

    # Image cross-attention (I2V only)
    if context_img is not None:
        L_img = context_img.shape[1]
        k_img = block_weights.cross_attn.norm_k_img.apply(
            block_weights.cross_attn.k_img.apply(context_img.reshape(B * L_img, -1))
        ).reshape(B, L_img, n, d)
        v_img = block_weights.cross_attn.v_img.apply(
            context_img.reshape(B * L_img, -1)
        ).reshape(B, L_img, n, d)
        img_out = _scaled_dot_product_attn(q_c, k_img, v_img)
        cross_out = cross_out + img_out   # Original: x = x + img_x; then x = o(x)

    cross_attn_out = block_weights.cross_attn.o.apply(
        cross_out.reshape(B * L, n * d)
    ).reshape(B, L, D)
    x = x + cross_attn_out

    # --- FFN ---
    # Original (L207-208):
    #   y = ffn((norm2(x).float() * (1 + e[4]) + e[3]).type_as(x))
    #   x = x + y * e[5]
    with amp.autocast("cuda", dtype=torch.float32):
        x_norm2 = F.layer_norm(x.float(), (D,), weight=None, bias=None, eps=eps)
        x_ffn_in = (x_norm2 * (1.0 + e[4]) + e[3]).to(x.dtype)

    x_ffn_2d = x_ffn_in.reshape(B * L, D)
    ffn_out = F.gelu(block_weights.ffn_0.apply(x_ffn_2d), approximate="tanh")
    ffn_out = block_weights.ffn_2.apply(ffn_out).reshape(B, L, D)

    with amp.autocast("cuda", dtype=torch.float32):
        x = x + ffn_out * e[5].to(x.dtype)

    return x


# ---------------------------------------------------------------------------
# Transformer loop
# ---------------------------------------------------------------------------

class Lyra2TransformerInfer:
    """
    Runs all Lyra2AttentionBlock forward passes.

    Original loop (Lyra2WanModel.forward L997-1001):
      for block in self.blocks:
          x_B_L_D = block(x_B_L_D, **kwargs_blocks)
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_heads = config["num_heads"]
        self.head_dim = config["dim"] // config["num_heads"]
        self.buffer_sincos_multires = config.get("buffer_sincos_multires", 2)
        self.inject_kq_only = config.get("inject_kq_only", True)
        self.cross_attn_type = (
            "t2v_cross_attn" if config.get("model_type", "i2v") == "t2v" else "i2v_cross_attn"
        )
        self.eps = config.get("eps", 1e-6)

    def infer(
        self,
        block_weights,               # Lyra2TransformerWeights
        pre_infer_out: "Lyra2PreInferOutput",
    ) -> torch.Tensor:
        """
        Run all blocks and return final hidden states [B, L, dim].

        Source: Lyra2WanModel.forward L997-1001 (blocks loop).
        """
        x = pre_infer_out.x_tokens          # [B, L, dim]
        freqs = pre_infer_out.freqs_tokens   # [L, 1, 1, head_dim]
        context = pre_infer_out.context
        e0 = pre_infer_out.e0_B_6_D
        camera = pre_infer_out.camera_tokens
        buffer = pre_infer_out.buffer_tokens

        for blk_w in block_weights.blocks:
            x = infer_block(
                x=x,
                block_weights=blk_w,
                e0_B_6_D=e0,
                freqs_tokens=freqs,
                context=context,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                buffer_sincos_multires=self.buffer_sincos_multires,
                inject_kq_only=self.inject_kq_only,
                cross_attn_type=self.cross_attn_type,
                camera=camera,
                buffer=buffer,
                eps=self.eps,
            )
        return x
