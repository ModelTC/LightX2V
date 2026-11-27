# Source port; attribution and licensing: see the model directory LICENSE/NOTICE.
from __future__ import annotations

from lightx2v.models.networks.hymotion.weights.hymotion_mmdit import Optional, Tensor, Tuple, apply_gate, attention, get_module_device, modulate, rearrange, torch


def MMDoubleStreamBlock_forward(self, motion_feat: Tensor, text_feat: Tensor, adapter: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    motion_shift_msa, motion_scale_msa, motion_gate_msa, motion_shift_mlp, motion_scale_mlp, motion_gate_mlp = self.motion_mod(adapter).chunk(6, dim=-1)
    text_shift_msa, text_scale_msa, text_gate_msa, text_shift_mlp, text_scale_mlp, text_gate_mlp = self.text_mod(adapter).chunk(6, dim=-1)
    motion_modulated = self.motion_norm1(motion_feat)
    motion_modulated = modulate(motion_modulated, shift=motion_shift_msa, scale=motion_scale_msa)
    motion_qkv = self.motion_qkv(motion_modulated)
    motion_q, motion_k, motion_v = rearrange(motion_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
    motion_q = self.motion_q_norm(motion_q).to(motion_v)
    motion_k = self.motion_k_norm(motion_k).to(motion_v)
    if self.apply_rope_to_single_branch:
        motion_q, motion_k = self.rotary_emb.apply_rotary_emb(motion_q, motion_k)
    text_modulated = self.text_norm1(text_feat)
    text_modulated = modulate(text_modulated, shift=text_shift_msa, scale=text_scale_msa)
    text_qkv = self.text_qkv(text_modulated)
    text_q, text_k, text_v = rearrange(text_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
    text_q = self.text_q_norm(text_q).to(text_v)
    text_k = self.text_k_norm(text_k).to(text_v)
    q = torch.cat((motion_q, text_q), dim=1)
    k = torch.cat((motion_k, text_k), dim=1)
    v = torch.cat((motion_v, text_v), dim=1)
    if not self.apply_rope_to_single_branch:
        q, k = self.rotary_emb.apply_rotary_emb(q, k)
    bsz, _total_len, _, _ = q.shape
    motion_len = motion_feat.shape[1]
    text_len = text_feat.shape[1]
    dropout_p = 0.0 if not self.training else self.dropout
    attn_output = attention(
        q,
        k,
        v,
        mode="torch",
        drop_rate=dropout_p,
        attn_mask=attn_mask,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        batch_size=bsz,
        training=self.training,
    )
    motion_attn_output, text_attn_output = (attn_output[:, :motion_len, ...], attn_output[:, motion_len:, ...])
    motion_feat = motion_feat + apply_gate(self.motion_out_proj(motion_attn_output), gate=motion_gate_msa)
    motion_feat = motion_feat + apply_gate(self.motion_mlp(modulate(self.motion_norm2(motion_feat), shift=motion_shift_mlp, scale=motion_scale_mlp)), gate=motion_gate_mlp)
    text_feat = text_feat + apply_gate(self.text_out_proj(text_attn_output), gate=text_gate_msa)
    text_feat = text_feat + apply_gate(self.text_mlp(modulate(self.text_norm2(text_feat), shift=text_shift_mlp, scale=text_scale_mlp)), gate=text_gate_mlp)
    return (motion_feat, text_feat)


def MMSingleStreamBlock_forward(self, x: Tensor, split_len: int, adapter: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    shift_msa, scale_msa, gate_msa = self.modulation(adapter).chunk(3, dim=-1)
    x_modulated = modulate(self.norm(x), shift_msa, scale_msa)
    qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [3 * self.feat_dim, self.mlp_hidden_dim], dim=-1)
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
    q = self.q_norm(q).to(v)
    k = self.k_norm(k).to(v)
    q1, q2 = (q[:, :split_len, ...], q[:, split_len:, ...])
    k1, k2 = (k[:, :split_len, ...], k[:, split_len:, ...])
    if self.apply_rope_to_single_branch:
        q1, k1 = self.rotary_emb.apply_rotary_emb(q1, k1)
    q = torch.cat((q1, q2), dim=1)
    k = torch.cat((k1, k2), dim=1)
    if not self.apply_rope_to_single_branch:
        q, k = self.rotary_emb.apply_rotary_emb(q, k)
    bsz, _total_len = x_modulated.shape[:2]
    dropout_p = 0.0 if not self.training else self.dropout
    attn_output = attention(
        q,
        k,
        v,
        mode="torch",
        drop_rate=dropout_p,
        attn_mask=attn_mask,
        causal=False,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        max_seqlen_q=None,
        max_seqlen_kv=None,
        batch_size=bsz,
        training=self.training,
    )
    output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp_hidden)), 2))
    return x + apply_gate(output, gate=gate_msa)


def HunyuanMotionMMDiT_forward(self, x: Tensor, ctxt_input: Tensor, vtxt_input: Tensor, timesteps: Tensor, x_mask_temporal: Tensor, ctxt_mask_temporal: Tensor, **kwargs) -> Tensor:
    device = get_module_device(self)
    motion_feat = self.input_encoder(x)
    if self.with_long_skip_connection:
        origin_feat = motion_feat
    if self.insert_start_token:
        start_token = self.start_token.to(motion_feat.dtype)[None].repeat(motion_feat.shape[0], 1, 1)
        motion_feat = torch.cat((start_token, motion_feat), dim=1)
        x_mask_temporal = torch.cat([torch.ones_like(x_mask_temporal[:, :1], dtype=torch.bool), x_mask_temporal], dim=1)
    timestep_feat = self.timestep_encoder(timesteps)
    vtxt_feat = self.vtxt_encoder(vtxt_input.to(motion_feat.dtype))
    adapter = timestep_feat + vtxt_feat
    motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
    ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
    seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)
    if self.mask_mode is None:
        seq_mask = None
    elif self.mask_mode == "causal":
        motion_len = motion_feat.shape[1]
        seq_mask = torch.triu(torch.full((motion_len, motion_len), float("-inf"), device=device), diagonal=1)
    elif self.mask_mode == "narrowband":
        window = round(self.narrowband_length)
        motion_len = motion_feat.shape[1]
        idx = torch.arange(motion_len, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()
        band = dist <= window
        seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
        seq_mask = seq_mask.masked_fill(band, 0.0)
    else:
        raise ValueError(f"Unsupported mask mode: {self.mask_mode}")
    ctxt_feat = self.ctxt_encoder(ctxt_input.to(motion_feat.dtype))
    if hasattr(self, "text_refiner"):
        ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))
    bsz = x.shape[0]
    motion_len = motion_feat.shape[1]
    text_len = ctxt_feat.shape[1]
    total_len = motion_len + text_len
    mask_dtype = motion_feat.dtype
    attn_mask_double = self._build_dmm_attn_mask_shared(bsz=bsz, motion_len=motion_len, text_len=text_len, dtype=mask_dtype, key_padding_mask=seq_key_padding_mask, attn_mask=seq_mask, device=device)
    for i_layer, mod in enumerate(self.double_blocks):
        motion_feat, ctxt_feat = mod(motion_feat=motion_feat, text_feat=ctxt_feat, adapter=adapter, attn_mask=attn_mask_double)
    split_len = motion_feat.shape[1]
    x = torch.cat((motion_feat, ctxt_feat), 1)
    attn_mask_single = self._build_smm_attn_mask_shared(bsz=bsz, split_len=split_len, total_len=total_len, dtype=mask_dtype, key_padding_mask=seq_key_padding_mask, attn_mask=seq_mask, device=device)
    for i_layer, mod in enumerate(self.single_blocks):
        x = mod(x=x, split_len=split_len, adapter=adapter, attn_mask=attn_mask_single)
    x = x[:, :split_len, ...]
    if self.insert_start_token:
        x = x[:, 1:, ...]
    if self.with_long_skip_connection:
        x = self.long_skip_net(origin_feat, timestep_feat) + x
    predicted_res = self.final_layer(x, adapter)
    return predicted_res


def HunyuanMotionMMDiT__canonical_mask(input_mask: Tensor) -> Tensor:
    if input_mask.ndim == 1:
        input_mask = input_mask.unsqueeze(1)
    key_padding_mask = torch.where(input_mask, torch.zeros_like(input_mask, dtype=torch.float), torch.full_like(input_mask, float("-inf"), dtype=torch.float))
    return key_padding_mask


def HunyuanMotionMMDiT__build_dmm_attn_mask_shared(
    self, bsz: int, motion_len: int, text_len: int, dtype: torch.dtype, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], device: torch.device
) -> Tensor:
    """
    NOTE:
            motion_k  text_k
    motion_q [M→M]   [M→T]
    text_q   [T→M]   [T→T]
    only [M→M] contains given mask
    """
    total_len = motion_len + text_len
    base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
    if attn_mask is not None:
        if attn_mask.dim() != 2 or attn_mask.shape != (motion_len, motion_len):
            raise RuntimeError(f"attn_mask should be 2D with shape {(motion_len, motion_len)}, got {attn_mask.shape}")
        base[:, :, :motion_len, :motion_len] += attn_mask.view(1, 1, motion_len, motion_len)
    if key_padding_mask is not None:
        mask_total_len = key_padding_mask.shape[1]
        if mask_total_len == motion_len:
            pad = torch.zeros((bsz, text_len), dtype=key_padding_mask.dtype, device=device)
            key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
        base = base + key_padding_mask.view(bsz, 1, 1, total_len)
    base[:, :, motion_len:, :motion_len] = float("-inf")
    return base


def HunyuanMotionMMDiT__build_smm_attn_mask_shared(
    self, bsz: int, split_len: int, total_len: int, dtype: torch.dtype, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], device: torch.device
) -> Tensor:
    """
    NOTE:
            motion_k  text_k
    motion_q [M→M]   [M→T]
    text_q   [T→M]   [T→T]
    only [M→M] contains given mask
    """
    base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
    if attn_mask is not None:
        if attn_mask.dim() != 2 or attn_mask.shape != (split_len, split_len):
            raise RuntimeError(f"attn_mask should be 2D with shape {(split_len, split_len)}, got {attn_mask.shape}")
        base[:, :, :split_len, :split_len] += attn_mask.view(1, 1, split_len, split_len)
    if key_padding_mask is not None:
        mask_total_len = key_padding_mask.shape[1]
        if mask_total_len == split_len:
            pad = torch.zeros((bsz, total_len - split_len), dtype=key_padding_mask.dtype, device=device)
            key_padding_mask = torch.cat((key_padding_mask, pad), dim=-1)
        base = base + key_padding_mask.view(bsz, 1, 1, total_len)
    base[:, :, split_len:, :split_len] = float("-inf")
    return base
