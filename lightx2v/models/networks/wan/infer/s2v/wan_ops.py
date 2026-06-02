# Vendored / mirrored from Wan2.2 for numerical alignment.
import torch

try:
    import flash_attn_interface
except ImportError:
    flash_attn_interface = None


def mm_weight_fp32_nd(linear, x):
    """FP32 linear on [..., in_dim], matching nn.Linear on 3D [B, L, C]."""
    from lightx2v.models.networks.wan.infer.s2v.causal_audio_encoder import mm_weight_fp32

    shape = x.shape
    out = mm_weight_fp32(linear, x.reshape(-1, shape[-1]))
    return out.view(*shape[:-1], -1)


def mm_weight_autocast_nd(linear, x, autocast_dtype=torch.bfloat16):
    """Match nn.Linear under torch.amp.autocast(param_dtype): fp32 input + bf16 weight -> bf16 matmul."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    w = linear._get_actual_weight()
    b = linear._get_actual_bias() if linear.bias is not None else None

    if flat.dtype == torch.float32 and w.dtype in (torch.float16, torch.bfloat16):
        inp = flat.to(autocast_dtype)
        w = w.to(autocast_dtype)
        if b is not None:
            b = b.to(autocast_dtype)
    else:
        inp = flat

    if b is not None:
        out = torch.addmm(b, inp, w)
    else:
        out = torch.mm(inp, w)
    return out.view(*shape[:-1], -1)


def wan_layer_norm(ln_weight, x, force_float=False):
    weight = ln_weight._get_actual_weight() if ln_weight.weight is not None else None
    bias = ln_weight._get_actual_bias() if getattr(ln_weight, "bias", None) is not None else None
    if weight is not None:
        weight = weight.float()
    if bias is not None:
        bias = bias.float()
    out = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight, bias, ln_weight.eps)
    out = out.to(x.dtype)
    return out.float() if force_float else out


def wan_layer_norm_float(ln_weight, x):
    return wan_layer_norm(ln_weight, x, force_float=True)


def wan_rms_norm(rms_weight, x):
    w = rms_weight._get_actual_weight()
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + rms_weight.eps)
    return normed.type_as(x) * w


def segment_modulate_bld(norm_x, shift, scale, seg_idx):
    parts = []
    for i in range(2):
        parts.append(norm_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + scale[:, i : i + 1]) + shift[:, i : i + 1])
    return torch.cat(parts, dim=1)


def segment_gate_bld(y, gate, seg_idx):
    parts = []
    for i in range(2):
        parts.append(y[:, seg_idx[i] : seg_idx[i + 1]] * gate[:, i : i + 1])
    return torch.cat(parts, dim=1)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
):
    """Mirror of Wan2.2/wan/modules/attention.py flash_attention."""
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if flash_attn_interface is None:
        raise RuntimeError("flash_attn3 is required for Wan S2V alignment")

    x = flash_attn_interface.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device),
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    if isinstance(x, tuple):
        x = x[0]
    return x.unflatten(0, (b, lq)).type(out_dtype)


def s2v_self_attn_forward(phase0, norm_x, seq_lens, freqs, num_heads, head_dim, rope_apply_fn):
    """Mirror of WanS2VSelfAttention.forward (model_s2v.py + model.py)."""
    b, s, n, d = norm_x.size(0), norm_x.size(1), num_heads, head_dim

    q = wan_rms_norm(phase0.self_attn_norm_q, mm_weight_autocast_nd(phase0.self_attn_q, norm_x)).view(b, s, n, d)
    k = wan_rms_norm(phase0.self_attn_norm_k, mm_weight_autocast_nd(phase0.self_attn_k, norm_x)).view(b, s, n, d)
    v = mm_weight_autocast_nd(phase0.self_attn_v, norm_x).view(b, s, n, d)

    attn = flash_attention(
        q=rope_apply_fn(q, freqs),
        k=rope_apply_fn(k, freqs),
        v=v,
        k_lens=seq_lens,
    )
    return mm_weight_autocast_nd(phase0.self_attn_o, attn.flatten(2))


def _attn_modules(phase, role):
    for prefix in ("cross_attn_", ""):
        key = f"{prefix}{role}"
        if hasattr(phase, key):
            return getattr(phase, key)
    raise AttributeError(f"Missing attention module {role!r} on {phase!r}")


def cross_attn_forward(phase1, norm_x, context, context_lens, num_heads, head_dim):
    """Mirror of WanCrossAttention.forward."""
    b, n, d = norm_x.size(0), num_heads, head_dim
    if context.dim() == 2:
        context = context.unsqueeze(0)

    q = wan_rms_norm(
        _attn_modules(phase1, "norm_q"),
        mm_weight_autocast_nd(_attn_modules(phase1, "q"), norm_x),
    ).view(b, -1, n, d)
    k = wan_rms_norm(
        _attn_modules(phase1, "norm_k"),
        mm_weight_autocast_nd(_attn_modules(phase1, "k"), context),
    ).view(b, -1, n, d)
    v = mm_weight_autocast_nd(_attn_modules(phase1, "v"), context).view(b, -1, n, d)

    attn = flash_attention(q, k, v, k_lens=context_lens)
    return mm_weight_autocast_nd(_attn_modules(phase1, "o"), attn.flatten(2))
