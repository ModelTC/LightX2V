import torch

from lightx2v.models.networks.hidream_o1_image.qwen3_vl import apply_rotary_pos_emb

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None


def apply_hidream_rope_with_torch(q, k, rope_cos_sin):
    cos, sin = rope_cos_sin[:2]
    q_rope = q.transpose(1, 2)
    k_rope = k.transpose(1, 2)
    q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
    return q_rope.transpose(1, 2).contiguous(), k_rope.transpose(1, 2).contiguous()


def apply_hidream_rope_with_flashinfer(q, k, rope_cos_sin):
    if apply_rope_with_cos_sin_cache_inplace is None:
        raise ImportError("flashinfer is required when hidream_o1_image rope_type='flashinfer'.")
    if q.shape[0] != 1:
        raise NotImplementedError("HiDream flashinfer RoPE currently expects batch=1 CFG forwards.")

    cos, sin = rope_cos_sin[:2]
    seq_len, q_heads, head_dim = q.shape[1], q.shape[2], q.shape[3]
    kv_heads = k.shape[2]
    rotary_dim = head_dim // 2
    cos_sin_cache = torch.cat([cos[0, :, :rotary_dim].float(), sin[0, :, :rotary_dim].float()], dim=-1).contiguous()
    if len(rope_cos_sin) > 2:
        positions = rope_cos_sin[2]
    else:
        positions = torch.arange(seq_len, device=q.device, dtype=torch.long)

    query = q[0].reshape(seq_len, q_heads * head_dim).contiguous()
    key = k[0].reshape(seq_len, kv_heads * head_dim).contiguous()
    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=True,
    )
    return query.view(1, seq_len, q_heads, head_dim), key.view(1, seq_len, kv_heads, head_dim)
