from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER

_attention = None


def attention(q, k, v, mode="torch", attn_mask=None, causal=False, drop_rate=0.0, **kwargs):
    global _attention
    if mode != "torch" or drop_rate:
        raise ValueError("HY-Motion inference requires masked torch_sdpa attention")
    if _attention is None:
        _attention = ATTN_WEIGHT_REGISTER["torch_sdpa"]()
    b, s, h, d = q.shape
    return _attention.apply(q, k, v, attn_mask=attn_mask, causal=causal).reshape(b, s, h * d)
