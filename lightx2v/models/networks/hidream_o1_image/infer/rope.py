import torch

from lightx2v.common.ops.rope import FlashInferRope, TorchRealRope

_HIDREAM_TORCH_ROPE = TorchRealRope(layout="split_half")
_HIDREAM_FLASH_ROPE = FlashInferRope(layout="split_half")


def apply_hidream_rope_with_torch(q, k, rope_cos_sin):
    cos, sin = rope_cos_sin[:2]
    q_out, k_out = _HIDREAM_TORCH_ROPE.apply(
        q.transpose(1, 2),
        k.transpose(1, 2),
        (cos, sin),
        unsqueeze_dim=1,
    )
    return q_out.transpose(1, 2).contiguous(), k_out.transpose(1, 2).contiguous()


def apply_hidream_rope_with_flashinfer(q, k, rope_cos_sin):
    if q.shape[0] != 1:
        raise NotImplementedError("HiDream FlashInfer RoPE expects batch=1 CFG forwards.")
    cos, sin = rope_cos_sin[:2]
    head_dim = q.shape[-1]
    cache = torch.cat(
        [cos[0, :, : head_dim // 2].float(), sin[0, :, : head_dim // 2].float()],
        dim=-1,
    ).contiguous()
    positions = rope_cos_sin[2] if len(rope_cos_sin) > 2 else None
    q_out, k_out = _HIDREAM_FLASH_ROPE.apply(q[0], k[0], cache, positions=positions)
    return q_out.unsqueeze(0), k_out.unsqueeze(0)
