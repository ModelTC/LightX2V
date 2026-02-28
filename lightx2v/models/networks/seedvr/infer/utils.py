import torch
from einops import rearrange

from lightx2v.models.networks.seedvr.utils.ops import slice_inputs


def rms_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    out = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return out.type_as(x)


def layer_norm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    var = (x_float - mean).pow(2).mean(dim=-1, keepdim=True)
    out = (x_float - mean) * torch.rsqrt(var + eps)
    return out.type_as(x)


def norm_no_weight(x: torch.Tensor, norm_type: str, eps: float) -> torch.Tensor:
    if norm_type is None:
        return x
    if norm_type in ["rms", "fusedrms"]:
        return rms_norm_no_weight(x, eps)
    if norm_type in ["layer", "fusedln"]:
        return layer_norm_no_weight(x, eps)
    raise NotImplementedError(f"Unsupported norm type: {norm_type}")


def apply_adaln_single(
    hid: torch.Tensor,
    emb: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    mode: str,
    cache,
    hid_len: torch.LongTensor,
    branch_tag: str,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    emb = rearrange(emb, "b (d l g) -> b d l g", l=num_layers, g=3)[..., layer_idx, :]
    target_dim = shift.shape[-1] if shift is not None else emb.shape[1]
    if emb.shape[1] != target_dim:
        if emb.shape[1] > target_dim:
            emb = emb[:, :target_dim, ...]
        else:
            raise RuntimeError(f"AdaLN embedding dim mismatch: emb_dim={emb.shape[1]} target_dim={target_dim}")

    if hid_len is not None:
        emb = cache(
            f"emb_repeat_{layer_idx}_{branch_tag}",
            lambda: slice_inputs(
                torch.cat([e.repeat(int(hl), *([1] * e.ndim)) for e, hl in zip(emb, hid_len)]),
                dim=0,
            ),
        )

    shift_a, scale_a, gate_a = emb.unbind(-1)

    if mode == "in":
        return hid.mul(scale_a + scale).add_(shift_a + shift)
    if mode == "out":
        return hid.mul(gate_a + gate)

    raise NotImplementedError(f"Unsupported AdaLN mode: {mode}")
