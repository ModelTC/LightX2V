"""Checkpoint-native camera-conditioned DINOv2-L and OpenCLIP ViT-B/32."""

import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.three_models.ops import Attention

_attention = Attention()


def dinov2_image(w, images, condition):
    packed = images.ndim == 4
    if packed:
        images = images[:, None]
        condition = condition[:, None]
    b, views, _c, h, width = images.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 1, 3, 1, 1)
    pixels = ((images - mean) / std).flatten(0, 1)
    condition = condition.flatten(0, 1)
    p = "embeddings"
    x = w.conv(p + ".patch_embeddings.projection", stride=14)(pixels).flatten(2).transpose(1, 2)
    cls = w.tensor(p + ".cls_token").expand(b * views, -1, -1)
    x = torch.cat((cls, x), dim=1)
    pos = w.tensor(p + ".position_embeddings")
    if x.shape[1] != pos.shape[1] or h != width:
        grid = math.isqrt(pos.shape[1] - 1)
        patch = pos[:, 1:].reshape(1, grid, grid, 1024).permute(0, 3, 1, 2)
        patch = F.interpolate(patch, scale_factor=((h // 14 + 0.1) / grid, (width // 14 + 0.1) / grid), mode="bicubic", align_corners=False)
        pos = torch.cat((pos[:, :1], patch.permute(0, 2, 3, 1).reshape(1, -1, 1024)), dim=1)
    x = x + pos
    batch, n, dim = x.shape
    for i in range(24):
        p = f"encoder.layer.{i}"

        def normalized(k):
            y = w.norm(p + f".norm{k}", 1e-6)(x)  # noqa: B023 - synchronous calls, x includes the current residual
            scale, shift = w.linear(p + f".norm{k}_modulation.linear2")(F.silu(condition)).chunk(2, dim=-1)  # noqa: B023 - synchronous calls, x includes the current residual
            return y * (1 + scale[:, None]) + shift[:, None]

        y = normalized(1)
        q, k, v = [w.linear(p + ".attention.attention." + key)(y).reshape(batch, n, 16, 64).transpose(1, 2) for key in ("query", "key", "value")]
        y = _attention(q, k, v).transpose(1, 2).reshape(batch, n, dim)
        y = w.linear(p + ".attention.output.dense")(y) * w.tensor(p + ".layer_scale1.lambda1")
        x = x + y
        y = w.linear(p + ".mlp.fc2")(F.gelu(w.linear(p + ".mlp.fc1")(normalized(2))))
        x = x + y * w.tensor(p + ".layer_scale2.lambda1")
    x = w.norm("layernorm", 1e-6)(x)
    x = x.transpose(1, 2).reshape(b, views, dim, n)
    return x[:, 0] if packed else x


def openclip_image(w, pixels):
    def norm(name, value):
        # OpenCLIP's LayerNorm explicitly casts its result back to the input
        # dtype. Autocast otherwise promotes F.layer_norm to FP32.
        return w.norm(name)(value).to(value.dtype)

    x = w.conv("conv1", stride=32)(pixels).flatten(2).transpose(1, 2)
    cls = w.tensor("class_embedding").to(x.dtype)
    x = torch.cat((cls[None, None].expand(x.shape[0], 1, -1), x), dim=1)
    x = x + w.tensor("positional_embedding").to(x.dtype)
    x = norm("ln_pre", x)
    b, n, c = x.shape
    for i in range(12):
        p = f"transformer.resblocks.{i}"
        y = norm(p + ".ln_1", x)
        # OpenCLIP stores nn.MultiheadAttention's packed projection names.
        name = p + ".attn.in_proj"
        if name + ".weight" not in w.tensors:
            w.tensors[name + ".weight"] = w.tensors[p + ".attn.in_proj_weight"]
            w.tensors[name + ".bias"] = w.tensors[p + ".attn.in_proj_bias"]
        q, k, v = w.linear(name)(y).reshape(b, n, 3, 12, 64).permute(2, 0, 3, 1, 4).unbind(0)
        y = _attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        x = x + w.linear(p + ".attn.out_proj")(y)
        y = w.linear(p + ".mlp.c_fc")(norm(p + ".ln_2", x))
        # The saved LAION ViT-B/32 uses GELU (not OpenAI QuickGELU).
        x = x + w.linear(p + ".mlp.c_proj")(F.gelu(y))
    x = norm("ln_post", x[:, 0])
    if "projection.weight" not in w.tensors:
        w.tensors["projection.weight"] = w.tensor("proj").T
    return w.linear("projection")(x)
