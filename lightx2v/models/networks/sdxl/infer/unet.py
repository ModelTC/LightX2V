"""SDXL base UNet: residual CNN, spatial transformers and conditioning."""

import math

import torch
import torch.nn.functional as F

from lightx2v.models.networks.three_models.ops import Attention


def timestep_embedding(t, channels):
    half = channels // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    phases = t.float().reshape(-1, 1) * frequencies[None]
    return torch.cat((phases.cos(), phases.sin()), dim=-1)


def group_norm(w, p, x, eps=1e-5):
    weight, bias = w.group_norm(p)
    return F.group_norm(x, 32, weight, bias, eps)


def resnet(w, p, x, temb=None, eps=1e-5):
    y = w.conv(p + ".conv1", padding=1)(F.silu(group_norm(w, p + ".norm1", x, eps)))
    if temb is not None:
        y = y + w.linear(p + ".time_emb_proj")(F.silu(temb))[:, :, None, None]
    y = w.conv(p + ".conv2", padding=1)(F.silu(group_norm(w, p + ".norm2", y, eps)))
    if p + ".conv_shortcut.weight" in w.tensors:
        x = w.conv(p + ".conv_shortcut")(x)
    return x + y


class SDXLUNetInfer:
    def __init__(self, config):
        self.config = config
        self.attention = Attention()

    def attend(self, w, p, x, context, heads):
        b, n, c = x.shape
        q = w.linear(p + ".to_q")(x).reshape(b, n, heads, -1).transpose(1, 2)
        k, v = [w.linear(p + ".to_" + t)(context).reshape(b, context.shape[1], heads, -1).transpose(1, 2) for t in ("k", "v")]
        y = self.attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return w.linear(p + ".to_out.0")(y)

    def spatial_transformer(self, w, p, x, context, heads, depth):
        residual = x
        x = group_norm(w, p + ".norm", x, 1e-6)
        b, c, h, width = x.shape
        x = w.linear(p + ".proj_in")(x.flatten(2).transpose(1, 2))
        for j in range(depth):
            q = p + f".transformer_blocks.{j}"
            y = w.norm(q + ".norm1")(x)
            x = x + self.attend(w, q + ".attn1", y, y, heads)
            y = w.norm(q + ".norm2")(x)
            x = x + self.attend(w, q + ".attn2", y, context, heads)
            y = w.linear(q + ".ff.net.0.proj")(w.norm(q + ".norm3")(x))
            value, gate = y.chunk(2, dim=-1)
            x = x + w.linear(q + ".ff.net.2")(value * F.gelu(gate))
        return residual + w.linear(p + ".proj_out")(x).transpose(1, 2).reshape(b, c, h, width)

    def infer(self, w, sample, timestep, context, pooled, time_ids):
        cfg = self.config
        t = timestep_embedding(timestep.expand(sample.shape[0]), 320).to(sample.dtype)
        temb = w.linear("time_embedding.linear_2")(F.silu(w.linear("time_embedding.linear_1")(t)))
        extra = timestep_embedding(time_ids.flatten(), 256).reshape(sample.shape[0], -1).to(sample.dtype)
        extra = torch.cat((pooled, extra), dim=-1)
        temb = temb + w.linear("add_embedding.linear_2")(F.silu(w.linear("add_embedding.linear_1")(extra)))
        x = w.conv("conv_in", padding=1)(sample)
        skips = [x]
        for i in range(3):
            p = f"down_blocks.{i}"
            for j in range(2):
                x = resnet(w, p + f".resnets.{j}", x, temb)
                if i > 0:
                    x = self.spatial_transformer(w, p + f".attentions.{j}", x, context, cfg["attention_head_dim"][i], cfg["transformer_layers_per_block"][i])
                skips.append(x)
            if i < 2:
                x = w.conv(p + ".downsamplers.0.conv", stride=2, padding=1)(x)
                skips.append(x)
        x = resnet(w, "mid_block.resnets.0", x, temb)
        x = self.spatial_transformer(w, "mid_block.attentions.0", x, context, 20, 10)
        x = resnet(w, "mid_block.resnets.1", x, temb)
        for i in range(3):
            p = f"up_blocks.{i}"
            for j in range(3):
                x = resnet(w, p + f".resnets.{j}", torch.cat((x, skips.pop()), dim=1), temb)
                if i < 2:
                    x = self.spatial_transformer(w, p + f".attentions.{j}", x, context, cfg["attention_head_dim"][2 - i], cfg["transformer_layers_per_block"][2 - i])
            if i < 2:
                x = w.conv(p + ".upsamplers.0.conv", padding=1)(F.interpolate(x, scale_factor=2, mode="nearest"))
        return w.conv("conv_out", padding=1)(F.silu(group_norm(w, "conv_norm_out", x)))
