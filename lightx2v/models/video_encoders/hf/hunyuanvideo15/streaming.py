"""Causal decoder state for the native HunyuanVideo 1.5 codec."""

import torch
from einops import rearrange
from torch.nn import functional as F

from lightx2v.models.video_encoders.hf.hunyuanvideo15.hunyuanvideo_15_vae import _pad, swish


class HunyuanVideo15StreamingDecoder:
    def __init__(self, vae):
        self.vae = vae
        self.tile_states = {}
        self.first_chunk = True

    def conv(self, layer, x, state):
        history = state.get(layer)
        temporal = layer.time_causal_padding[4]
        if temporal:
            joined = _pad(x, (0, 0, 0, 0, temporal, 0), mode=layer.pad_mode) if history is None else torch.cat((history, x), dim=2)
            state[layer] = joined[:, :, -temporal:].clone()
        else:
            joined = x
        return layer.conv(_pad(joined, (*layer.time_causal_padding[:4], 0, 0), mode=layer.pad_mode))

    def residual(self, layer, x, state):
        h = self.conv(layer.conv1, swish(layer.norm1(x)), state)
        h = self.conv(layer.conv2, swish(layer.norm2(h)), state)
        shortcut = layer.nin_shortcut(x) if layer.in_channels != layer.out_channels else x
        return shortcut + h

    def attention(self, layer, x, state):
        h = layer.norm(x)
        q, k, v = (projection(h) for projection in (layer.q, layer.k, layer.v))
        batch, channels, frames, height, width = q.shape
        q, k, v = (value.flatten(2).transpose(1, 2).unsqueeze(1).contiguous() for value in (q, k, v))
        history = state.get(layer)
        past_frames = 0
        if history is not None:
            past_frames = history[0].shape[2] // (height * width)
            k, v = torch.cat((history[0], k), dim=2), torch.cat((history[1], v), dim=2)
        state[layer] = (k, v)
        query_frames = torch.arange(past_frames, past_frames + frames, device=q.device).repeat_interleave(height * width)
        key_frames = torch.arange(past_frames + frames, device=q.device).repeat_interleave(height * width)
        mask = q.new_full((q.shape[2], k.shape[2]), float("-inf"))
        mask.masked_fill_(query_frames[:, None] >= key_frames[None, :], 0)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
        h = h.squeeze(1).transpose(1, 2).reshape(batch, channels, frames, height, width)
        return x + layer.proj_out(h)

    def upsample(self, layer, x, state):
        temporal = 2 if layer.add_temporal_upsample else 1
        h = self.conv(layer.conv, x, state)
        if layer.add_temporal_upsample and self.first_chunk:
            first = rearrange(h[:, :, :1], "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            first = first[:, : first.shape[1] // 2]
            remaining = rearrange(h[:, :, 1:], "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=2, r2=2, r3=2)
            h = torch.cat((first, remaining), dim=2)
            first = rearrange(x[:, :, :1], "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            first = first.repeat_interleave(layer.repeats // 2, dim=1)
            remaining = rearrange(x[:, :, 1:], "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=2, r2=2, r3=2)
            shortcut = torch.cat((first, remaining.repeat_interleave(layer.repeats, dim=1)), dim=2)
        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=temporal, r2=2, r3=2)
            shortcut = rearrange(x.repeat_interleave(layer.repeats, dim=1), "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=temporal, r2=2, r3=2)
        return h + shortcut

    def decode_tile(self, z, tile_id):
        state = self.tile_states.setdefault(tile_id, {})
        decoder = self.vae.decoder
        h = self.conv(decoder.conv_in, z, state) + z.repeat_interleave(decoder.block_out_channels[0] // decoder.z_channels, dim=1)
        h = self.residual(decoder.mid.block_1, h, state)
        h = self.attention(decoder.mid.attn_1, h, state)
        h = self.residual(decoder.mid.block_2, h, state)
        for stage in decoder.up:
            for block in stage.block:
                h = self.residual(block, h, state)
            if hasattr(stage, "upsample"):
                h = self.upsample(stage.upsample, h, state)
        return self.conv(decoder.conv_out, swish(decoder.norm_out(h)), state)

    def decode(self, z):
        vae = self.vae
        if vae.use_spatial_tiling and max(z.shape[-2:]) > vae.tile_latent_min_size:
            stride = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
            blend = int(vae.tile_sample_min_size * vae.tile_overlap_factor)
            limit = vae.tile_sample_min_size - blend
            rows = []
            for top in range(0, z.shape[-2], stride):
                row = []
                for left in range(0, z.shape[-1], stride):
                    tile = z[..., top : top + vae.tile_latent_min_size, left : left + vae.tile_latent_min_size]
                    row.append(self.decode_tile(tile, (top, left)))
                rows.append(row)
            assembled = []
            for top, row in enumerate(rows):
                columns = []
                for left, tile in enumerate(row):
                    if top:
                        tile = vae.blend_v(rows[top - 1][left], tile, blend)
                    if left:
                        tile = vae.blend_h(row[left - 1], tile, blend)
                    columns.append(tile[..., :limit, :limit])
                assembled.append(torch.cat(columns, dim=-1))
            images = torch.cat(assembled, dim=-2)
        else:
            images = self.decode_tile(z, (0, 0))
        self.first_chunk = False
        return images
