# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
# Copyright 2026 The LightX2V Team.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3VideoEncoder3d,
    MiniMaxH3VideoRotaryPosEmbed,
    MiniMaxH3VideoTransformerBlock,
    MiniMaxH3VideoVAE,
)
from lightx2v.models.video_encoders.hf.minimax_h3_image.weights import load_image_vae_weights
from lightx2v_platform.base.global_var import AI_DEVICE


def _decoder_config(config):
    original = config.get("vit_decoder_kwargs", {})
    return {
        "decoder_num_layers": config.get("decoder_num_layers", original.get("num_layers", 24)),
        "decoder_num_attention_heads": config.get("decoder_num_attention_heads", original.get("heads", 16)),
        "decoder_attention_head_dim": config.get("decoder_attention_head_dim", original.get("dim_head", 64)),
        "decoder_num_register_tokens": config.get("decoder_num_register_tokens", 4),
        "decoder_ffn_mult": config.get("decoder_ffn_mult", 4),
        "decoder_rope_theta": config.get("decoder_rope_theta", original.get("rope_theta", 100.0)),
        "decoder_rope_dim_ratio": config.get("decoder_rope_dim_ratio", original.get("rope_dim_ratio", 0.5)),
        "decoder_norm_eps": config.get("decoder_norm_eps", 1e-5),
    }


class MiniMaxH3ImageTransformerBlock(MiniMaxH3VideoTransformerBlock):
    def forward(self, hidden_states, rotary_emb):
        compute_dtype = self.attn.to_q.weight.dtype
        normed = self.norm1(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.attn(normed, rotary_emb) * self.scale1
        normed = self.norm2(hidden_states.float()).to(compute_dtype)
        return hidden_states + self.ff(normed) * self.scale2


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    def __init__(self, latent_channels, patch_size, config, attn_type):
        super().__init__()
        config = _decoder_config(config)
        heads = int(config["decoder_num_attention_heads"])
        dim_head = int(config["decoder_attention_head_dim"])
        dim = heads * dim_head
        eps = float(config["decoder_norm_eps"])
        self.patch_size = patch_size
        self.num_register_tokens = int(config["decoder_num_register_tokens"])
        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(dim_head * config["decoder_rope_dim_ratio"]), config["decoder_rope_theta"], num_axes=2)
        self.proj_in = nn.Linear(latent_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [MiniMaxH3ImageTransformerBlock(dim, heads, dim_head, ffn_mult=int(config["decoder_ffn_mult"]), eps=eps, attn_type=attn_type) for _ in range(int(config["decoder_num_layers"]))]
        )
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.proj_out = nn.Linear(dim, 3 * patch_size * patch_size)

    def forward(self, latents):
        batch_size, channels, height, width = latents.shape
        tokens = latents.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.proj_in(tokens)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        hidden_states = torch.cat((hidden_states, register_tokens, torch.zeros_like(hidden_states[:, :1])), dim=1)
        grids = [2.0 * (torch.arange(0.5, size, device=latents.device, dtype=torch.float32) / size) - 1.0 for size in (height, width)]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros(batch_size, self.num_register_tokens + 1, 2)
        rotary_emb = self.rope(torch.cat((position_ids, suffix_ids), dim=1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)
        hidden_states = self.norm_out(hidden_states).to(self.proj_out.weight.dtype)
        output = self.proj_out(hidden_states)[:, : height * width]
        patch = self.patch_size
        output = output.view(batch_size, height, width, 3, patch, patch)
        return output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, 3, height * patch, width * patch)


class MiniMaxH3ImageVAE(nn.Module):
    _keep_in_fp32_modules = {"encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"}
    _split_tiles = MiniMaxH3VideoVAE._split_tiles
    _blend = staticmethod(MiniMaxH3VideoVAE._blend)
    _stitch_clip = MiniMaxH3VideoVAE._stitch_clip

    def __init__(self, config, *, device=AI_DEVICE, cpu_offload=False, attn_type="torch_sdpa"):
        super().__init__()
        self.config = {**config, **_decoder_config(config)}
        self.execution_device = torch.device(device)
        self.cpu_offload = cpu_offload
        self.encode_parallel = self.decode_parallel = False
        self.parallel_group = None
        channels = int(config.get("latent_channels", 24))
        spatial_factors = tuple(config.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1)))
        self.spatial_compression_ratio = math.prod(spatial_factors)
        self.encoder = MiniMaxH3VideoEncoder3d(
            out_channels=2 * channels,
            block_out_channels=tuple(config.get("block_out_channels", (128, 256, 256, 512, 512, 1024))),
            layers_per_block=int(config.get("layers_per_block", 2)),
            spatial_downsample_factors=spatial_factors,
            temporal_downsample_factors=tuple(config.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1))),
            norm_num_groups=int(config.get("norm_num_groups", 32)),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            spatial_padding_mode=config.get("spatial_padding_mode", "reflect"),
            infer_dtype=torch.float32,
            sensitive_layer_dtype=torch.float32,
        )
        self.quant_conv = nn.Conv3d(2 * channels, 2 * channels, 1)
        self.post_quant_conv = nn.Conv2d(channels, channels, 1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(channels, self.spatial_compression_ratio, self.config, attn_type)
        self.stitch_preallocate = False
        self.tile_sample_min_height = self.tile_sample_min_width = int(config.get("vae_tile_size", 256))
        self.tile_sample_min_overlap_height = self.tile_sample_min_overlap_width = int(config.get("vae_tile_overlap_min", 64))
        for name, values in (
            ("latents_mean", config.get("latents_mean", [0.0] * channels)),
            ("latents_std", config.get("latents_std", [1.0] * channels)),
            ("pixel_mean", MINIMAX_H3_PIXEL_MEAN),
            ("pixel_std", MINIMAX_H3_PIXEL_STD),
        ):
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32, device="cpu").view(1, -1, 1, 1, 1), persistent=False)

    @classmethod
    def from_pretrained(cls, model_path, *, device=AI_DEVICE, cpu_offload=False, attn_type="torch_sdpa"):
        model_path = Path(model_path)
        if (model_path / "config.json").is_file():
            component = model_path
        else:
            component = model_path / ("vae_diffusers" if (model_path / "vae_diffusers").is_dir() else "vae")
        with (component / "config.json").open() as handle:
            config = json.load(handle)
        original = config.get("_class_name") != "AutoencoderKLMiniMaxH3Image"
        if original and not config.get("vit_decoder_2d", False):
            raise ValueError("MiniMax-H3-Image requires an image VAE checkpoint")
        with torch.device("meta"):
            model = cls(config, device=device, cpu_offload=cpu_offload, attn_type=attn_type)
        load_image_vae_weights(model, component, original)
        # Match Diffusers' per-module FP32 exceptions to the BF16 VAE.
        for name, parameter in model.named_parameters():
            dtype = torch.float32 if model._keep_in_fp32_modules.intersection(name.split(".")) else torch.bfloat16
            parameter.data = parameter.data.to(dtype=dtype)
        model.eval().requires_grad_(False)
        if not cpu_offload:
            model.to(device=model.execution_device)
        return model

    def _activate(self, encode):
        modules = (self.encoder, self.quant_conv) if encode else (self.post_quant_conv, self.decoder)
        for module in modules:
            module.to(device=self.execution_device)
        return self.execution_device

    def offload(self):
        for module in (self.encoder, self.quant_conv, self.post_quant_conv, self.decoder):
            module.to(device="cpu")

    def _enable_parallel(self):
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            raise RuntimeError("MiniMax-H3-Image VAE parallel execution requires an initialized multi-rank process group")
        if self.parallel_group is None:
            self.parallel_group = dist.new_group()
            # Initialize every member before the decoder uses subset P2P.
            dist.all_reduce(torch.zeros(1, device=self.execution_device), group=self.parallel_group)

    def enable_encode_parallel(self):
        self._enable_parallel()
        self.encode_parallel = True

    def enable_decode_parallel(self):
        self._enable_parallel()
        self.decode_parallel = True

    def _tile_layout(self, tensor, encode):
        ratio = self.spatial_compression_ratio
        height, width = tensor.shape[-2:]
        if not encode:
            height, width = height * ratio, width * ratio
        y_starts, y_lengths, y_overlaps = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)
        x_starts, x_lengths, x_overlaps = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)
        input_ratio = 1 if encode else ratio
        slices = [(slice(y // input_ratio, (y + h) // input_ratio), slice(x // input_ratio, (x + w) // input_ratio)) for y, h in zip(y_starts, y_lengths) for x, w in zip(x_starts, x_lengths)]
        output_ratio = ratio if encode else 1
        return slices, [value // output_ratio for value in y_overlaps], [value // output_ratio for value in x_overlaps]

    def _forward_tile(self, tile, encode):
        if encode:
            return self.quant_conv(self.encoder(tile))
        return self.decoder(self.post_quant_conv(tile[:, :, 0])).unsqueeze(2)

    def _tiled_forward(self, tensor, encode):
        slices, y_overlaps, x_overlaps = self._tile_layout(tensor, encode)
        tiles = [self._forward_tile(tensor[..., y, x], encode) for y, x in slices]
        return self._stitch_clip(tiles, y_overlaps, x_overlaps)

    def _raise_parallel_errors(self, error):
        errors = [None] * dist.get_world_size(self.parallel_group)
        dist.all_gather_object(errors, f"{type(error).__name__}: {error}" if error is not None else None, group=self.parallel_group)
        for rank, detail in enumerate(errors):
            if detail is not None:
                raise RuntimeError(f"MiniMax-H3-Image VAE rank {rank} failed: {detail}") from error

    def _collect_tiles(self, local_tiles, task_counts, *, all_gather):
        """Collect variable tile shapes/counts without changing global tile order."""
        group = self.parallel_group
        rank, world_size = dist.get_rank(group), dist.get_world_size(group)
        metadata = [None] * world_size
        dist.all_gather_object(metadata, ([(tuple(tile.shape)) for tile in local_tiles], local_tiles[0].dtype if local_tiles else None), group=group)
        if any(len(shapes) != count for (shapes, _), count in zip(metadata, task_counts)):
            raise RuntimeError("MiniMax-H3-Image VAE tile counts do not match the global task layout")
        dtype = next(dtype for _, dtype in metadata if dtype is not None)
        if any(other is not None and other != dtype for _, other in metadata):
            raise RuntimeError("MiniMax-H3-Image VAE tile dtypes differ across ranks")
        counts = [sum(math.prod(shape) for shape in shapes) for shapes, _ in metadata]
        packed = torch.cat([tile.reshape(-1) for tile in local_tiles]) if local_tiles else torch.empty(0, device=self.execution_device, dtype=dtype)
        if all_gather:
            padded = packed.new_zeros(max(counts))
            padded[: packed.numel()].copy_(packed)
            buffers = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(buffers, padded, group=group)
        elif rank == 0:
            buffers = [packed, *[packed.new_empty(count) for count in counts[1:]]]
            ops = [dist.P2POp(dist.irecv, buffers[peer], peer, group=group) for peer in range(1, world_size) if counts[peer]]
            if ops:
                for request in dist.batch_isend_irecv(ops):
                    request.wait()
        else:
            if packed.numel():
                for request in dist.batch_isend_irecv([dist.P2POp(dist.isend, packed, 0, group=group)]):
                    request.wait()
            return None
        tiles = []
        for buffer, (shapes, _) in zip(buffers, metadata):
            offset = 0
            for shape in shapes:
                length = math.prod(shape)
                tiles.append(buffer[offset : offset + length].view(shape))
                offset += length
        return tiles

    def _parallel_tiled_forward(self, tensors, encode):
        group = self.parallel_group
        rank, world_size = dist.get_rank(group), dist.get_world_size(group)
        error, signature = None, None
        try:
            signature = (
                [(tuple(tensor.shape), tensor.dtype) for tensor in tensors],
                encode,
                self.tile_sample_min_height,
                self.tile_sample_min_width,
                self.tile_sample_min_overlap_height,
                self.tile_sample_min_overlap_width,
            )
        except Exception as exc:
            error = exc
        self._raise_parallel_errors(error)
        signatures = [None] * world_size
        dist.all_gather_object(signatures, signature, group=group)
        if any(other != signature for other in signatures):
            raise ValueError("MiniMax-H3-Image VAE parallel ranks must receive the same input shapes and tile settings")
        if not tensors:
            return []
        local_tiles, layouts, error = [], [], None
        try:
            device = self._activate(encode)
            prepared = [self._prepare_pixels(tensor, device) if encode else self._prepare_latents(tensor, device) for tensor in tensors]
            layouts = [self._tile_layout(tensor, encode) for tensor in prepared]
            tasks = [(image_index, y, x) for image_index, (slices, _, _) in enumerate(layouts) for y, x in slices]
            task_counts = MiniMaxH3VideoVAE._balanced_task_counts(len(tasks), world_size)
            start = sum(task_counts[:rank])
            with torch.autocast(device_type=device.type, enabled=False):
                for image_index, y, x in tasks[start : start + task_counts[rank]]:
                    local_tiles.append(self._forward_tile(prepared[image_index][..., y, x], encode))
            logger.info(f"MiniMax-H3-Image VAE {'encode' if encode else 'decode'} tile parallel: rank {rank}/{world_size}, images={len(tensors)}, tiles={task_counts[rank]}/{len(tasks)}")
        except Exception as exc:
            error = exc
        finally:
            if self.cpu_offload:
                try:
                    self.offload()
                except Exception as exc:
                    if error is None:
                        error = exc
        self._raise_parallel_errors(error)
        tiles = self._collect_tiles(local_tiles, task_counts, all_gather=encode)
        stitched, error = None, None
        try:
            if tiles is not None:
                stitched, start = [], 0
                for slices, y_overlaps, x_overlaps in layouts:
                    end = start + len(slices)
                    stitched.append(self._stitch_clip(tiles[start:end], y_overlaps, x_overlaps))
                    start = end
        except Exception as exc:
            error = exc
        self._raise_parallel_errors(error)
        return stitched

    def _prepare_pixels(self, pixels, device):
        if pixels.ndim != 5 or tuple(pixels.shape[:3]) != (1, 3, 1):
            raise ValueError(f"Image VAE pixels must be [1,3,1,H,W], got {tuple(pixels.shape)}")
        ratio = self.spatial_compression_ratio
        height, width = (size // ratio * ratio for size in pixels.shape[-2:])
        if min(height, width) < 2 * ratio:
            raise ValueError("Image VAE references need at least two latent cells on each axis")
        pixels = pixels[..., :height, :width].to(device=device)
        byte_pixels = pixels.dtype == torch.uint8
        pixels = pixels.to(dtype=torch.float32)
        if byte_pixels:
            pixels = pixels.div(255.0)
        return (pixels - self.pixel_mean.to(device)) / self.pixel_std.to(device)

    def _sample_condition_latents(self, moments, generator):
        mean, logvar = moments.float().chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=generator, device="cpu" if generator is None else generator.device, dtype=torch.float32).to(mean.device)
        latents = (mean + logvar.clamp(-30.0, 20.0).mul(0.5).exp() * noise).half().float()
        # Preserve draws for the uncropped posterior before making even grids.
        latent_height, latent_width = latents.shape[-2:]
        latents = latents[..., : latent_height // 2 * 2, : latent_width // 2 * 2]
        return (latents - self.latents_mean.to(mean.device)) / self.latents_std.to(mean.device)

    @torch.no_grad()
    def encode_conditions(self, pixels_list, *, generator=None):
        """Encode ordered references; every parallel rank uses the same CPU RNG stream."""
        if not self.encode_parallel:
            return [self.encode_condition(pixels, generator=generator) for pixels in pixels_list]
        error, state = None, None
        try:
            if generator is not None and generator.device.type != "cpu":
                raise ValueError("MiniMax-H3-Image parallel VAE encoding requires a CPU generator")
            state = bytes((generator.get_state() if generator is not None else torch.get_rng_state()).tolist())
        except Exception as exc:
            error = exc
        self._raise_parallel_errors(error)
        states = [None] * dist.get_world_size(self.parallel_group)
        dist.all_gather_object(states, state, group=self.parallel_group)
        if any(other != state for other in states):
            raise ValueError("MiniMax-H3-Image parallel VAE ranks must use the same CPU generator state")
        moments = self._parallel_tiled_forward(pixels_list, encode=True)
        latents, error = None, None
        try:
            latents = [self._sample_condition_latents(value, generator) for value in moments]
        except Exception as exc:
            error = exc
        self._raise_parallel_errors(error)
        return latents

    @torch.no_grad()
    def encode_condition(self, pixels, *, generator=None):
        """Encode uint8 RGB, or pre-scaled floating RGB in [0, 1]."""
        if self.encode_parallel:
            return self.encode_conditions([pixels], generator=generator)[0]
        try:
            device = self._activate(encode=True)
            pixels = self._prepare_pixels(pixels, device)
            with torch.autocast(device_type=device.type, enabled=False):
                moments = self._tiled_forward(pixels, encode=True)
            return self._sample_condition_latents(moments, generator)
        finally:
            if self.cpu_offload:
                self.offload()

    def _prepare_latents(self, normalized_latents, device):
        if normalized_latents.ndim != 5 or normalized_latents.shape[2] != 1:
            raise ValueError(f"Image VAE latents must be [B,C,1,H,W], got {tuple(normalized_latents.shape)}")
        latents = normalized_latents.to(device=device, dtype=torch.float32)
        latents = latents * self.latents_std.to(device) + self.latents_mean.to(device)
        return latents.to(self.post_quant_conv.weight.dtype)

    def _decode_pixels(self, decoded):
        return (decoded.float() * self.pixel_std.to(decoded.device) + self.pixel_mean.to(decoded.device)).clamp(0, 1)

    @torch.no_grad()
    def decode(self, normalized_latents):
        if self.decode_parallel:
            decoded = self._parallel_tiled_forward([normalized_latents], encode=False)
            pixels, error = None, None
            try:
                if decoded is not None:
                    pixels = self._decode_pixels(decoded[0])
            except Exception as exc:
                error = exc
            self._raise_parallel_errors(error)
            return pixels
        try:
            device = self._activate(encode=False)
            latents = self._prepare_latents(normalized_latents, device)
            with torch.autocast(device_type=device.type, enabled=False):
                decoded = self._tiled_forward(latents, encode=False)
            return self._decode_pixels(decoded)
        finally:
            if self.cpu_offload:
                self.offload()
