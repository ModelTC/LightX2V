import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from lightx2v.models.video_encoders.hf.cosmos3.vae import (
    CACHE_T,
    DecoderOutput,
    WanAttentionBlock,
    WanCausalConv3d,
    WanMidBlock,
    WanRMSNorm,
    WanResample,
    WanResidualBlock,
    patchify,
    unpatchify,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


class LingBotVideoWanUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_res_blocks, dropout=0.0, upsample_mode=None, non_linearity="silu"):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=None):
        if feat_idx is None:
            feat_idx = [0]
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class LingBotVideoWanEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_downsample=None,
        dropout=0.0,
        non_linearity="silu",
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]

        dims = [dim * u for u in [1] + list(dim_mult)]
        scale = 1.0
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(WanAttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(WanResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = WanMidBlock(out_dim, dropout)
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx) if feat_cache is not None else layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = F.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class LingBotVideoWanDecoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        temperal_upsample=None,
        dropout=0.0,
        non_linearity="silu",
        out_channels=3,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]

        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult)[::-1]]
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0], dropout)
        self.up_blocks = nn.ModuleList([])

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            self.up_blocks.append(
                LingBotVideoWanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            )

        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        x = F.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class LingBotVideoAutoencoderKLWanDecodeOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.z_dim = config["z_dim"]
        self.patch_size = config.get("patch_size")
        self.post_quant_conv = WanCausalConv3d(self.z_dim, self.z_dim, 1)
        self.decoder = LingBotVideoWanDecoder3d(
            dim=config.get("decoder_base_dim", config["base_dim"]),
            z_dim=self.z_dim,
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            temperal_upsample=list(config["temperal_downsample"])[::-1],
            dropout=config.get("dropout", 0.0),
            non_linearity=config.get("non_linearity", "silu"),
            out_channels=config["out_channels"],
        )
        self._conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.decoder.modules())
        self.clear_cache()

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    def decode(self, z, return_dict=True):
        _, _, num_frame, _, _ = z.shape
        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(num_frame):
            self._conv_idx = [0]
            decoded = self.decoder(
                x[:, :, i : i + 1],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=(i == 0),
            )
            out = decoded if out is None else torch.cat([out, decoded], dim=2)
        if self.patch_size is not None:
            out = unpatchify(out, self.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)


class LingBotVideoDiagonalGaussianDistribution:
    def __init__(self, parameters):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        noise = torch.randn(
            self.mean.shape,
            dtype=self.mean.dtype,
            device=self.mean.device,
            generator=generator,
        )
        return self.mean + self.std * noise


class LingBotVideoAutoencoderKLOutput:
    def __init__(self, latent_dist):
        self.latent_dist = latent_dist


class LingBotVideoAutoencoderKLWan(LingBotVideoAutoencoderKLWanDecodeOnly):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = LingBotVideoWanEncoder3d(
            in_channels=config["in_channels"],
            dim=config["base_dim"],
            z_dim=self.z_dim * 2,
            dim_mult=config["dim_mult"],
            num_res_blocks=config["num_res_blocks"],
            attn_scales=config.get("attn_scales", []),
            temperal_downsample=config["temperal_downsample"],
            dropout=config.get("dropout", 0.0),
            non_linearity=config.get("non_linearity", "silu"),
        )
        self.quant_conv = WanCausalConv3d(self.z_dim * 2, self.z_dim * 2, 1)
        self._enc_conv_num = sum(isinstance(m, WanCausalConv3d) for m in self.encoder.modules())
        self.clear_cache()

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        if hasattr(self, "encoder"):
            self._enc_conv_idx = [0]
            self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x):
        _, _, num_frame, _, _ = x.shape
        self.clear_cache()
        if self.patch_size is not None:
            x = patchify(x, self.patch_size)

        out = None
        iter_count = 1 + (num_frame - 1) // 4
        for i in range(iter_count):
            self._enc_conv_idx = [0]
            if i == 0:
                encoded = self.encoder(x[:, :, :1], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                encoded = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            out = encoded if out is None else torch.cat([out, encoded], dim=2)

        encoded = self.quant_conv(out)
        self.clear_cache()
        return encoded

    def encode(self, x, return_dict=True):
        posterior = LingBotVideoDiagonalGaussianDistribution(self._encode(x))
        if not return_dict:
            return (posterior,)
        return LingBotVideoAutoencoderKLOutput(latent_dist=posterior)


class LingBotVideoWanVAE:
    def __init__(self, config):
        self.config = config
        self.cpu_offload = config.get("vae_cpu_offload", config.get("cpu_offload", False))
        self.device = torch.device("cpu") if self.cpu_offload else torch.device(AI_DEVICE)
        self.dtype = GET_DTYPE()
        self.load()

    def load(self):
        vae_path = self.config.get("vae_path", os.path.join(self.config["model_path"], "vae"))
        with open(os.path.join(vae_path, "config.json"), "r") as f:
            self.vae_config = json.load(f)
        self.vae_config.setdefault("in_channels", 3)
        self.vae_config.setdefault("out_channels", 3)
        self.vae_config.setdefault("num_res_blocks", 2)
        self.load_encoder = self.config.get("task") == "i2v" or self.config.get("lingbot_video_load_vae_encoder", False)
        model_cls = LingBotVideoAutoencoderKLWan if self.load_encoder else LingBotVideoAutoencoderKLWanDecodeOnly
        self.model = model_cls(self.vae_config).to(self.device).to(self.dtype)

        state = {}
        weight_path = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
        prefixes = ("decoder.", "post_quant_conv.")
        if self.load_encoder:
            prefixes = prefixes + ("encoder.", "quant_conv.")
        with safe_open(weight_path, framework="pt", device=str(self.device)) as f:
            for key in f.keys():
                if not key.startswith(prefixes):
                    continue
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=self.dtype)
                state[key] = tensor
        self.model.load_state_dict(state, strict=False)
        self.model.eval().requires_grad_(False)
        self.latents_mean = torch.tensor(self.vae_config["latents_mean"], dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(self.vae_config["latents_std"], dtype=torch.float32).view(1, -1, 1, 1, 1)

    @staticmethod
    def _to_pil_frames(video):
        frames = video[0].permute(1, 0, 2, 3)
        frames = (frames + 1.0) / 2.0
        frames = frames.detach().cpu().float().clamp_(0, 1).permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((frame * 255.0).clip(0, 255).astype(np.uint8)) for frame in frames]

    @staticmethod
    def _to_frame_tensor(video):
        frames = video[0].permute(1, 2, 3, 0)
        return ((frames + 1.0) / 2.0).detach().cpu().float().clamp_(0, 1)

    def _dit_latent_to_vae(self, latents):
        mean = self.latents_mean.to(latents.device)
        std = self.latents_std.to(latents.device)
        return latents.float() * std + mean

    def _vae_latent_to_dit(self, latents):
        mean = self.latents_mean.to(latents.device)
        std = self.latents_std.to(latents.device)
        return (latents.float() - mean) / std

    @torch.no_grad()
    def encode_image_latent(self, pixel, generator=None):
        if not self.load_encoder:
            raise RuntimeError("LingBotVideoWanVAE was loaded without encoder. Use task=i2v or set lingbot_video_load_vae_encoder=True.")
        if self.cpu_offload:
            self.model.to(AI_DEVICE)
        device = next(self.model.parameters()).device
        pixel = pixel.to(device=device, dtype=torch.float32)
        norm_pixel = (pixel - 0.5) / 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = self.model.encode(norm_pixel)
        latents = encoded.latent_dist.sample(generator)
        latents = self._vae_latent_to_dit(latents).to(dtype=torch.float32)
        if self.cpu_offload:
            latents = latents.to(AI_DEVICE)
            self.model.to("cpu")
            if hasattr(torch_device_module, "empty_cache"):
                torch_device_module.empty_cache()
            gc.collect()
        return latents

    @torch.no_grad()
    def decode(self, latents, input_info=None):
        if self.cpu_offload:
            self.model.to(AI_DEVICE)
        latents = latents.to(device=next(self.model.parameters()).device, dtype=torch.float32)
        z_raw = self._dit_latent_to_vae(latents).to(dtype=self.dtype)
        if z_raw.ndim == 5:
            z_raw = z_raw.contiguous(memory_format=torch.channels_last_3d)
        decoded = self.model.decode(z_raw).sample.float().clamp_(-1, 1)
        if self.cpu_offload:
            decoded = decoded.cpu()
            self.model.to("cpu")
            if hasattr(torch_device_module, "empty_cache"):
                torch_device_module.empty_cache()
            gc.collect()
        if input_info is not None and input_info.return_result_tensor:
            if self.config.get("task") in ("t2v", "i2v"):
                return self._to_frame_tensor(decoded)
            return (decoded + 1.0) / 2.0
        if self.config.get("task") in ("t2v", "i2v"):
            return self._to_frame_tensor(decoded)
        return self._to_pil_frames(decoded)
