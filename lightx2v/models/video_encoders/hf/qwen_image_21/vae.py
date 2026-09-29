"""Single-frame RGBA VAE, with the released Qwen-Image-2.1 weight names.

The residual spatial shortcuts are shared with native Wan 2.2. Convolutions
are the image specialization (2D checkpoint kernels); temporal resampler
weights are loaded for compatibility but are unused for the first frame.
"""

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

from lightx2v.models.video_encoders.hf.qwen_image_21.fp8_conv import FP8_DECODER_PROFILE, FP8_DECODER_PROFILE_KEY, FP8Conv2d, is_fp8_decoder_conv
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import AvgDown3D, DupUp3D
from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class ImageConv(FP8Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x):
        return super().forward(F.pad(x.squeeze(2), self.image_padding)).unsqueeze(2)


class ImageRMSNorm(nn.Module):
    def __init__(self, channels, images=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((channels,) + ((1, 1) if images else (1, 1, 1))))
        self.scale = channels**0.5

    def forward(self, x):
        return F.normalize(x.float(), dim=1).to(x.dtype) * self.scale * self.gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm1 = ImageRMSNorm(in_dim)
        self.conv1 = ImageConv(in_dim, out_dim, 3, padding=1)
        self.norm2 = ImageRMSNorm(out_dim)
        self.conv2 = ImageConv(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = ImageConv(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return self.conv2(F.silu(self.norm2(h))) + self.conv_shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = ImageRMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, _, h, w = x.shape
        value = x.permute(0, 2, 1, 3, 4).reshape(b, c, h, w)
        qkv = self.to_qkv(self.norm(value)).reshape(b, 1, 3 * c, -1).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, -1)
        value = F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(value).unsqueeze(2)


class MidBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resnets = nn.ModuleList([ResidualBlock(dim, dim), ResidualBlock(dim, dim)])
        self.attentions = nn.ModuleList([AttentionBlock(dim)])

    def forward(self, x):
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class Upsample(nn.Module):
    def forward(self, x):
        return F.interpolate(x.float(), scale_factor=2, mode="nearest-exact").to(x.dtype)


class Resample(nn.Module):
    def __init__(self, dim, up, temporal):
        super().__init__()
        if up:
            self.resample = nn.Sequential(Upsample(), FP8Conv2d(dim, dim, 3, padding=1))
        else:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), FP8Conv2d(dim, dim, 3, stride=2))
        if temporal:
            self.time_conv = ImageConv(dim, dim * 2 if up else dim, 1)

    def forward(self, x):
        b, c, _, h, w = x.shape
        # Keep the reference's frame folding, including singleton-axis strides:
        # these affect memory-format selection in BF16 convolution kernels.
        x = self.resample(x.permute(0, 2, 1, 3, 4).reshape(b, c, h, w))
        return x.view(b, 1, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)


class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, count, down, temporal):
        super().__init__()
        self.resnets = nn.ModuleList([ResidualBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(count)])
        self.downsampler = Resample(out_dim, False, temporal) if down else None
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, 2 if temporal else 1, 2 if down else 1)

    def forward(self, x):
        h = x
        for block in self.resnets:
            h = block(h)
        if self.downsampler is not None:
            h = self.downsampler(h)
        return h + self.avg_shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, count, up, temporal):
        super().__init__()
        self.resnets = nn.ModuleList([ResidualBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(count + 1)])
        self.upsampler = Resample(out_dim, True, temporal) if up else None
        self.avg_shortcut = DupUp3D(in_dim, out_dim, 2 if temporal else 1, 2) if up else None

    def forward(self, x):
        h = x
        for block in self.resnets:
            h = block(h)
        if self.upsampler is not None:
            h = self.upsampler(h)
            h = h + self.avg_shortcut(x, first_chunk=True)
        return h


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = [config["base_dim"] * m for m in [1] + config["dim_mult"]]
        self.conv_in = ImageConv(config["in_channels"], dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            DownBlock(a, b, config["num_res_blocks"], i < len(dims) - 2, config["temperal_downsample"][i] if i < len(dims) - 2 else False) for i, (a, b) in enumerate(zip(dims[:-1], dims[1:]))
        )
        self.mid_block = MidBlock(dims[-1])
        self.norm_out = ImageRMSNorm(dims[-1])
        self.conv_out = ImageConv(dims[-1], config["z_dim"] * 2, 3, padding=1)

    def spatial_halo(self):
        # Conv-in and residual convolutions pad symmetrically; downsamplers
        # pad only on the right/bottom, so their receptive field is asymmetric.
        left = right = 1
        stride = 1
        for block in self.down_blocks:
            left += 2 * len(block.resnets) * stride
            right += 2 * len(block.resnets) * stride
            if block.downsampler is not None:
                right += 2 * stride
                stride *= 2
        return (max(left, right) + stride - 1) // stride * stride

    def forward_down(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return x

    def forward_mid(self, x):
        return self.conv_out(F.silu(self.norm_out(self.mid_block(x))))

    def forward(self, x):
        return self.forward_mid(self.forward_down(x))


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        mult = config["dim_mult"]
        dims = [config["decoder_base_dim"] * m for m in [mult[-1]] + mult[::-1]]
        temporal = config["temperal_downsample"][::-1]
        self.conv_in = ImageConv(config["z_dim"], dims[0], 3, padding=1)
        self.mid_block = MidBlock(dims[0])
        self.up_blocks = nn.ModuleList(UpBlock(a, b, config["num_res_blocks"], i < len(dims) - 2, temporal[i] if i < len(dims) - 2 else False) for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])))
        self.norm_out = ImageRMSNorm(dims[-1])
        self.conv_out = ImageConv(dims[-1], config["out_channels"], 3, padding=1)

    def forward_up(self, x):
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    def forward_mid(self, x):
        return self.mid_block(self.conv_in(x))

    def forward(self, x):
        return self.forward_up(self.forward_mid(x))


class QwenImage21VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        path = Path(config["model_path"]) / "vae"
        self.vae_encode_parallel = config.get("vae_encode_parallel", False)
        self.vae_decode_parallel = config.get("vae_decode_parallel", False)
        self.vae_decode_parallel_mode = config.get("vae_decode_parallel_mode", "full")
        if self.vae_decode_parallel_mode not in ("post_mid", "full"):
            raise ValueError(f"vae_decode_parallel_mode must be 'post_mid' or 'full', got {self.vae_decode_parallel_mode!r}")
        self.config = json.loads((path / "config.json").read_text())
        cfg = self.config
        if not cfg["is_residual"] or cfg.get("patch_size") is not None or cfg["attn_scales"]:
            raise ValueError("Expected the released residual, unpatched Qwen-Image-2.1 image VAE")
        self.scale_factor = config["vae_scale_factor"]
        with torch.device("meta"):
            self.encoder = Encoder(cfg)
            self.decoder = Decoder(cfg)
            self.quant_conv = ImageConv(cfg["z_dim"] * 2, cfg["z_dim"] * 2, 1)
            self.post_quant_conv = ImageConv(cfg["z_dim"], cfg["z_dim"], 1)
        conv_mode = config.get("vae_decoder_conv_mode", "torch")
        if conv_mode not in ("torch", "cutlass_fp8_f16_accum"):
            raise ValueError(f"Unsupported vae_decoder_conv_mode: {conv_mode!r}")
        quantized_ckpt = config.get("vae_quantized_ckpt")
        if (conv_mode != "torch") != bool(quantized_ckpt):
            raise ValueError("FP8 VAE requires both vae_decoder_conv_mode='cutlass_fp8_f16_accum' and vae_quantized_ckpt")
        if quantized_ckpt:
            state = self._load_fp8_state(quantized_ckpt)
        else:
            state = {}
            for shard in sorted(path.glob("*.safetensors")):
                state.update(load_file(shard))
        self.load_state_dict(state, strict=True, assign=True)
        self.register_buffer("latents_mean", torch.tensor(cfg["latents_mean"]).to(GET_DTYPE()).reshape(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(cfg["latents_std"]).to(GET_DTYPE()).reshape(1, -1, 1, 1, 1), persistent=False)
        # FP8 weights and FP32 scales must retain their checkpoint dtypes.
        if quantized_ckpt:
            self.to(device=AI_DEVICE)
        else:
            self.to(dtype=GET_DTYPE(), device=AI_DEVICE)
        self.eval().requires_grad_(False)
        if config.get("vae_use_compile", False):
            logger.info("[Compile] Using torch.compile for Qwen-Image-2.1 VAE")
            # These boundaries are shared by serial and spatial-parallel paths;
            # collectives stay outside the compiled compute regions.
            self.encoder.forward_down = torch.compile(self.encoder.forward_down, dynamic=None)
            self.encoder.forward_mid = torch.compile(self.encoder.forward_mid, dynamic=None)
            self.decoder.forward_mid = torch.compile(self.decoder.forward_mid, dynamic=None)
            self.decoder.forward_up = torch.compile(self.decoder.forward_up, dynamic=None)

    def _load_fp8_state(self, checkpoint_path):
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            if (checkpoint.metadata() or {}).get(FP8_DECODER_PROFILE_KEY) != FP8_DECODER_PROFILE:
                raise ValueError("VAE checkpoint does not have the mixed FP8-F16 qmax21 decoder profile")
            state = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        fp8_weights = set()
        scales = set()
        for name, module in self.named_modules():
            if not isinstance(module, FP8Conv2d) or not is_fp8_decoder_conv(name, module.weight.shape):
                continue
            weight_name, scale_name = f"{name}.weight", f"{name}.weight_scale"
            weight, scale = state[weight_name], state[scale_name]
            if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
                raise ValueError(f"Invalid FP8 VAE weight/scale: {name}")
            state[weight_name] = weight.contiguous(memory_format=torch.channels_last)
            module.register_buffer("weight_scale", torch.empty((), device="meta", dtype=torch.float32))
            fp8_weights.add(weight_name)
            scales.add(scale_name)
        if len(fp8_weights) != 32:
            raise ValueError("The mixed FP8 decoder profile requires the released Qwen-Image-2.1 VAE architecture")
        for key, tensor in state.items():
            if key not in fp8_weights | scales:
                if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                    raise ValueError(f"Expected an unquantized VAE tensor: {key}")
                state[key] = tensor.to(GET_DTYPE())
        return state

    @torch.inference_mode()
    def encode(self, image):
        image = image.to(device=AI_DEVICE, dtype=GET_DTYPE())
        use_parallel = self.vae_encode_parallel and dist.is_initialized() and dist.get_world_size() > 1
        moments = self._encode_dist(image) if use_parallel else self.encoder(image)
        mean = self.quant_conv(moments).chunk(2, dim=1)[0]
        return ((mean - self.latents_mean) / self.latents_std).flatten(2).transpose(1, 2)

    def _encode_dist(self, image):
        """Shard local convolutions, then restore the full global attention input."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ratio = self.scale_factor
        height, width = image.shape[-2:]
        if height % ratio or width % ratio:
            raise ValueError(f"Parallel VAE encode requires image dimensions divisible by {ratio}")
        grid_h, grid_w = self._get_2d_grid(height // ratio, width // ratio, world_size)
        row, column = divmod(rank, grid_w)
        chunk_h, chunk_w = height // grid_h, width // grid_w
        halo = self.encoder.spatial_halo()
        top, left = row * chunk_h, column * chunk_w
        begin_h, begin_w = max(0, top - halo), max(0, left - halo)
        end_h, end_w = min(height, top + chunk_h + halo), min(width, left + chunk_w + halo)
        # Preserve the input's NHWC strides and the ordinary convolution path.
        features = self.encoder.forward_down(image[..., begin_h:end_h, begin_w:end_w])
        offset_h, offset_w = (top - begin_h) // ratio, (left - begin_w) // ratio
        piece = features[..., offset_h : offset_h + chunk_h // ratio, offset_w : offset_w + chunk_w // ratio].contiguous()
        pieces = [torch.empty_like(piece) for _ in range(world_size)]
        dist.all_gather(pieces, piece)
        rows = [torch.cat(pieces[i * grid_w : (i + 1) * grid_w], dim=-1) for i in range(grid_h)]
        full = torch.cat(rows, dim=-2)
        if features.is_contiguous(memory_format=torch.channels_last_3d):
            full = full.contiguous(memory_format=torch.channels_last_3d)
        return self.encoder.forward_mid(full)

    @staticmethod
    def _get_2d_grid(total_h, total_w, world_size):
        best_h = best_w = None
        min_aspect_diff = float("inf")
        for grid_h in range(1, world_size + 1):
            if world_size % grid_h == 0:
                grid_w = world_size // grid_h
                if total_h % grid_h == 0 and total_w % grid_w == 0:
                    aspect_diff = abs(total_h / grid_h - total_w / grid_w)
                    if aspect_diff < min_aspect_diff:
                        min_aspect_diff = aspect_diff
                        best_h, best_w = grid_h, grid_w
        if best_h is None:
            raise ValueError(f"Cannot split latent grid {total_h}x{total_w} evenly across {world_size} ranks")
        return best_h, best_w

    def _decode_dist(self, z):
        """Decode overlapping spatial shards using the configured cut point."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        total_h, total_w = z.shape[-2:]
        grid_h, grid_w = self._get_2d_grid(total_h, total_w, world_size)
        rank_h, rank_w = divmod(rank, grid_w)
        chunk_h, chunk_w = total_h // grid_h, total_w // grid_w
        # A two-latent halo is the measured accuracy/performance tradeoff. It does
        # not cover the full upsampling receptive field, so distributed decode is
        # numerically close to, but not bitwise identical to, serial decode.
        padding = 2

        if rank_h == 0:
            h_start, h_end = 0, chunk_h + 2 * padding
        elif rank_h == grid_h - 1:
            h_start, h_end = total_h - chunk_h - 2 * padding, total_h
        else:
            h_start, h_end = rank_h * chunk_h - padding, (rank_h + 1) * chunk_h + padding
        if rank_w == 0:
            w_start, w_end = 0, chunk_w + 2 * padding
        elif rank_w == grid_w - 1:
            w_start, w_end = total_w - chunk_w - 2 * padding, total_w
        else:
            w_start, w_end = rank_w * chunk_w - padding, (rank_w + 1) * chunk_w + padding

        # Keep halo coordinates inside the image, including when the shard
        # is smaller than its halo. Crop relative to the actual shard origin.
        h_start, h_end = max(0, h_start), min(total_h, h_end)
        w_start, w_end = max(0, w_start), min(total_w, w_end)

        if self.vae_decode_parallel_mode == "post_mid":
            # Preserve the decoder's global low-resolution attention exactly,
            # then parallelize the substantially more expensive upsampling path.
            z = self.decoder.forward_mid(z)
            shard = z[:, :, :, h_start:h_end, w_start:w_end].contiguous()
            decoded = self.decoder.forward_up(shard)
        else:
            # Maximize the parallel fraction by sharding the complete decoder.
            # The mid-block attention is local to each overlapping shard, so this
            # mode is faster in principle but less numerically faithful.
            shard = z[:, :, :, h_start:h_end, w_start:w_end].contiguous()
            decoded = self.decoder(shard)
        ratio = self.scale_factor
        dh_start = (rank_h * chunk_h - h_start) * ratio
        dh_end = dh_start + chunk_h * ratio
        dw_start = (rank_w * chunk_w - w_start) * ratio
        dw_end = dw_start + chunk_w * ratio
        piece = decoded[:, :, :, dh_start:dh_end, dw_start:dw_end].contiguous()

        pieces = [torch.empty_like(piece) for _ in range(world_size)]
        dist.all_gather(pieces, piece)
        rows = [torch.cat(pieces[row * grid_w : (row + 1) * grid_w], dim=-1) for row in range(grid_h)]
        return torch.cat(rows, dim=-2)

    @torch.inference_mode()
    def decode(self, latents, size):
        h, w = size
        z = latents.transpose(1, 2).reshape(1, self.config["z_dim"], 1, h // self.scale_factor, w // self.scale_factor).to(GET_DTYPE())
        z = self.post_quant_conv(z * self.latents_std + self.latents_mean)
        use_parallel = self.vae_decode_parallel and dist.is_initialized() and dist.get_world_size() > 1
        decoded = self._decode_dist(z) if use_parallel else self.decoder(z)
        return decoded.clamp(-1, 1)[:, :, 0]
