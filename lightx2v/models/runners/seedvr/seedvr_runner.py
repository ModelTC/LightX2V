"""
Runner for SeedVR video super-resolution model.

SeedVR is a video super-resolution model that uses:
- NaDiT (Native Resolution Diffusion Transformer)
- Video VAE for encoding/decoding
- Pre-computed text embeddings
"""

import gc
import os

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.seedvr.scheduler import SeedVRScheduler
from lightx2v.models.video_encoders.hf.seedvr import attn_video_vae_v3_s8_c16_t4_inflation_sd3_init
from lightx2v.models.video_encoders.hf.seedvr.color_fix import wavelet_reconstruction
from lightx2v.utils.envs import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("seedvr2")
class SeedVRRunner(DefaultRunner):
    """Runner for SeedVR video super-resolution model."""

    def __init__(self, config):
        super().__init__(config)

        model_path_base = config.get("model_path", "ByteDance-Seed/SeedVR2-3B")
        self.model_path = os.path.join(model_path_base, "seedvr2_ema_3b.pth")
        self.vae_path = os.path.join(model_path_base, "ema_vae.pth")
        self.pos_emb_path = os.path.join(model_path_base, "pos_emb.pt")
        self.neg_emb_path = os.path.join(model_path_base, "neg_emb.pt")

    def _build_video_transform(self):
        from torchvision.transforms import Compose, Lambda, Normalize

        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.divisible_crop import DivisibleCrop
        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.na_resize import NaResize
        from lightx2v.models.video_encoders.hf.seedvr.data.video.transforms.rearrange import Rearrange

        target_height = self.config.get("target_height", 720)
        target_width = self.config.get("target_width", 1280)
        resolution = min((self.ori_h * self.ori_w) ** 0.5 * self.input_info.sr_ratio, (target_height * target_width) ** 0.5)

        return Compose(
            [
                NaResize(
                    resolution=resolution,
                    mode="area",
                    downsample_only=False,
                ),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ]
        )

    def _cut_videos(self, videos, sp_size):
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

    def _timestep_transform(self, timesteps: torch.Tensor, latents_shapes: torch.Tensor):
        transform = self.config.get("diffusion", {}).get("timesteps", {}).get("transform", True)
        if not transform:
            return timesteps

        vt = 4
        vs = 8
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        schedule_T = 1000.0
        if hasattr(self, "scheduler") and getattr(self.scheduler, "schedule", None) is not None:
            schedule_T = float(self.scheduler.schedule.T)
        timesteps = timesteps / schedule_T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * schedule_T
        return timesteps

    def _add_noise(self, x: torch.Tensor, aug_noise: torch.Tensor, cond_noise_scale: float = 0.0):
        t = torch.tensor([1000.0 * cond_noise_scale], device=x.device, dtype=x.dtype)
        shape = torch.tensor(x.shape[1:], device=x.device)[None]
        t = self._timestep_transform(t, shape)
        if hasattr(self, "scheduler") and getattr(self.scheduler, "schedule", None) is not None:
            return self.scheduler.schedule.forward(x, aug_noise, t)
        return (1 - (t / 1000.0)) * x + (t / 1000.0) * aug_noise

    def init_scheduler(self):
        """Initialize the scheduler for SeedVR."""
        self.scheduler = SeedVRScheduler(self.config)

    def load_transformer(self):
        """Load the SeedVR transformer model."""
        from lightx2v.models.networks.seedvr import SeedVRNaDiTModel

        model = SeedVRNaDiTModel(
            model_path=self.model_path,
            config=self.config,
            device=self.init_device,
        )
        return model

    def load_text_encoder(self):
        """Load text encoder for SeedVR.

        SeedVR uses pre-computed text embeddings (pos_emb.pt, neg_emb.pt).
        We load them from disk and cache them.
        """
        # For SeedVR, text embeddings are pre-computed
        # Load them during run_text_encoder
        return []

    def load_image_encoder(self):
        """SeedVR SR task doesn't use separate image encoder.

        The input video/image is encoded by VAE directly.
        """
        return None

    def load_vae_encoder(self):
        vae = attn_video_vae_v3_s8_c16_t4_inflation_sd3_init(
            device=AI_DEVICE,
            dtype=GET_DTYPE(),
            weights_path=self.vae_path,
            weights_map_location="cpu",
            weights_mmap=True,
            strict=False,
        )
        vae.requires_grad_(False).eval()
        vae.set_causal_slicing(split_size=4, memory_device="same")
        vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
        return vae

    def load_vae_decoder(self):
        pass

    def load_vae(self):
        """Load VAE encoder and decoder for SeedVR.

        SeedVR's VAE is a single model that can both encode and decode,
        so we return the same instance for both.
        """
        vae_encoder = self.load_vae_encoder()
        # Use the same VAE for encoding and decoding
        vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    def run_vae_decoder(self, latents):
        samples = self.vae_decoder.vae_decode(latents)
        sample = [(rearrange(video[:, None], "c t h w -> t c h w") if video.ndim == 3 else rearrange(video, "c t h w -> t c h w")) for video in samples][0]
        if self._ori_length < sample.shape[0]:
            sample = sample[: self._ori_length]

        # color fix
        input = rearrange(self._input[:, None], "c t h w -> t c h w") if self._input.ndim == 3 else rearrange(self._input, "c t h w -> t c h w")
        sample = wavelet_reconstruction(sample.to("cpu"), input[: sample.size(0)].to("cpu"))
        sample = rearrange(sample[:, None], "t c h w -> c t h w") if sample.ndim == 3 else rearrange(sample, "t c h w -> c t h w")
        sample = sample[None, :]

        return sample

    def run_text_encoder(self, input_info):
        """Run text encoder for SeedVR.

        SeedVR uses pre-computed text embeddings.
        Load them from disk and return as context.
        """
        # Load positive embeddings
        if self.pos_emb_path:
            try:
                pos_emb = torch.load(self.pos_emb_path, map_location="cpu")
                pos_emb = pos_emb.to(self.init_device)
            except Exception as e:
                print(f"[SeedVRRunner] Failed to load pos_emb: {e}")
                pos_emb = None
        else:
            pos_emb = None

        # Load negative embeddings
        if self.neg_emb_path:
            try:
                neg_emb = torch.load(self.neg_emb_path, map_location="cpu")
                neg_emb = neg_emb.to(self.init_device)
            except Exception as e:
                print(f"[SeedVRRunner] Failed to load neg_emb: {e}")
                neg_emb = None
        else:
            neg_emb = None

        # Return text encoder output
        text_encoder_output = {
            "texts_pos": [pos_emb],
            "texts_neg": [neg_emb],
        }

        return text_encoder_output

    def run_image_encoder(self, img):
        """SeedVR SR task doesn't use separate image encoder."""
        return None

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        """Get latent shape for SeedVR.

        Args:
            latent_h: Latent height
            latent_w: Latent width

        Returns:
            [num_channels_latents, latent_h, latent_w]
        """
        latent_shape = [
            self.num_channels_latents,
            latent_h,
            latent_w,
        ]
        return latent_shape

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    def _run_input_encoder_local_sr(self):
        """Run input encoder for SR task.

        Args:
            input_info: Input information

        Returns:
            Dictionary with encoder outputs
        """
        # Read input video/image
        # Check video_path first (priority for SR task)
        if "video_path" in self.input_info.__dataclass_fields__ and self.input_info.video_path:
            video_path = self.input_info.video_path
            from torchvision.io import read_video

            video, _, _ = read_video(video_path, output_format="TCHW")
            if video.numel() == 0:
                raise ValueError(f"Failed to read video from {video_path}")
            img = (video / 255.0).to(self.init_device)
        elif "image_path" in self.input_info.__dataclass_fields__ and self.input_info.image_path:
            from PIL import Image

            img_path = self.input_info.image_path
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0)  # [1, C, H, W]
            img = img.to(self.init_device)
        else:
            raise ValueError("SR task requires image_path or video_path")

        # Apply SeedVR-style video transforms
        _, _, ori_h, ori_w = img.shape
        self.ori_h = ori_h
        self.ori_w = ori_w
        video_transform = self._build_video_transform()
        img = video_transform(img)
        self._input = img
        self._ori_length = img.shape[1]

        # Apply cut_videos and add_noise similar to original logic
        sp_size = 1
        img = self._cut_videos(img, sp_size)
        cond_latents = [img]
        cond_latents = self.vae_encoder.vae_encode(cond_latents)
        text_encoder_output = self.run_text_encoder(self.input_info)

        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]
        conditions = [
            self.get_condition(
                noise,
                task="sr",
                latent_blur=self._add_noise(latent_blur, aug_noise),
            )
            for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
        ]

        # # Get latent shape
        # B, C, T, H, W = cond_latent.shape
        # latent_shape = [B, C, T, H, W]
        # self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info

        torch.cuda.empty_cache()
        gc.collect()

        first_latent = cond_latents[0]
        latent_shape = [1, first_latent.shape[-1], first_latent.shape[0], first_latent.shape[1], first_latent.shape[2]]

        return {
            "x": cond_latents[0],
            "conditions": conditions,
            "noises": noises,
            "vae_encoder_out": cond_latents[0],
            "image_encoder_output": None,
            "text_encoder_output": text_encoder_output,
            "latent_shape": latent_shape,
        }

    def __call__(self, *args, **kwargs):
        """Main inference entry point.

        This handles the SR task specifically.
        """
        # Set up SR task input encoder
        self.run_input_encoder = self._run_input_encoder_local_sr

        # Call parent __call__
        return super().__call__(*args, **kwargs)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "text_encoders"):
            del self.text_encoders
        if hasattr(self, "image_encoder"):
            del self.image_encoder
        if hasattr(self, "vae_encoder"):
            del self.vae_encoder
        if hasattr(self, "vae_decoder"):
            del self.vae_decoder

        if hasattr(AI_DEVICE, "__class__"):
            torch_device_module = getattr(torch, AI_DEVICE)
            torch_device_module.empty_cache()
        gc.collect()
