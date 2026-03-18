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
from loguru import logger
from torch import Tensor

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.seedvr.scheduler import SeedVRScheduler
from lightx2v.models.video_encoders.hf.seedvr import attn_video_vae_v3_s8_c16_t4_inflation_sd3_init
from lightx2v.models.video_encoders.hf.seedvr.color_fix import wavelet_reconstruction
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("seedvr2")
class SeedVRRunner(DefaultRunner):
    """Runner for SeedVR video super-resolution model."""

    def __init__(self, config):
        super().__init__(config)
        self.run_input_encoder = self._run_input_encoder_local_sr
        self.text_encoder_output = None

        model_path_base = config.get("model_path", "ByteDance-Seed/SeedVR2-3B")
        self.model_path = os.path.join(model_path_base, "seedvr2_ema_3b.pth")
        self.vae_path = os.path.join(model_path_base, "ema_vae.pth")
        self.pos_emb_path = os.path.join(model_path_base, "pos_emb.pt")
        self.neg_emb_path = os.path.join(model_path_base, "neg_emb.pt")

    def _build_video_transform(self, img):
        from torchvision.transforms import Normalize

        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.divisible_crop import DivisibleCrop
        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.na_resize import NaResize
        from lightx2v.models.video_encoders.hf.seedvr.data.video.transforms.rearrange import Rearrange

        target_height = self.config.get("target_height", 720)
        target_width = self.config.get("target_width", 1280)
        resolution = min((self.ori_h * self.ori_w) ** 0.5 * self.input_info.sr_ratio, (target_height * target_width) ** 0.5)

        img = NaResize(
            resolution=resolution,
            mode="area",
            downsample_only=False,
        )(img)

        img.clamp_(0.0, 1.0)

        img = DivisibleCrop((16, 16))(img)

        Normalize(0.5, 0.5, inplace=True)(img)

        img = Rearrange("t c h w -> c t h w")(img)

        return img

    def _get_sr_segment_params(self):
        seg_len = int(self.config.get("sr_segment_length", 81))
        overlap = int(self.config.get("sr_overlap", 0))
        if seg_len <= 0:
            return None, 0
        if overlap >= seg_len:
            overlap = max(seg_len - 1, 0)
            logger.warning(f"[SeedVRRunner] sr_overlap >= sr_segment_length, clamp to {overlap}")
        return seg_len, overlap

    def _set_output_fps(self, fps):
        if fps is None:
            return
        try:
            fps = float(fps)
        except Exception:
            return
        if fps <= 0:
            return
        with self.config.temporarily_unlocked():
            self.config["fps"] = fps

    def _probe_video(self, video_path):
        from torchvision.io import read_video_timestamps

        try:
            pts, fps = read_video_timestamps(video_path, pts_unit="sec")
        except Exception as e:
            logger.warning(f"[SeedVRRunner] read_video_timestamps failed: {e}")
            pts, fps = [], None

        total_frames = len(pts) if pts is not None else 0
        fps_for_seek = fps
        if fps_for_seek is None or fps_for_seek == 0:
            fps_for_seek = float(self.config.get("fps", 16))
        if fps is not None and fps != 0:
            self._set_output_fps(fps)
        return total_frames, fps_for_seek, pts

    def _build_sr_segments(self, total_frames, seg_len, overlap):
        if total_frames <= seg_len:
            return [(0, total_frames)]
        step = max(seg_len - overlap, 1)
        segments = []
        start = 0
        while start < total_frames:
            end = min(start + seg_len, total_frames)
            segments.append((start, end))
            if end >= total_frames:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return segments

    def _read_video_segment(self, video_path, start_idx, end_idx):
        from torchvision.io import read_video

        total_len = max(end_idx - start_idx, 0)
        if total_len == 0:
            return torch.empty(0, 3, 0, 0)

        start_pts = None
        end_pts = None
        if getattr(self, "_sr_pts", None):
            start_pts = float(self._sr_pts[start_idx])
            end_pts = float(self._sr_pts[end_idx - 1]) + 1.0 / max(self._sr_fps, 1.0)
        else:
            start_pts = float(start_idx) / max(self._sr_fps, 1.0)
            end_pts = float(end_idx - 1) / max(self._sr_fps, 1.0) + 1.0 / max(self._sr_fps, 1.0)

        video, _, info = read_video(
            video_path,
            start_pts=start_pts,
            end_pts=end_pts,
            pts_unit="sec",
            output_format="TCHW",
        )
        if info is not None and self._sr_fps in [None, 0]:
            self._sr_fps = info.get("video_fps", self._sr_fps)
            self._set_output_fps(self._sr_fps)

        if video.shape[0] > total_len:
            video = video[:total_len]
        return video

    def _run_sr_single_segment(self):
        cached_input_info = self.input_info
        self.init_run()
        if self.config.get("compile", False) and hasattr(self.model, "comple"):
            self.model.select_graph_for_compile(self.input_info)

        segment_idx = 0
        self.init_run_segment(segment_idx)
        latents = self.run_segment(segment_idx)
        self.gen_video = self.run_vae_decoder(latents)
        self.end_run_segment(segment_idx)
        raw_video = self.gen_video_final
        self.end_run()
        self.input_info = cached_input_info
        return raw_video

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
        if self.text_encoder_output is not None:
            return self.text_encoder_output
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
        self.text_encoder_output = text_encoder_output

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

            if getattr(self, "_sr_segment", None) is not None:
                start_idx, end_idx = self._sr_segment
                video = self._read_video_segment(video_path, start_idx, end_idx)
            else:
                video, _, info = read_video(video_path, output_format="TCHW")
                if info is not None:
                    self._set_output_fps(info.get("video_fps", None))
            if video.numel() == 0:
                raise ValueError(f"Failed to read video from {video_path}")

            img = video.to(GET_DTYPE()).div_(255.0).to(self.init_device)

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
        img = self._build_video_transform(img)
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
                latent_blur=self.scheduler._add_noise(latent_blur, aug_noise),
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

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info

        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()

        video_path = getattr(self.input_info, "video_path", "")
        seg_len, overlap = self._get_sr_segment_params()
        if not video_path or seg_len is None:
            self.inputs = self.run_input_encoder()
            return self.run_main()

        total_frames, fps, pts = self._probe_video(video_path)
        if total_frames <= seg_len or total_frames == 0:
            self.inputs = self.run_input_encoder()
            return self.run_main()

        self._sr_fps = fps
        self._sr_pts = pts
        segments = self._build_sr_segments(total_frames, seg_len, overlap)
        logger.info(f"[SeedVRRunner] SR segmenting: total_frames={total_frames}, seg_len={seg_len}, overlap={overlap}, segments={len(segments)}")

        raw_segments = []
        original_save_path = self.input_info.save_result_path
        original_return_tensor = self.input_info.return_result_tensor
        self.input_info.save_result_path = ""
        self.input_info.return_result_tensor = True

        for idx, (start_idx, end_idx) in enumerate(segments):
            logger.info(f"[SeedVRRunner] Processing segment {idx + 1}/{len(segments)}: frames {start_idx}:{end_idx}")
            self._sr_segment = (start_idx, end_idx)
            self.inputs = self.run_input_encoder()
            raw = self._run_sr_single_segment()
            if overlap > 0 and idx > 0 and raw is not None:
                raw = raw[:, :, overlap:, :, :]
            raw_segments.append(raw)

        self._sr_segment = None
        self.input_info.save_result_path = original_save_path
        self.input_info.return_result_tensor = original_return_tensor

        self.gen_video_final = torch.cat(raw_segments, dim=2)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final
