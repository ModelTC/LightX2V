"""
Runner for SeedVR video super-resolution model.

SeedVR is a video super-resolution model that uses:
- NaDiT (Native Resolution Diffusion Transformer)
- Video VAE for encoding/decoding
- Pre-computed text embeddings
"""

import gc
import os
import shutil
import subprocess
import tempfile

import imageio_ffmpeg as ffmpeg
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor

from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.seedvr.scheduler import SeedVRScheduler
from lightx2v.models.video_encoders.hf.seedvr import attn_video_vae_v3_s8_c16_t4_inflation_sd3_init
from lightx2v.models.video_encoders.hf.seedvr.color_fix import wavelet_reconstruction
from lightx2v.models.video_encoders.hf.seedvr.common.distributed.advanced import set_sequence_parallel_group
from lightx2v.models.video_encoders.hf.seedvr.common.distributed.ops import set_sequence_parallel_a2a_backend
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video, save_to_video, wan_vae_to_comfy
from lightx2v.utils.video_recorder import VideoRecorder
from lightx2v_platform.base.global_var import AI_DEVICE


class SeedVRVideoRecorder(VideoRecorder):
    """High-quality local-file recorder for SeedVR super-resolution output."""

    def start_ffmpeg_process_local(self):
        crf = str(self.config_crf) if hasattr(self, "config_crf") else "16"
        preset = str(self.config_preset) if hasattr(self, "config_preset") else "medium"
        ffmpeg_cmd = [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-color_range",
            "pc",
            "-colorspace",
            "rgb",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "iec61966-2-1",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            f"tcp://127.0.0.1:{self.video_port}",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            crf,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.livestream_url,
            "-y",
            "-loglevel",
            self.ffmpeg_log_level,
        ]
        try:
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd)
            logger.info(f"SeedVR FFmpeg file encoder started with PID: {self.ffmpeg_process.pid}, preset={preset}, crf={crf}")
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        except Exception as e:
            logger.error(f"Failed to start SeedVR FFmpeg file encoder: {e}")


def _get_read_video():
    """Return ``read_video`` with a 3-level fallback chain.

    torchvision moved ``read_video`` between releases; the last-resort PyAV
    fallback handles environments where torchvision isn't installed at all.
    """
    try:
        from torchvision.io import read_video
    except ImportError:
        try:
            from torchvision.io.video import read_video
        except ImportError:
            import av

            def read_video(filename, start_pts=0, end_pts=None, pts_unit="pts", output_format="THWC"):
                container = av.open(filename)
                try:
                    if not container.streams.video:
                        raise ValueError(f"No video stream found in {filename}")
                    stream = container.streams.video[0]
                    try:
                        fps = float(stream.average_rate) if stream.average_rate else 0.0
                    except ZeroDivisionError:
                        fps = 0.0
                    frames = []
                    for frame_index, frame in enumerate(container.decode(video=0)):
                        if pts_unit == "sec":
                            frame_time = float(frame.time) if frame.time is not None else frame_index / max(fps, 1.0)
                            if frame_time < float(start_pts or 0):
                                continue
                            if end_pts is not None and frame_time >= float(end_pts):
                                break
                        elif pts_unit == "pts" and frame.pts is not None:
                            if frame.pts < int(start_pts or 0):
                                continue
                            if end_pts is not None and frame.pts >= int(end_pts):
                                break
                        img = frame.to_ndarray(format="rgb24")
                        frames.append(img)
                    if not frames:
                        raise ValueError(f"No frames decoded from {filename}")
                finally:
                    container.close()
                video = torch.from_numpy(np.stack(frames))  # T H W C
                if output_format == "TCHW":
                    video = video.permute(0, 3, 1, 2)
                return video, torch.zeros(0), {"video_fps": fps}

    return read_video


@RUNNER_REGISTER("seedvr2")
class SeedVRRunner(DefaultRunner):
    """Runner for SeedVR video super-resolution model."""

    def __init__(self, config):
        super().__init__(config)
        self.run_input_encoder = self._run_input_encoder_local_sr
        self.text_encoder_output = None

        self._seedvr_sp_group = None
        self._seedvr_sp_size = 1
        self._seedvr_sp_rank = 0
        if self.config.get("seq_parallel", False):
            parallel = self.config.get("parallel", {})
            if not isinstance(parallel, dict):
                raise ValueError("SeedVR sequence parallel requires a parallel configuration dictionary")
            if int(parallel.get("tensor_p_size", 1)) != 1 or int(parallel.get("cfg_p_size", 1)) != 1:
                raise ValueError("SeedVR sequence parallel currently requires tensor_p_size=1 and cfg_p_size=1")
            seq_p_attn_type = parallel.get("seq_p_attn_type", "ulysses")
            if seq_p_attn_type not in ("ulysses", "ulysses-4090"):
                raise ValueError("SeedVR sequence parallel supports seq_p_attn_type=ulysses or ulysses-4090 only")
            if not parallel.get("vae_parallel", True):
                raise ValueError("SeedVR sequence parallel requires parallel.vae_parallel=true")

            self._seedvr_sp_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
            self._seedvr_sp_size = dist.get_world_size(self._seedvr_sp_group)
            self._seedvr_sp_rank = dist.get_rank(self._seedvr_sp_group)
            heads = 24 if self.config.get("model_size") == "7b" else 20
            if heads % self._seedvr_sp_size != 0:
                raise ValueError(f"SeedVR attention heads ({heads}) must be divisible by seq_p_size ({self._seedvr_sp_size})")
            self.config.setdefault("load_from_rank0", True)
            set_sequence_parallel_group(self._seedvr_sp_group)
            set_sequence_parallel_a2a_backend("round_robin" if seq_p_attn_type == "ulysses-4090" else "torch")
            logger.info(
                f"[SeedVRRunner] sequence parallel enabled: rank={self._seedvr_sp_rank}/{self._seedvr_sp_size}, "
                f"DiT={seq_p_attn_type}, VAE=causal temporal, spatial_tiling={self.config.get('use_tiling_vae', False)}"
            )
        else:
            set_sequence_parallel_group(None)
            set_sequence_parallel_a2a_backend("torch")

        model_path_base = config.get("model_path", "ByteDance-Seed/SeedVR2-3B")
        if self.config.get("dit_quantized_ckpt", None):
            self.model_path = self.config.get("dit_quantized_ckpt")
        elif self.config.get("dit_original_ckpt", None):
            self.model_path = self.config.get("dit_original_ckpt")
        else:
            model_size = self.config.get("model_size", "3b")
            self.model_path = os.path.join(model_path_base, f"seedvr2_ema_{model_size}.pth")
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
        overlap = int(self.config.get("sr_overlap", 1))
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

    def _probe_video_torchcodec(self, video_path):
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path, device="cpu")
        metadata = decoder.metadata

        total_frames = metadata.num_frames
        if total_frames is None:
            total_frames = len(decoder)

        fps = metadata.average_fps
        if fps is None or fps <= 0:
            fps = float(self.config.get("fps", 16))
        else:
            fps = float(fps)
            self._set_output_fps(fps)

        return int(total_frames), fps, []

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

    def _read_video_segment_torchcodec(self, video_path, start_idx, end_idx):
        from torchcodec.decoders import VideoDecoder

        total_len = max(end_idx - start_idx, 0)
        if total_len == 0:
            return torch.empty(0, 3, 0, 0)

        decoder = VideoDecoder(video_path, device="cpu")
        video = decoder[start_idx:end_idx]  # [T, C, H, W], uint8

        if video.shape[0] > total_len:
            video = video[:total_len]

        return video

    def _read_video_segment(self, video_path, start_idx, end_idx):
        read_video = _get_read_video()

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
        segment_idx = 0

        self.init_run()
        self.init_run_segment(segment_idx)

        with ProfilingContext4DebugL1("Run DiT", profile_memory=True):
            latents = self.run_segment(segment_idx)

        self.gen_video = self.run_vae_decoder(latents)

        self.end_run_segment(segment_idx)
        raw_video = self.gen_video_final
        self.end_run()
        self.input_info = cached_input_info
        return raw_video

    def run_segment(self, segment_idx=0):
        """Run SeedVR diffusion steps under the single outer DiT profile."""
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            if self.video_segment_num == 1:
                self.check_stop()
            logger.debug(f"[SeedVRRunner] diffusion step {step_index + 1}/{infer_steps}")
            self.model.scheduler.step_pre(step_index=step_index)
            self.model.infer(self.inputs)
            self.model.scheduler.step_post()

            if self.progress_callback:
                current_step = segment_idx * infer_steps + step_index + 1
                total_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_steps) * 100, 100)

        if segment_idx == self.video_segment_num - 1:
            del self.inputs
        return self.model.scheduler.latents

    def run_main(self):
        raw_video = self._run_sr_single_segment()
        if self._seedvr_sp_size > 1:
            if self._seedvr_sp_rank == 0:
                self.gen_video_final = raw_video
                result = self.process_images_after_vae_decoder()
            else:
                result = {"video": None}
            dist.barrier(group=self._seedvr_sp_group)
            return result

        self.gen_video_final = raw_video
        return self.process_images_after_vae_decoder()

    def _save_sr_segment_video(self, raw_video, output_path, fps):
        video = wan_vae_to_comfy(raw_video).float().clamp(0.0, 1.0)
        save_to_video(video, output_path, fps=fps, method="ffmpeg")
        del video

    def _stream_sr_segment_video(self, raw_video, video_recorder, segment_idx, segment_count):
        if raw_video is None:
            raise RuntimeError(f"SeedVR rank 0 produced no output for segment {segment_idx + 1}.")
        video = wan_vae_to_comfy(raw_video).float().clamp_(0.0, 1.0).cpu()
        logger.info(f"[SeedVRRunner] stream save segment {segment_idx + 1}/{segment_count}: frames={video.shape[0]}, size={video.shape[2]}x{video.shape[1]}")
        video_recorder.pub_video(video)
        del video

    def _concat_sr_segment_videos(self, segment_paths, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if len(segment_paths) == 1:
            shutil.move(segment_paths[0], output_path)
            return

        concat_path = os.path.join(os.path.dirname(output_path) or ".", f".{os.path.basename(output_path)}.concat.txt")
        try:
            with open(concat_path, "w", encoding="utf-8") as f:
                for path in segment_paths:
                    escaped = os.path.abspath(path).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped}'\n")

            command = [
                ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c",
                "copy",
                output_path,
            ]
            process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {process.stderr.strip()}")
        finally:
            if os.path.exists(concat_path):
                os.remove(concat_path)

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

        logger.info(
            f"[SeedVRRunner] DiT config: model_size={self.config.get('model_size', '3b')}, "
            f"cpu_offload={self.config.get('cpu_offload', False)}, "
            f"offload_granularity={self.config.get('offload_granularity', 'block')}, "
            f"quant_scheme={self.config.get('dit_quant_scheme', 'Default')}"
        )
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
        vae_causal_slice_size = int(self.config.get("vae_causal_slice_size", 4))
        vae_memory_limit_gb = float(self.config.get("vae_memory_limit_gb", 0.5))
        vae_memory_limit = None if vae_memory_limit_gb <= 0 else vae_memory_limit_gb
        vae = attn_video_vae_v3_s8_c16_t4_inflation_sd3_init(
            device=AI_DEVICE,
            dtype=GET_DTYPE(),
            weights_path=self.vae_path,
            weights_map_location="cpu",
            weights_mmap=True,
            strict=False,
            cpu_offload=self.config.get("cpu_offload", False),
            use_tiling=self.config.get("use_tiling_vae", False),
            tile_size=int(self.config.get("vae_tile_size", 512)),
            tile_overlap=int(self.config.get("vae_tile_overlap", 64)),
            sp_gather_decode_to_rank0=self._seedvr_sp_size > 1,
        )
        vae.requires_grad_(False).eval()
        vae.set_causal_slicing(
            split_size=vae_causal_slice_size if vae_causal_slice_size > 0 else None,
            memory_device="same" if vae_causal_slice_size > 0 else None,
        )
        vae.set_memory_limit(conv_max_mem=vae_memory_limit, norm_max_mem=vae_memory_limit)
        logger.info(
            f"[SeedVRRunner] VAE config: tiling={self.config.get('use_tiling_vae', False)}, "
            f"tile={self.config.get('vae_tile_size', 512)}, overlap={self.config.get('vae_tile_overlap', 64)}, "
            f"causal_slice={vae_causal_slice_size if vae_causal_slice_size > 0 else 'off'}, "
            f"memory_limit={vae_memory_limit_gb if vae_memory_limit_gb > 0 else 'off'}GiB"
        )
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

    def _restore_target_size(self, sample):
        if self.config.get("resize_mode") == "adaptive":
            return sample
        target_height = int(self.config.get("target_height", sample.shape[-2]) or sample.shape[-2])
        target_width = int(self.config.get("target_width", sample.shape[-1]) or sample.shape[-1])
        if target_height <= 0 or target_width <= 0:
            return sample

        height, width = sample.shape[-2:]
        if (height, width) == (target_height, target_width):
            return sample

        if height >= target_height and width >= target_width:
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            logger.info(f"[SeedVRRunner] center crop SR output from {width}x{height} to {target_width}x{target_height}")
            return sample[..., top : top + target_height, left : left + target_width]

        logger.info(f"[SeedVRRunner] resize SR output from {width}x{height} to {target_width}x{target_height}")
        dtype = sample.dtype
        device = sample.device
        return F.interpolate(sample.float(), size=(target_height, target_width), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["SeedVRRunner"],
        profile_memory=True,
    )
    def run_vae_decoder(self, latents):
        try:
            samples = self.vae_decoder.vae_decode(latents)
        finally:
            cache_count, cache_bytes = self.vae_decoder.clear_causal_memory()
            if cache_count:
                logger.debug(f"[SeedVRRunner] released {cache_count} VAE decode caches ({cache_bytes / 1024**3:.3f} GiB)")

        if not samples:
            self._input = None
            return None

        sample = [(rearrange(video[:, None], "c t h w -> t c h w") if video.ndim == 3 else rearrange(video, "c t h w -> t c h w")) for video in samples][0]
        if self._ori_length < sample.shape[0]:
            sample = sample[: self._ori_length]

        color_fix = str(self.config.get("color_fix", "cpu")).lower()
        if color_fix not in ("cpu", "gpu", "off"):
            logger.warning(f"[SeedVRRunner] Unknown color_fix={color_fix}; fallback to cpu")
            color_fix = "cpu"
        if color_fix != "off":
            input = rearrange(self._input[:, None], "c t h w -> t c h w") if self._input.ndim == 3 else rearrange(self._input, "c t h w -> t c h w")
            if self._seedvr_sp_size > 1 and color_fix == "gpu" and sample.device.type == "cpu":
                chunk_size = max(1, int(self.config.get("vae_sp_color_chunk_size", 4)))
                fixed_chunks = []
                for start in range(0, sample.size(0), chunk_size):
                    end = min(start + chunk_size, sample.size(0))
                    fixed = wavelet_reconstruction(sample[start:end].to(AI_DEVICE), input[start:end].to(AI_DEVICE))
                    fixed_chunks.append(fixed.cpu())
                sample = torch.cat(fixed_chunks, dim=0)
            else:
                fix_device = torch.device("cpu") if color_fix == "cpu" else sample.device
                sample = wavelet_reconstruction(sample.to(fix_device), input[: sample.size(0)].to(fix_device))

        sample = self._restore_target_size(sample)
        sample = rearrange(sample[:, None], "t c h w -> c t h w") if sample.ndim == 3 else rearrange(sample, "t c h w -> c t h w")
        sample = sample[None, :]

        logger.debug(f"[SeedVRRunner] decoded video shape={tuple(sample.shape)}, color_fix={color_fix}")

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
                logger.warning(f"[SeedVRRunner] Failed to load pos_emb: {e}")
                pos_emb = None
        else:
            pos_emb = None

        # Load negative embeddings
        if self.neg_emb_path:
            try:
                neg_emb = torch.load(self.neg_emb_path, map_location="cpu")
                neg_emb = neg_emb.to(self.init_device)
            except Exception as e:
                logger.warning(f"[SeedVRRunner] Failed to load neg_emb: {e}")
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

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["SeedVRRunner"],
        profile_memory=True,
    )
    def run_vae_encoder(self, img):
        try:
            return self.vae_encoder.vae_encode([img])
        finally:
            cache_count, cache_bytes = self.vae_encoder.clear_causal_memory()
            if cache_count:
                logger.debug(f"[SeedVRRunner] released {cache_count} VAE encode caches ({cache_bytes / 1024**3:.3f} GiB)")

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
        """Prepare the input video, VAE latents and diffusion condition."""
        if "video_path" in self.input_info.__dataclass_fields__ and self.input_info.video_path:
            video_path = self.input_info.video_path

            if getattr(self, "_sr_segment", None) is not None:
                start_idx, end_idx = self._sr_segment
                if getattr(self, "_sr_video_backend", None) == "torchcodec":
                    video = self._read_video_segment_torchcodec(video_path, start_idx, end_idx)
                else:
                    try:
                        video = self._read_video_segment(video_path, start_idx, end_idx)
                    except Exception as e:
                        logger.warning(f"[SeedVRRunner] torchvision segment decode failed, switching to torchcodec: {e}")
                        self._sr_video_backend = "torchcodec"
                        video = self._read_video_segment_torchcodec(video_path, start_idx, end_idx)
                logger.debug(f"[SeedVRRunner] decoded segment frames={start_idx}:{end_idx}, actual_frames={video.shape[0]}, backend={getattr(self, '_sr_video_backend', 'torchvision')}")
            else:
                read_video = _get_read_video()
                video, _, info = read_video(video_path, output_format="TCHW")
                if info is not None:
                    self._set_output_fps(info.get("video_fps", None))
            if video.numel() == 0:
                raise ValueError(f"Failed to read video from {video_path}")

            input_device = torch.device("cpu") if self._seedvr_sp_size > 1 else self.init_device
            input_dtype = torch.float32 if self._seedvr_sp_size > 1 else GET_DTYPE()
            img = video.to(device=input_device, dtype=input_dtype).div_(255.0)
            input_source = video_path
        elif "image_path" in self.input_info.__dataclass_fields__ and self.input_info.image_path:
            from PIL import Image

            img_path = self.input_info.image_path
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            input_device = torch.device("cpu") if self._seedvr_sp_size > 1 else self.init_device
            img = img.unsqueeze(0).to(input_device)
            input_source = img_path
        else:
            raise ValueError("SR task requires image_path or video_path")

        input_shape = tuple(img.shape)
        _, _, self.ori_h, self.ori_w = img.shape
        img = self._build_video_transform(img)
        if self._seedvr_sp_size > 1:
            img = img.to(dtype=GET_DTYPE())
        self._input = img
        self._ori_length = img.shape[1]
        img = self._cut_videos(img, sp_size=self._seedvr_sp_size)

        logger.debug(f"[SeedVRRunner] input={input_source}, input_shape={input_shape}, transformed_shape={tuple(img.shape)}")

        cond_latents = self.run_vae_encoder(img)
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

        torch.cuda.empty_cache()
        gc.collect()

        first_latent = cond_latents[0]
        latent_shape = [1, first_latent.shape[-1], first_latent.shape[0], first_latent.shape[1], first_latent.shape[2]]
        logger.debug(f"[SeedVRRunner] VAE latent shape={tuple(first_latent.shape)}, scheduler latent_shape={latent_shape}")

        return {
            "x": cond_latents[0],
            "conditions": conditions,
            "noises": noises,
            "vae_encoder_out": cond_latents[0],
            "image_encoder_output": None,
            "text_encoder_output": text_encoder_output,
            "latent_shape": latent_shape,
        }

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["SeedVRRunner"],
        profile_memory=True,
    )
    def run_pipeline(self, input_info):
        self.input_info = input_info

        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()

        video_path = getattr(self.input_info, "video_path", "")
        if self._seedvr_sp_size > 1 and not video_path:
            raise ValueError("SeedVR VAE sequence parallel currently supports video SR input only")
        seg_len, overlap = self._get_sr_segment_params()
        if not video_path or seg_len is None:
            self.inputs = self.run_input_encoder()
            return self.run_main()

        try:
            total_frames, fps, pts = self._probe_video(video_path)
            self._sr_video_backend = "torchvision"
        except Exception as e:
            logger.warning(f"[SeedVRRunner] torchvision video probe failed, switching segmented decode to torchcodec: {e}")
            total_frames, fps, pts = self._probe_video_torchcodec(video_path)
            self._sr_video_backend = "torchcodec"

        if total_frames <= seg_len or total_frames == 0:
            self.inputs = self.run_input_encoder()
            return self.run_main()

        self._sr_fps = fps
        self._sr_pts = pts
        segments = self._build_sr_segments(total_frames, seg_len, overlap)
        logger.info(f"[SeedVRRunner] SR segmenting: total_frames={total_frames}, seg_len={seg_len}, overlap={overlap}, segments={len(segments)}")

        original_save_path = self.input_info.save_result_path
        original_return_tensor = self.input_info.return_result_tensor
        file_output = bool(original_save_path) and not bool(original_return_tensor)
        stream_file_output = file_output and bool(self.config.get("stream_save_video", True))
        is_sp_root = self._seedvr_sp_rank == 0
        raw_segments = [] if (not file_output and is_sp_root) else None
        segment_paths = []
        tmp_dir = None
        video_recorder = None
        try:
            if file_output and is_sp_root:
                output_dir = os.path.dirname(original_save_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                if stream_file_output:
                    video_recorder = SeedVRVideoRecorder(
                        livestream_url=original_save_path,
                        fps=float(self.config.get("fps", 16)),
                    )
                    video_recorder.config_crf = int(self.config.get("video_crf", 16))
                    video_recorder.config_preset = str(self.config.get("video_preset", "medium"))
                    logger.info(f"[SeedVRRunner] segment stream writer initialized: {original_save_path}")
                else:
                    tmp_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(original_save_path)}.segments.", dir=output_dir)
            else:
                self.input_info.save_result_path = ""
                self.input_info.return_result_tensor = True

            for idx, (start_idx, end_idx) in enumerate(segments):
                with ProfilingContext4DebugL1(f"Segment {idx + 1}/{len(segments)} [{start_idx}:{end_idx}]"):
                    self._sr_segment = (start_idx, end_idx)
                    self.inputs = self.run_input_encoder()
                    raw = self._run_sr_single_segment()
                    if overlap > 0 and idx > 0 and raw is not None:
                        raw = raw[:, :, overlap:, :, :]

                    if file_output:
                        if is_sp_root:
                            if stream_file_output:
                                self._stream_sr_segment_video(raw, video_recorder, idx, len(segments))
                            else:
                                segment_path = os.path.join(tmp_dir, f"segment_{idx:05d}.mp4")
                                self._save_sr_segment_video(raw, segment_path, fps=self.config.get("fps", 16))
                                segment_paths.append(segment_path)
                        if raw is not None:
                            del raw
                        self.gen_video = None
                        self.gen_video_final = None
                        self._input = None
                        torch.cuda.empty_cache()
                        gc.collect()
                    elif is_sp_root:
                        raw_segments.append(raw)

                    if self._seedvr_sp_size > 1:
                        dist.barrier(group=self._seedvr_sp_group)

            if file_output:
                if is_sp_root:
                    if stream_file_output:
                        if video_recorder is None or video_recorder.width is None:
                            raise RuntimeError("SeedVR produced no video segments to stream.")
                        video_recorder.stop(wait=False)
                        video_recorder = None
                        if not os.path.isfile(original_save_path) or os.path.getsize(original_save_path) == 0:
                            raise RuntimeError(f"SeedVR stream writer did not produce a video: {original_save_path}")
                    else:
                        if not segment_paths:
                            raise RuntimeError("SeedVR produced no video segments to save.")
                        self._concat_sr_segment_videos(segment_paths, original_save_path)
                    input_video_path = getattr(self.input_info, "video_path", "")
                    if input_video_path:
                        mux_audio_from_video(input_video_path, original_save_path)
                    logger.info(f"✅ Video saved successfully to: {original_save_path} ✅")
                    result = {"video": None, "save_result_path": original_save_path}
                else:
                    result = {"video": None}
                if self._seedvr_sp_size > 1:
                    dist.barrier(group=self._seedvr_sp_group)
                return result
        finally:
            # Critical: restore per-request output mode even when cancelled/interrupted.
            self._sr_segment = None
            self._sr_video_backend = None
            self.input_info.save_result_path = original_save_path
            self.input_info.return_result_tensor = original_return_tensor
            if video_recorder is not None:
                video_recorder.stop(wait=False)
            if tmp_dir is not None and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

        if is_sp_root:
            self.gen_video_final = torch.cat(raw_segments, dim=2)
            result = self.process_images_after_vae_decoder()
            self.end_run()
        else:
            result = {"video": None}
        if self._seedvr_sp_size > 1:
            dist.barrier(group=self._seedvr_sp_group)
        return result
