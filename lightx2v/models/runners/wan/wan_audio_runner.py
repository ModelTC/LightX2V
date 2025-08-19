import gc
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
from PIL import Image
from einops import rearrange
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from transformers import AutoFeatureExtractor

from lightx2v.models.networks.wan.audio_adapter import AudioAdapter, AudioAdapterPipe, rank0_load_state_dict_from_path
from lightx2v.models.networks.wan.audio_model import Wan22MoeAudioModel, WanAudioModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner
from lightx2v.models.schedulers.wan.audio.scheduler import EulerSchedulerTimestepFix
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, save_to_video, vae_to_comfyui_image


@contextmanager
def memory_efficient_inference():
    """Context manager for memory-efficient inference"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_optimal_patched_size_with_sp(patched_h, patched_w, sp_size):
    def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

    assert is_power_of_two(sp_size), "sp_size must be a power of 2"
    h_ratio, w_ratio = 1, 1
    while sp_size != 1:
        sp_size //= 2
        if patched_h % 2 == 0:
            patched_h //= 2
            h_ratio *= 2
        elif patched_w % 2 == 0:
            patched_w //= 2
            w_ratio *= 2
        else:
            if patched_h > patched_w:
                patched_h //= 2
                h_ratio *= 2
            else:
                patched_w //= 2
                w_ratio *= 2
    return patched_h * h_ratio, patched_w * w_ratio


def get_crop_bbox(ori_h, ori_w, tgt_h, tgt_w):
    tgt_ar = tgt_h / tgt_w
    ori_ar = ori_h / ori_w
    if abs(ori_ar - tgt_ar) < 0.01:
        return 0, ori_h, 0, ori_w
    if ori_ar > tgt_ar:
        crop_h = int(tgt_ar * ori_w)
        y0 = (ori_h - crop_h) // 2
        y1 = y0 + crop_h
        return y0, y1, 0, ori_w
    else:
        crop_w = int(ori_h / tgt_ar)
        x0 = (ori_w - crop_w) // 2
        x1 = x0 + crop_w
        return 0, ori_h, x0, x1


def isotropic_crop_resize(frames: torch.Tensor, size: tuple):
    """
    frames: (T, C, H, W)
    size: (H, W)
    """
    ori_h, ori_w = frames.shape[2:]
    h, w = size
    y0, y1, x0, x1 = get_crop_bbox(ori_h, ori_w, h, w)
    cropped_frames = frames[:, :, y0:y1, x0:x1]
    resized_frames = resize(cropped_frames, [h, w], InterpolationMode.BICUBIC, antialias=True)
    return resized_frames


SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
    "1024*1024": (1024, 1024),
    "544*960": (544, 960),
    "960*544": (960 * 544),
    "576*576": (576, 576),
    "704*704": (704, 704),
    "960*960": (960, 960),
    "704*1280": (704, 1280),
    "1280*704": (1280, 704),
}

SUPPORTED_SIZES = {
    "t2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2v-1.3B": ("480*832", "832*480"),
    "i2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "flf2v-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "t2i-14B": tuple(SIZE_CONFIGS.keys()),
    "vace-1.3B": ("480*832", "832*480"),
    "vace-14B": ("720*1280", "1280*720", "480*832", "832*480"),
    "r2v-audio": ("720*1280", "1280*720", "480*832", "832*480", "544*960", "960*544", "576*576", "704*704", "960*960"),
    "tr2v-audio": ("704*1280", "1280*704"),
}

DEFAULT_BUCKET_CONFIG = {
    0.58: ((480, 832), (544, 960), (720, 1280)),
    1.0: ((576, 576), (704, 704), (960, 960)),
    1.7: ((832, 480), (960, 544), (1280, 720)),
}


def _get_bucket_config(target_resolutions=None):
    if target_resolutions is None:
        return DEFAULT_BUCKET_CONFIG

    resolutions = sorted((SIZE_CONFIGS[key] for key in target_resolutions), key=lambda r: r[0] * r[1])
    bucket_config = {}
    for w, h in resolutions:
        aspect = round(w / h, 2)
        bucket_config.setdefault(aspect, []).append([w, h])
    return bucket_config


def get_adaptive_size(height, width, target_resolutions=None):
    bucket_config = _get_bucket_config(target_resolutions)
    ori_ratio = height / width
    aspect_ratios = np.array(np.array(list(bucket_config.keys())))
    closet_ratio = aspect_ratios[np.argmin(np.abs(aspect_ratios - ori_ratio))]
    target_h, target_w = bucket_config[closet_ratio][0]
    for resolution in bucket_config[closet_ratio]:
        if height * width >= resolution[0] * resolution[1]:
            target_h, target_w = resolution
    return target_h, target_w


@dataclass
class AudioSegment:
    """Data class for audio segment information"""

    audio_array: np.ndarray
    start_frame: int
    end_frame: int
    is_last: bool = False
    useful_length: Optional[int] = None


class FramePreprocessor:
    """Handles frame preprocessing including noise and masking"""

    def __init__(self, noise_mean: float = -3.0, noise_std: float = 0.5, mask_rate: float = 0.1):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.mask_rate = mask_rate

    def add_noise(self, frames: np.ndarray, rnd_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Add noise to frames"""
        if self.noise_mean is None or self.noise_std is None:
            return frames

        if rnd_state is None:
            rnd_state = np.random.RandomState()

        shape = frames.shape
        bs = 1 if len(shape) == 4 else shape[0]
        sigma = rnd_state.normal(loc=self.noise_mean, scale=self.noise_std, size=(bs,))
        sigma = np.exp(sigma)
        sigma = np.expand_dims(sigma, axis=tuple(range(1, len(shape))))
        noise = rnd_state.randn(*shape) * sigma
        return frames + noise

    def add_mask(self, frames: np.ndarray, rnd_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Add mask to frames"""
        if self.mask_rate is None:
            return frames

        if rnd_state is None:
            rnd_state = np.random.RandomState()

        h, w = frames.shape[-2:]
        mask = rnd_state.rand(h, w) > self.mask_rate
        return frames * mask

    def process_prev_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Process previous frames with noise and masking"""
        frames_np = frames.cpu().detach().numpy()
        frames_np = self.add_noise(frames_np)
        frames_np = self.add_mask(frames_np)
        return torch.from_numpy(frames_np).to(dtype=frames.dtype, device=frames.device)


class AudioProcessor:
    """Handles audio loading and segmentation"""

    def __init__(self, audio_sr: int = 16000, target_fps: int = 16):
        self.audio_sr = audio_sr
        self.target_fps = target_fps

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio"""
        audio_array, ori_sr = ta.load(audio_path)
        audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=self.audio_sr)
        return audio_array.numpy()

    def get_audio_range(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Calculate audio range for given frame range"""
        audio_frame_rate = self.audio_sr / self.target_fps
        return round(start_frame * audio_frame_rate), round((end_frame + 1) * audio_frame_rate)

    def segment_audio(self, audio_array: np.ndarray, expected_frames: int, max_num_frames: int, prev_frame_length: int = 5) -> List[AudioSegment]:
        """Segment audio based on frame requirements"""
        segments = []

        # Calculate intervals
        interval_num = 1
        res_frame_num = 0

        if expected_frames <= max_num_frames:
            interval_num = 1
        else:
            interval_num = max(int((expected_frames - max_num_frames) / (max_num_frames - prev_frame_length)) + 1, 1)
            res_frame_num = expected_frames - interval_num * (max_num_frames - prev_frame_length)
            if res_frame_num > 5:
                interval_num += 1

        # Create segments
        for idx in range(interval_num):
            if idx == 0:
                # First segment
                audio_start, audio_end = self.get_audio_range(0, max_num_frames)
                segment_audio = audio_array[audio_start:audio_end]
                useful_length = None

                if expected_frames < max_num_frames:
                    useful_length = segment_audio.shape[0]
                    max_num_audio_length = int((max_num_frames + 1) / self.target_fps * self.audio_sr)
                    segment_audio = np.concatenate((segment_audio, np.zeros(max_num_audio_length - useful_length)), axis=0)

                segments.append(AudioSegment(segment_audio, 0, max_num_frames, False, useful_length))

            elif res_frame_num > 5 and idx == interval_num - 1:
                # Last segment (might be shorter)
                start_frame = idx * max_num_frames - idx * prev_frame_length
                audio_start, audio_end = self.get_audio_range(start_frame, expected_frames)
                segment_audio = audio_array[audio_start:audio_end]
                useful_length = segment_audio.shape[0]

                max_num_audio_length = int((max_num_frames + 1) / self.target_fps * self.audio_sr)
                segment_audio = np.concatenate((segment_audio, np.zeros(max_num_audio_length - useful_length)), axis=0)

                segments.append(AudioSegment(segment_audio, start_frame, expected_frames, True, useful_length))

            else:
                # Middle segments
                start_frame = idx * max_num_frames - idx * prev_frame_length
                end_frame = (idx + 1) * max_num_frames - idx * prev_frame_length
                audio_start, audio_end = self.get_audio_range(start_frame, end_frame)
                segment_audio = audio_array[audio_start:audio_end]

                segments.append(AudioSegment(segment_audio, start_frame, end_frame, False))

        return segments


class VideoGenerator:
    """Handles video generation for each segment"""

    def __init__(self, model, vae_encoder, vae_decoder, config, progress_callback=None):
        self.model = model
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.config = config
        self.frame_preprocessor = FramePreprocessor()
        self.progress_callback = progress_callback
        self.total_segments = 1

    def prepare_prev_latents(self, prev_video: Optional[torch.Tensor], prev_frame_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare previous latents for conditioning"""
        device = torch.device("cuda")
        dtype = GET_DTYPE()
        vae_dtype = torch.float

        tgt_h, tgt_w = self.config.tgt_h, self.config.tgt_w
        prev_frames = torch.zeros((1, 3, self.config.target_video_length, tgt_h, tgt_w), device=device)

        if prev_video is not None:
            # Extract and process last frames
            last_frames = prev_video[:, :, -prev_frame_length:].clone().to(device)
            last_frames = self.frame_preprocessor.process_prev_frames(last_frames)
            prev_frames[:, :, :prev_frame_length] = last_frames
            prev_len = (prev_frame_length - 1) // 4 + 1
        else:
            prev_len = 0

        _, nframe, height, width = self.model.scheduler.latents.shape
        if self.config.model_cls == "wan2.2_audio":
            if prev_video is not None:
                prev_latents = self.vae_encoder.encode(prev_frames.to(vae_dtype)).to(dtype)
            else:
                prev_latents = None
            _, prev_mask = self._wan22_masks_like([self.model.scheduler.latents], zero=True, prev_length=prev_len)
        else:
            prev_latents = self.vae_encoder.encode(prev_frames.to(vae_dtype))[0].to(dtype)

            if prev_latents is not None:
                prev_token_length = (prev_frame_length - 1) // 4 + 1
                prev_frame_len = max((prev_token_length - 1) * 4 + 1, 0)
            else:
                prev_frame_len = 0

            frames_n = (nframe - 1) * 4 + 1
            prev_mask = torch.ones((1, frames_n, height, width), device=device, dtype=dtype)
            prev_mask[:, prev_frame_len:] = 0
            prev_mask = self._wan_mask_rearrange(prev_mask).unsqueeze(0)

        if prev_latents is not None:
            if prev_latents.shape[-2:] != (height, width):
                logger.warning(f"Size mismatch: prev_latents {prev_latents.shape} vs scheduler latents (H={height}, W={width}). Config tgt_h={self.config.tgt_h}, tgt_w={self.config.tgt_w}")
                prev_latents = torch.nn.functional.interpolate(prev_latents, size=(height, width), mode="bilinear", align_corners=False)

        return {"prev_latents": prev_latents, "prev_mask": prev_mask}

    def _wan22_masks_like(self, tensor, zero=False, generator=None, p=0.2, prev_length=1):
        assert isinstance(tensor, list)
        out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]
        out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

        if prev_length == 0:
            return out1, out2

        if zero:
            if generator is not None:
                for u, v in zip(out1, out2):
                    random_num = torch.rand(1, generator=generator, device=generator.device).item()
                    if random_num < p:
                        u[:, :prev_length] = torch.normal(mean=-3.5, std=0.5, size=(1,), device=u.device, generator=generator).expand_as(u[:, :prev_length]).exp()
                        v[:, :prev_length] = torch.zeros_like(v[:, :prev_length])
                    else:
                        u[:, :prev_length] = u[:, :prev_length]
                        v[:, :prev_length] = v[:, :prev_length]
            else:
                for u, v in zip(out1, out2):
                    u[:, :prev_length] = torch.zeros_like(u[:, :prev_length])
                    v[:, :prev_length] = torch.zeros_like(v[:, :prev_length])

        return out1, out2

    def _wan_mask_rearrange(self, mask: torch.Tensor) -> torch.Tensor:
        """Rearrange mask for WAN model"""
        if mask.ndim == 3:
            mask = mask[None]
        assert mask.ndim == 4
        _, t, h, w = mask.shape
        assert t == ((t - 1) // 4 * 4 + 1)
        mask_first_frame = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        mask = torch.concat([mask_first_frame, mask[:, 1:]], dim=1)
        mask = mask.view(mask.shape[1] // 4, 4, h, w)
        return mask.transpose(0, 1)

    @torch.no_grad()
    def generate_segment(self, inputs, audio_features, prev_video=None, prev_frame_length=5, segment_idx=0, total_steps=None):
        """Generate video segment"""
        # Update inputs with audio features
        inputs["audio_encoder_output"] = audio_features

        # Reset scheduler for non-first segments
        if segment_idx > 0:
            self.model.scheduler.reset()

        inputs["previmg_encoder_output"] = self.prepare_prev_latents(prev_video, prev_frame_length)

        # Run inference loop
        if total_steps is None:
            total_steps = self.model.scheduler.infer_steps
        for step_index in range(total_steps):
            logger.info(f"==> Segment {segment_idx}, Step {step_index}/{total_steps}")

            with ProfilingContext4Debug("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)

            with ProfilingContext4Debug("🚀 infer_main"):
                self.model.infer(inputs)

            with ProfilingContext4Debug("step_post"):
                self.model.scheduler.step_post()
                if self.config.model_cls == "wan2.2_audio":
                    prev_mask = inputs["previmg_encoder_output"]["prev_mask"]
                    prev_latents = inputs["previmg_encoder_output"]["prev_latents"]
                    if prev_latents is not None:
                        self.model.scheduler.latents = (1.0 - prev_mask[0]) * prev_latents + prev_mask[0] * self.model.scheduler.latents

            if self.progress_callback:
                segment_progress = (segment_idx * total_steps + step_index + 1) / (self.total_segments * total_steps)
                self.progress_callback(int(segment_progress * 100), 100)

        # Decode latents
        latents = self.model.scheduler.latents
        generator = self.model.scheduler.generator
        with ProfilingContext("Run VAE Decoder"):
            gen_video = self.vae_decoder.decode(latents, generator=generator, config=self.config)
        gen_video = torch.clamp(gen_video, -1, 1).to(torch.float)

        return gen_video


@RUNNER_REGISTER("wan2.1_audio")
class WanAudioRunner(WanRunner):  # type:ignore
    def __init__(self, config):
        super().__init__(config)
        self._audio_adapter_pipe = None
        self._audio_processor = None
        self._video_generator = None
        self._audio_preprocess = None

    def initialize(self):
        """Initialize all models once for multiple runs"""

        # Initialize audio processor
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)

        # Initialize scheduler
        self.init_scheduler()

    def init_scheduler(self):
        """Initialize consistency model scheduler"""
        # scheduler = ConsistencyModelScheduler(self.config)
        scheduler = EulerSchedulerTimestepFix(self.config)
        self.model.set_scheduler(scheduler)

    def load_audio_adapter_lazy(self):
        """Lazy load audio adapter when needed"""
        if self._audio_adapter_pipe is not None:
            return self._audio_adapter_pipe

        # Audio adapter
        audio_adapter_path = self.config["model_path"] + "/audio_adapter.safetensors"
        audio_adapter = AudioAdapter.from_transformer(
            self.model,
            audio_feature_dim=1024,
            interval=1,
            time_freq_dim=256,
            projection_transformer_layers=4,
        )

        # Audio encoder
        cpu_offload = self.config.get("cpu_offload", False)
        if cpu_offload:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        audio_encoder_repo = self.config["model_path"] + "/audio_encoder"

        if self.model.transformer_infer.seq_p_group is not None:
            seq_p_group = self.model.transformer_infer.seq_p_group
        else:
            seq_p_group = None

        audio_adapter = rank0_load_state_dict_from_path(audio_adapter, audio_adapter_path, strict=False)

        self._audio_adapter_pipe = AudioAdapterPipe(
            audio_adapter, audio_encoder_repo=audio_encoder_repo, dtype=GET_DTYPE(), device=device, weight=1.0, cpu_offload=cpu_offload, seq_p_group=seq_p_group
        )

        return self._audio_adapter_pipe

    def prepare_inputs(self):
        """Prepare inputs for the model"""
        image_encoder_output = None

        if os.path.isfile(self.config.image_path):
            with ProfilingContext("Run Img Encoder"):
                vae_encoder_out, clip_encoder_out = self.run_image_encoder(self.config, self.vae_encoder)
                image_encoder_output = {
                    "clip_encoder_out": clip_encoder_out,
                    "vae_encoder_out": vae_encoder_out,
                }

        with ProfilingContext("Run Text Encoder"):
            img = Image.open(self.config["image_path"]).convert("RGB")
            text_encoder_output = self.run_text_encoder(self.config["prompt"], img)

        self.set_target_shape()

        return {"text_encoder_output": text_encoder_output, "image_encoder_output": image_encoder_output, "audio_adapter_pipe": self.load_audio_adapter_lazy()}

    def run_pipeline(self, save_video=True):
        """Optimized pipeline with modular components"""

        try:
            self.initialize()

            assert self._audio_processor is not None
            assert self._audio_preprocess is not None

            self._video_generator = VideoGenerator(self.model, self.vae_encoder, self.vae_decoder, self.config, self.progress_callback)

            with memory_efficient_inference():
                if self.config["use_prompt_enhancer"]:
                    self.config["prompt_enhanced"] = self.post_prompt_enhancer()

                self.inputs = self.prepare_inputs()
                # Re-initialize scheduler after image encoding sets correct dimensions
                self.init_scheduler()
                self.model.scheduler.prepare(self.inputs["image_encoder_output"])

            # Re-create video generator with updated model/scheduler
            self._video_generator = VideoGenerator(self.model, self.vae_encoder, self.vae_decoder, self.config, self.progress_callback)

            # Process audio
            audio_array = self._audio_processor.load_audio(self.config["audio_path"])
            video_duration = self.config.get("video_duration", 5)
            target_fps = self.config.get("target_fps", 16)
            max_num_frames = self.config.get("target_video_length", 81)

            audio_len = int(audio_array.shape[0] / self._audio_processor.audio_sr * target_fps)
            expected_frames = min(max(1, int(video_duration * target_fps)), audio_len)

            # Segment audio
            audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, max_num_frames)

            self._video_generator.total_segments = len(audio_segments)

            # Generate video segments
            gen_video_list = []
            cut_audio_list = []
            prev_video = None

            for idx, segment in enumerate(audio_segments):
                self.config.seed = self.config.seed + idx
                torch.manual_seed(self.config.seed)
                logger.info(f"Processing segment {idx + 1}/{len(audio_segments)}, seed: {self.config.seed}")

                # Process audio features
                audio_features = self._audio_preprocess(segment.audio_array, sampling_rate=self._audio_processor.audio_sr, return_tensors="pt").input_values.squeeze(0).to(self.model.device)

                # Generate video segment
                with memory_efficient_inference():
                    gen_video = self._video_generator.generate_segment(
                        self.inputs.copy(),  # Copy to avoid modifying original
                        audio_features,
                        prev_video=prev_video,
                        prev_frame_length=5,
                        segment_idx=idx,
                    )

                # Extract relevant frames
                start_frame = 0 if idx == 0 else 5
                start_audio_frame = 0 if idx == 0 else int(6 * self._audio_processor.audio_sr / target_fps)

                if segment.is_last and segment.useful_length:
                    end_frame = segment.end_frame - segment.start_frame
                    gen_video_list.append(gen_video[:, :, start_frame:end_frame].cpu())
                    cut_audio_list.append(segment.audio_array[start_audio_frame : segment.useful_length])
                elif segment.useful_length and expected_frames < max_num_frames:
                    gen_video_list.append(gen_video[:, :, start_frame:expected_frames].cpu())
                    cut_audio_list.append(segment.audio_array[start_audio_frame : segment.useful_length])
                else:
                    gen_video_list.append(gen_video[:, :, start_frame:].cpu())
                    cut_audio_list.append(segment.audio_array[start_audio_frame:])

                # Update prev_video for next iteration
                prev_video = gen_video

                # Clean up GPU memory after each segment
                del gen_video
                torch.cuda.empty_cache()

            # Merge results
            with memory_efficient_inference():
                gen_lvideo = torch.cat(gen_video_list, dim=2).float()
                merge_audio = np.concatenate(cut_audio_list, axis=0).astype(np.float32)
                comfyui_images = vae_to_comfyui_image(gen_lvideo)

            # Apply frame interpolation if configured
            if "video_frame_interpolation" in self.config and self.vfi_model is not None:
                interpolation_target_fps = self.config["video_frame_interpolation"]["target_fps"]
                logger.info(f"Interpolating frames from {target_fps} to {interpolation_target_fps}")
                comfyui_images = self.vfi_model.interpolate_frames(
                    comfyui_images,
                    source_fps=target_fps,
                    target_fps=interpolation_target_fps,
                )
                target_fps = interpolation_target_fps

            # Convert audio to ComfyUI format
            audio_waveform = torch.from_numpy(merge_audio).unsqueeze(0).unsqueeze(0)
            comfyui_audio = {"waveform": audio_waveform, "sample_rate": self._audio_processor.audio_sr}

            # Save video if requested
            if (self.config.get("device_mesh") is not None and dist.get_rank() == 0) or self.config.get("device_mesh") is None:
                if save_video and self.config.get("save_video_path", None):
                    self._save_video_with_audio(comfyui_images, merge_audio, target_fps)

            # Final cleanup
            self.end_run()

            return comfyui_images, comfyui_audio

        finally:
            self._video_generator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_video_with_audio(self, images, audio_array, fps):
        """Save video with audio"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_tmp:
            video_path = video_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = audio_tmp.name

        try:
            save_to_video(images, video_path, fps)
            ta.save(audio_path, torch.tensor(audio_array[None]), sample_rate=self._audio_processor.audio_sr)  # type: ignore

            output_path = self.config.get("save_video_path")
            parent_dir = os.path.dirname(output_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with ProfilingContext("Run FFmpeg MP4 Encoder"):
                subprocess.call(["/usr/bin/ffmpeg", "-y", "-i", video_path, "-i", audio_path, output_path])

            logger.info(f"Saved video with audio to: {output_path}")

        finally:
            # Clean up temp files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def load_transformer(self):
        """Load transformer with LoRA support"""
        base_model = WanAudioModel(self.config.model_path, self.config, self.init_device)
        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)
            lora_wrapper = WanLoraWrapper(base_model)
            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                lora_name = lora_wrapper.load_lora(lora_path)
                lora_wrapper.apply_lora(lora_name, strength)
                logger.info(f"Loaded LoRA: {lora_name} with strength: {strength}")

        # XXX: trick
        self._audio_preprocess = AutoFeatureExtractor.from_pretrained(self.config["model_path"], subfolder="audio_encoder")

        return base_model

    def run_image_encoder(self, config, vae_model):
        """Run image encoder"""

        ref_img = Image.open(config.image_path)
        ref_img = (np.array(ref_img).astype(np.float32) - 127.5) / 127.5
        ref_img = torch.from_numpy(ref_img).cuda()
        ref_img = rearrange(ref_img, "H W C -> 1 C H W")
        ref_img = ref_img[:, :3]

        adaptive = config.get("adaptive_resize", False)

        h, w = ref_img.shape[2:]
        if adaptive:
            # Use adaptive_resize to modify aspect ratio
            if self.config.model_cls == "wan2.2_audio":
                h, w = get_adaptive_size(h, w, SUPPORTED_SIZES["tr2v-audio"])
            else:
                h, w = get_adaptive_size(h, w, SUPPORTED_SIZES["r2v-audio"])

            patched_h = h // self.config.vae_stride[1] // self.config.patch_size[1]
            patched_w = w // self.config.vae_stride[2] // self.config.patch_size[2]

        else:
            aspect_ratio = h / w
            max_area = config.target_height * config.target_width

            patched_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1])
            patched_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2])

        patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

        config.lat_h = patched_h * self.config.patch_size[1]
        config.lat_w = patched_w * self.config.patch_size[2]

        config.tgt_h = config.lat_h * self.config.vae_stride[1]
        config.tgt_w = config.lat_w * self.config.vae_stride[2]

        logger.info(f"[wan_audio] adaptive_resize: {adaptive}, tgt_h: {config.tgt_h}, tgt_w: {config.tgt_w}, lat_h: {config.lat_h}, lat_w: {config.lat_w}")

        cond_frms = isotropic_crop_resize(ref_img, size=(config.tgt_h, config.tgt_w)).transpose(0, 1)

        # clip encoder
        clip_encoder_out = self.image_encoder.visual([cond_frms]).squeeze(0).to(GET_DTYPE()) if self.config.get("use_image_encoder", True) else None
        # vae encode
        # cond_frms = rearrange(cond_frms, "1 C H W -> 1 C 1 H W")
        vae_encoder_out = vae_model.encode(cond_frms.to(torch.float32), config)

        if self.config.model_cls == "wan2.2_audio":
            vae_encoder_out = vae_encoder_out.unsqueeze(0).to(GET_DTYPE())
        else:
            if isinstance(vae_encoder_out, list):
                vae_encoder_out = torch.stack(vae_encoder_out, dim=0).to(GET_DTYPE())

        return vae_encoder_out, clip_encoder_out

    def set_target_shape(self):
        """Set target shape for generation"""
        ret = {}
        num_channels_latents = 16
        if self.config.model_cls == "wan2.2_audio":
            num_channels_latents = self.config.num_channels_latents

        if self.config.task == "i2v":
            self.config.target_shape = (
                num_channels_latents,
                (self.config.target_video_length - 1) // self.config.vae_stride[0] + 1,
                self.config.lat_h,
                self.config.lat_w,
            )
            ret["lat_h"] = self.config.lat_h
            ret["lat_w"] = self.config.lat_w
        else:
            error_msg = "t2v task is not supported in WanAudioRunner"
            assert False, error_msg

        ret["target_shape"] = self.config.target_shape
        return ret

    def run_step(self):
        """Optimized pipeline with modular components"""

        self.initialize()

        assert self._audio_processor is not None
        assert self._audio_preprocess is not None

        self._video_generator = VideoGenerator(self.model, self.vae_encoder, self.vae_decoder, self.config, self.progress_callback)

        with memory_efficient_inference():
            if self.config["use_prompt_enhancer"]:
                self.config["prompt_enhanced"] = self.post_prompt_enhancer()

            self.inputs = self.prepare_inputs()
            # Re-initialize scheduler after image encoding sets correct dimensions
            self.init_scheduler()
            self.model.scheduler.prepare(self.inputs["image_encoder_output"])

        # Re-create video generator with updated model/scheduler
        self._video_generator = VideoGenerator(self.model, self.vae_encoder, self.vae_decoder, self.config, self.progress_callback)

        # Process audio
        audio_array = self._audio_processor.load_audio(self.config["audio_path"])
        video_duration = self.config.get("video_duration", 5)
        target_fps = self.config.get("target_fps", 16)
        max_num_frames = self.config.get("target_video_length", 81)

        audio_len = int(audio_array.shape[0] / self._audio_processor.audio_sr * target_fps)
        expected_frames = min(max(1, int(video_duration * target_fps)), audio_len)

        # Segment audio
        audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, max_num_frames)

        self._video_generator.total_segments = len(audio_segments)

        # Generate video segments
        prev_video = None

        torch.manual_seed(self.config.seed)
        # Process audio features
        audio_features = self._audio_preprocess(audio_segments[0].audio_array, sampling_rate=self._audio_processor.audio_sr, return_tensors="pt").input_values.squeeze(0).to(self.model.device)

        # Generate video segment
        with memory_efficient_inference():
            self._video_generator.generate_segment(
                self.inputs.copy(),  # Copy to avoid modifying original
                audio_features,
                prev_video=prev_video,
                prev_frame_length=5,
                segment_idx=0,
                total_steps=1,
            )
            # Final cleanup
            self.end_run()


@RUNNER_REGISTER("wan2.2_audio")
class Wan22AudioRunner(WanAudioRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_vae_decoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        vae_decoder = Wan2_2_VAE(**vae_config)
        return vae_decoder

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device("cuda")
        vae_config = {
            "vae_pth": find_torch_model_path(self.config, "vae_pth", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
        }
        if self.config.task != "i2v":
            return None
        else:
            return Wan2_2_VAE(**vae_config)

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder


@RUNNER_REGISTER("wan2.2_moe_audio")
class Wan22MoeAudioRunner(WanAudioRunner):
    def __init__(self, config):
        super().__init__(config)

    def load_transformer(self):
        # encoder -> high_noise_model -> low_noise_model -> vae -> video_output
        high_noise_model = Wan22MoeAudioModel(
            os.path.join(self.config.model_path, "high_noise_model"),
            self.config,
            self.init_device,
        )
        low_noise_model = Wan22MoeAudioModel(
            os.path.join(self.config.model_path, "low_noise_model"),
            self.config,
            self.init_device,
        )

        if self.config.get("lora_configs") and self.config.lora_configs:
            assert not self.config.get("dit_quantized", False) or self.config.mm_config.get("weight_auto_quant", False)

            for lora_config in self.config.lora_configs:
                lora_path = lora_config["path"]
                strength = lora_config.get("strength", 1.0)
                if lora_config.name == "high_noise_model":
                    lora_wrapper = WanLoraWrapper(high_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"{lora_config.name} Loaded LoRA: {lora_name} with strength: {strength}")

                if lora_config.name == "low_noise_model":
                    lora_wrapper = WanLoraWrapper(low_noise_model)
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, strength)
                    logger.info(f"{lora_config.name} Loaded LoRA: {lora_name} with strength: {strength}")
        # XXX: trick
        self._audio_preprocess = AutoFeatureExtractor.from_pretrained(self.config["model_path"], subfolder="audio_encoder")

        return MultiModelStruct([high_noise_model, low_noise_model], self.config, self.config.boundary)
