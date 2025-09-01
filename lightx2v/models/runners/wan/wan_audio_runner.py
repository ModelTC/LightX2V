import gc
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torchaudio as ta
import torchvision.transforms.functional as TF
from PIL import Image
from einops import rearrange
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import AudioAdapter
from lightx2v.models.input_encoders.hf.seko_audio.audio_encoder import SekoAudioEncoderModel
from lightx2v.models.networks.wan.audio_model import WanAudioModel
from lightx2v.models.networks.wan.lora_adapter import WanLoraWrapper
from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.audio.scheduler import EulerScheduler
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import ProfilingContext, ProfilingContext4Debug
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, load_weights, save_to_video, vae_to_comfyui_image


def get_optimal_patched_size_with_sp(patched_h, patched_w, sp_size):
    assert sp_size > 0 and (sp_size & (sp_size - 1)) == 0, "sp_size must be a power of 2"

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


def adaptive_resize(img):
    bucket_config = {
        0.667: (np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64), np.array([0.2, 0.5, 0.3])),
        1.0: (np.array([[480, 480], [576, 576], [704, 704], [960, 960]], dtype=np.int64), np.array([0.1, 0.1, 0.5, 0.3])),
        1.5: (np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64)[:, ::-1], np.array([0.2, 0.5, 0.3])),
    }
    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight
    aspect_ratios = np.array(np.array(list(bucket_config.keys())))
    closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
    closet_ratio = aspect_ratios[closet_aspect_idx]
    if ori_ratio < 1.0:
        target_h, target_w = 480, 832
    elif ori_ratio == 1.0:
        target_h, target_w = 480, 480
    else:
        target_h, target_w = 832, 480
    for resolution in bucket_config[closet_ratio][0]:
        if ori_height * ori_weight >= resolution[0] * resolution[1]:
            target_h, target_w = resolution
    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    return cropped_img, target_h, target_w


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


@RUNNER_REGISTER("seko_talk")
class WanAudioRunner(WanRunner):  # type:ignore
    def __init__(self, config):
        super().__init__(config)
        self.frame_preprocessor = FramePreprocessor()

    def init_scheduler(self):
        """Initialize consistency model scheduler"""
        scheduler = EulerScheduler(self.config)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.audio_adapter = self.load_audio_adapter()
            self.model.set_audio_adapter(self.audio_adapter)
        scheduler.set_audio_adapter(self.audio_adapter)
        self.model.set_scheduler(scheduler)

    def read_audio_input(self):
        """Read audio input"""
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)
        audio_array = self._audio_processor.load_audio(self.config["audio_path"])

        video_duration = self.config.get("video_duration", 5)

        audio_len = int(audio_array.shape[0] / audio_sr * target_fps)
        expected_frames = min(max(1, int(video_duration * target_fps)), audio_len)

        # Segment audio
        audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, self.config.get("target_video_length", 81))

        return audio_segments, expected_frames

    def read_image_input(self, img_path):
        ref_img = Image.open(img_path).convert("RGB")
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(0).cuda()

        ref_img, h, w = adaptive_resize(ref_img)
        patched_h = h // self.config.vae_stride[1] // self.config.patch_size[1]
        patched_w = w // self.config.vae_stride[2] // self.config.patch_size[2]

        patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

        self.config.lat_h = patched_h * self.config.patch_size[1]
        self.config.lat_w = patched_w * self.config.patch_size[2]

        self.config.tgt_h = self.config.lat_h * self.config.vae_stride[1]
        self.config.tgt_w = self.config.lat_w * self.config.vae_stride[2]

        logger.info(f"[wan_audio] tgt_h: {self.config.tgt_h}, tgt_w: {self.config.tgt_w}, lat_h: {self.config.lat_h}, lat_w: {self.config.lat_w}")

        ref_img = torch.nn.functional.interpolate(ref_img, size=(self.config.tgt_h, self.config.tgt_w), mode="bicubic")
        return ref_img

    def run_image_encoder(self, first_frame, last_frame=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        clip_encoder_out = self.image_encoder.visual([first_frame]).squeeze(0).to(GET_DTYPE()) if self.config.get("use_image_encoder", True) else None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    def run_vae_encoder(self, img):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        img = rearrange(img, "1 C H W -> 1 C 1 H W")
        vae_encoder_out = self.vae_encoder.encode(img.to(GET_DTYPE()))

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return vae_encoder_out

    @ProfilingContext("Run Encoders")
    def _run_input_encoder_local_r2v_audio(self):
        prompt = self.config["prompt_enhanced"] if self.config["use_prompt_enhancer"] else self.config["prompt"]
        img = self.read_image_input(self.config["image_path"])
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)
        audio_segments, expected_frames = self.read_audio_input()
        text_encoder_output = self.run_text_encoder(prompt, None)
        torch.cuda.empty_cache()
        gc.collect()
        return {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encode_out,
            },
            "audio_segments": audio_segments,
            "expected_frames": expected_frames,
        }

    def prepare_prev_latents(self, prev_video: Optional[torch.Tensor], prev_frame_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare previous latents for conditioning"""
        device = torch.device("cuda")
        dtype = GET_DTYPE()

        tgt_h, tgt_w = self.config.tgt_h, self.config.tgt_w
        prev_frames = torch.zeros((1, 3, self.config.target_video_length, tgt_h, tgt_w), device=device)

        if prev_video is not None:
            # Extract and process last frames
            last_frames = prev_video[:, :, -prev_frame_length:].clone().to(device)
            if self.config.model_cls != "wan2.2_audio":
                last_frames = self.frame_preprocessor.process_prev_frames(last_frames)
            prev_frames[:, :, :prev_frame_length] = last_frames
            prev_len = (prev_frame_length - 1) // 4 + 1
        else:
            prev_len = 0

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        _, nframe, height, width = self.model.scheduler.latents.shape
        if self.config.model_cls == "wan2.2_audio":
            if prev_video is not None:
                prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))
            else:
                prev_latents = None
            prev_mask = self.model.scheduler.mask
        else:
            prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))

            frames_n = (nframe - 1) * 4 + 1
            prev_mask = torch.ones((1, frames_n, height, width), device=device, dtype=dtype)
            prev_frame_len = max((prev_len - 1) * 4 + 1, 0)
            prev_mask[:, prev_frame_len:] = 0
            prev_mask = self._wan_mask_rearrange(prev_mask)

        if prev_latents is not None:
            if prev_latents.shape[-2:] != (height, width):
                logger.warning(f"Size mismatch: prev_latents {prev_latents.shape} vs scheduler latents (H={height}, W={width}). Config tgt_h={self.config.tgt_h}, tgt_w={self.config.tgt_w}")
                prev_latents = torch.nn.functional.interpolate(prev_latents, size=(height, width), mode="bilinear", align_corners=False)

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_encoder
            torch.cuda.empty_cache()
            gc.collect()

        return {"prev_latents": prev_latents, "prev_mask": prev_mask, "prev_len": prev_len}

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

    def get_video_segment_num(self):
        self.video_segment_num = len(self.inputs["audio_segments"])

    def init_run(self):
        super().init_run()

        self.gen_video_list = []
        self.cut_audio_list = []
        self.prev_video = None

    @ProfilingContext4Debug("Init run segment")
    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx

        self.segment = self.inputs["audio_segments"][segment_idx]

        self.config.seed = self.config.seed + segment_idx
        torch.manual_seed(self.config.seed)
        logger.info(f"Processing segment {segment_idx + 1}/{self.video_segment_num}, seed: {self.config.seed}")

        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not hasattr(self, "audio_encoder"):
            self.audio_encoder = self.load_audio_encoder()

        audio_features = self.audio_encoder.infer(self.segment.audio_array)
        audio_features = self.audio_adapter.forward_audio_proj(audio_features, self.model.scheduler.latents.shape[1])

        self.inputs["audio_encoder_output"] = audio_features
        self.inputs["previmg_encoder_output"] = self.prepare_prev_latents(self.prev_video, prev_frame_length=5)

        # Reset scheduler for non-first segments
        if segment_idx > 0:
            self.model.scheduler.reset(self.inputs["previmg_encoder_output"])

    @ProfilingContext4Debug("End run segment")
    def end_run_segment(self):
        self.gen_video = torch.clamp(self.gen_video, -1, 1).to(torch.float)

        # Extract relevant frames
        start_frame = 0 if self.segment_idx == 0 else 5
        start_audio_frame = 0 if self.segment_idx == 0 else int(6 * self._audio_processor.audio_sr / self.config.get("target_fps", 16))

        if self.segment.is_last and self.segment.useful_length:
            end_frame = self.segment.end_frame - self.segment.start_frame
            self.gen_video_list.append(self.gen_video[:, :, start_frame:end_frame].cpu())
            self.cut_audio_list.append(self.segment.audio_array[start_audio_frame : self.segment.useful_length])
        elif self.segment.useful_length and self.inputs["expected_frames"] < self.config.get("target_video_length", 81):
            self.gen_video_list.append(self.gen_video[:, :, start_frame : self.inputs["expected_frames"]].cpu())
            self.cut_audio_list.append(self.segment.audio_array[start_audio_frame : self.segment.useful_length])
        else:
            self.gen_video_list.append(self.gen_video[:, :, start_frame:].cpu())
            self.cut_audio_list.append(self.segment.audio_array[start_audio_frame:])

        # Update prev_video for next iteration
        self.prev_video = self.gen_video

        # Clean up GPU memory after each segment
        del self.gen_video
        torch.cuda.empty_cache()

    @ProfilingContext4Debug("Process after vae decoder")
    def process_images_after_vae_decoder(self, save_video=True):
        # Merge results
        gen_lvideo = torch.cat(self.gen_video_list, dim=2).float()
        merge_audio = np.concatenate(self.cut_audio_list, axis=0).astype(np.float32)

        comfyui_images = vae_to_comfyui_image(gen_lvideo)

        # Apply frame interpolation if configured
        if "video_frame_interpolation" in self.config and self.vfi_model is not None:
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            comfyui_images = self.vfi_model.interpolate_frames(
                comfyui_images,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if save_video:
            if "video_frame_interpolation" in self.config and self.config["video_frame_interpolation"].get("target_fps"):
                fps = self.config["video_frame_interpolation"]["target_fps"]
            else:
                fps = self.config.get("fps", 16)

            if not dist.is_initialized() or dist.get_rank() == 0:
                logger.info(f"🎬 Start to save video 🎬")

                self._save_video_with_audio(comfyui_images, merge_audio, fps)
                logger.info(f"✅ Video saved successfully to: {self.config.save_video_path} ✅")

        # Convert audio to ComfyUI format
        audio_waveform = torch.from_numpy(merge_audio).unsqueeze(0).unsqueeze(0)
        comfyui_audio = {"waveform": audio_waveform, "sample_rate": self._audio_processor.audio_sr}

        return {"video": comfyui_images, "audio": comfyui_audio}

    def init_modules(self):
        super().init_modules()
        self.run_input_encoder = self._run_input_encoder_local_r2v_audio

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

        return base_model

    def load_audio_encoder(self):
        audio_encoder_path = os.path.join(self.config["model_path"], "TencentGameMate-chinese-hubert-large")
        audio_encoder_offload = self.config.get("audio_encoder_cpu_offload", self.config.get("cpu_offload", False))
        model = SekoAudioEncoderModel(audio_encoder_path, self.config["audio_sr"], audio_encoder_offload)
        return model

    def load_audio_adapter(self):
        audio_adapter_offload = self.config.get("audio_adapter_cpu_offload", self.config.get("cpu_offload", False))
        if audio_adapter_offload:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        audio_adapter = AudioAdapter(
            attention_head_dim=self.config["dim"] // self.config["num_heads"],
            num_attention_heads=self.config["num_heads"],
            base_num_layers=self.config["num_layers"],
            interval=1,
            audio_feature_dim=1024,
            time_freq_dim=256,
            projection_transformer_layers=4,
            mlp_dims=(1024, 1024, 32 * 1024),
            quantized=self.config.get("adapter_quantized", False),
            quant_scheme=self.config.get("adapter_quant_scheme", None),
            cpu_offload=audio_adapter_offload,
        )
        audio_adapter.to(device)
        if self.config.get("adapter_quantized", False):
            if self.config.get("adapter_quant_scheme", None) in ["fp8", "fp8-q8f"]:
                model_name = "audio_adapter_model_fp8.safetensors"
            elif self.config.get("adapter_quant_scheme", None) == "int8":
                model_name = "audio_adapter_model_int8.safetensors"
            else:
                raise ValueError(f"Unsupported quant_scheme: {self.config.get('adapter_quant_scheme', None)}")
        else:
            model_name = "audio_adapter_model.safetensors"

        weights_dict = load_weights(os.path.join(self.config["model_path"], model_name), cpu_offload=audio_adapter_offload)
        audio_adapter.load_state_dict(weights_dict, strict=False)
        return audio_adapter.to(dtype=GET_DTYPE())

    @ProfilingContext("Load models")
    def load_model(self):
        super().load_model()
        self.audio_encoder = self.load_audio_encoder()
        self.audio_adapter = self.load_audio_adapter()
        self.model.set_audio_adapter(self.audio_adapter)

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
