import gc
import io
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio as ta
import torchvision.transforms.functional as TF
from PIL import Image, ImageCms, ImageOps
from einops import rearrange
from loguru import logger

from lightx2v.common.kvcache import KVCacheManager
from lightx2v.models.input_encoders.hf.seko_audio.audio_adapter import AudioAdapter, CausalAudioSlidingProcessor
from lightx2v.models.input_encoders.hf.seko_audio.audio_encoder import SekoAudioEncoderModel
from lightx2v.models.networks.wan.audio_model import WanAudioARModel, WanAudioModel
from lightx2v.models.runners.wan.wan_runner import WanRunner, build_wan_model_with_lora
from lightx2v.models.schedulers.wan.audio.scheduler import EulerScheduler, WanAudioARScheduler
from lightx2v.models.video_encoders.hf.wan.vae_2_2 import Wan2_2_VAE
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.async_vae import AsyncVAEChunkDecoder
from lightx2v.utils.audio_io import load_audio_file
from lightx2v.utils.envs import *
from lightx2v.utils.input_info import UNSET
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import find_torch_model_path, fixed_shape_resize, get_optimal_patched_size_with_sp, isotropic_crop_resize, load_weights, wan_vae_to_comfy
from lightx2v.utils.va_controller import VAController
from lightx2v_platform.base.global_var import AI_DEVICE

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io")


def resize_image(img, resize_mode="adaptive", bucket_shape=None, fixed_area=None, fixed_shape=None):
    assert resize_mode in ["adaptive", "keep_ratio_fixed_area", "fixed_min_area", "fixed_max_area", "fixed_shape", "fixed_min_side"]

    if resize_mode == "fixed_shape":
        assert fixed_shape is not None
        logger.info(f"[wan_audio] fixed_shape_resize fixed_height: {fixed_shape[0]}, fixed_width: {fixed_shape[1]}")
        return fixed_shape_resize(img, fixed_shape[0], fixed_shape[1])

    if bucket_shape is not None:
        """
        "adaptive_shape": {
            "0.667": [[480, 832], [544, 960], [720, 1280]],
            "1.500": [[832, 480], [960, 544], [1280, 720]],
            "1.000": [[480, 480], [576, 576], [704, 704], [960, 960]]
        }
        """
        bucket_config = {}
        for ratio, resolutions in bucket_shape.items():
            bucket_config[float(ratio)] = np.array(resolutions, dtype=np.int64)
        # logger.info(f"[wan_audio] use custom bucket_shape: {bucket_config}")
    else:
        bucket_config = {
            0.667: np.array([[480, 832], [544, 960], [720, 1280]], dtype=np.int64),
            1.500: np.array([[832, 480], [960, 544], [1280, 720]], dtype=np.int64),
            1.000: np.array([[480, 480], [576, 576], [704, 704], [960, 960]], dtype=np.int64),
        }
        # logger.info(f"[wan_audio] use default bucket_shape: {bucket_config}")

    ori_height = img.shape[-2]
    ori_weight = img.shape[-1]
    ori_ratio = ori_height / ori_weight

    if resize_mode == "adaptive":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        if ori_ratio < 1.0:
            target_h, target_w = 480, 832
        elif ori_ratio == 1.0:
            target_h, target_w = 480, 480
        else:
            target_h, target_w = 832, 480
        for resolution in bucket_config[closet_ratio]:
            if ori_height * ori_weight >= resolution[0] * resolution[1]:
                target_h, target_w = resolution
    elif resize_mode == "keep_ratio_fixed_area":
        area_in_pixels = 480 * 832
        if fixed_area == "480p":
            area_in_pixels = 480 * 832
        elif fixed_area == "720p":
            area_in_pixels = 720 * 1280
        else:
            area_in_pixels = 480 * 832
        target_h = round(np.sqrt(area_in_pixels * ori_ratio))
        target_w = round(np.sqrt(area_in_pixels / ori_ratio))
    elif resize_mode == "fixed_min_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][0]
    elif resize_mode == "fixed_min_side":
        min_side = 720
        if fixed_area == "720p":
            min_side = 720
        elif fixed_area == "480p":
            min_side = 480
        else:
            logger.warning(f"[wan_audio] fixed_area is not '480p' or '720p', using default 480p: {fixed_area}")
            min_side = 480
        if ori_ratio < 1.0:
            target_h = min_side
            target_w = round(target_h / ori_ratio)
        else:
            target_w = min_side
            target_h = round(target_w * ori_ratio)
    elif resize_mode == "fixed_max_area":
        aspect_ratios = np.array(np.array(list(bucket_config.keys())))
        closet_aspect_idx = np.argmin(np.abs(aspect_ratios - ori_ratio))
        closet_ratio = aspect_ratios[closet_aspect_idx]
        target_h, target_w = bucket_config[closet_ratio][-1]

    cropped_img = isotropic_crop_resize(img, (target_h, target_w))
    logger.info(f"[wan_audio] resize_image: {img.shape} -> {cropped_img.shape}, resize_mode: {resize_mode}, target_h: {target_h}, target_w: {target_w}")
    return cropped_img, target_h, target_w


@dataclass
class AudioSegment:
    """Data class for audio segment information"""

    audio_array: torch.Tensor
    start_frame: int
    end_frame: int


class FramePreprocessorTorchVersion:
    """Handles frame preprocessing including noise and masking"""

    def __init__(self, noise_mean: float = -3.0, noise_std: float = 0.5, mask_rate: float = 0.1):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.mask_rate = mask_rate

    def add_noise(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add noise to frames"""

        device = frames.device
        shape = frames.shape
        bs = 1 if len(shape) == 4 else shape[0]

        # Generate sigma values on the same device
        sigma = torch.normal(mean=self.noise_mean, std=self.noise_std, size=(bs,), device=device, generator=generator)
        sigma = torch.exp(sigma)

        for _ in range(1, len(shape)):
            sigma = sigma.unsqueeze(-1)

        # Generate noise on the same device
        noise = torch.randn(*shape, device=device, generator=generator) * sigma
        return frames + noise

    def add_mask(self, frames: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Add mask to frames"""

        device = frames.device
        h, w = frames.shape[-2:]

        # Generate mask on the same device
        mask = torch.rand(h, w, device=device, generator=generator) > self.mask_rate
        return frames * mask

    def process_prev_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Process previous frames with noise and masking"""
        frames = self.add_noise(frames, torch.Generator(device=frames.device))
        frames = self.add_mask(frames, torch.Generator(device=frames.device))
        return frames


class AudioProcessor:
    """Handles audio loading and segmentation"""

    def __init__(self, audio_sr: int = 16000, target_fps: int = 16):
        self.audio_sr = audio_sr
        self.target_fps = target_fps
        self.audio_frame_rate = audio_sr // target_fps

    def load_audio(self, audio_path: str):
        audio_array, ori_sr = load_audio_file(audio_path)
        audio_array = ta.functional.resample(audio_array.mean(0), orig_freq=ori_sr, new_freq=self.audio_sr)
        return audio_array

    def load_multi_person_audio(self, audio_paths: List[str]):
        audio_arrays = []
        max_len = 0

        for audio_path in audio_paths:
            audio_array = self.load_audio(audio_path)
            audio_arrays.append(audio_array)
            max_len = max(max_len, audio_array.numel())

        num_files = len(audio_arrays)
        padded = torch.zeros(num_files, max_len, dtype=torch.float32)

        for i, arr in enumerate(audio_arrays):
            length = arr.numel()
            padded[i, :length] = arr

        return padded

    def get_audio_range(self, start_frame: int, end_frame: int) -> Tuple[int, int]:
        """Calculate audio range for given frame range"""
        return round(start_frame * self.audio_frame_rate), round(end_frame * self.audio_frame_rate)

    def segment_audio(self, audio_array: torch.Tensor, expected_frames: int, max_num_frames: int, prev_frame_length: int = 5) -> List[AudioSegment]:
        """
        Segment audio based on frame requirements
        audio_array is (N, T) tensor
        """
        segments = []
        segments_idx = self.init_segments_idx(expected_frames, max_num_frames, prev_frame_length)

        audio_start, audio_end = self.get_audio_range(0, expected_frames)
        audio_array_ori = audio_array[:, audio_start:audio_end]

        for idx, (start_idx, end_idx) in enumerate(segments_idx):
            audio_start, audio_end = self.get_audio_range(start_idx, end_idx)
            audio_array = audio_array_ori[:, audio_start:audio_end]

            if idx < len(segments_idx) - 1:
                end_idx = segments_idx[idx + 1][0]
            else:  # for last segments
                if audio_array.shape[1] < audio_end - audio_start:
                    padding_len = audio_end - audio_start - audio_array.shape[1]
                    audio_array = F.pad(audio_array, (0, padding_len))
                    # Adjust end_idx to account for the frames added by padding
                    end_idx = end_idx - padding_len // self.audio_frame_rate

            segments.append(AudioSegment(audio_array, start_idx, end_idx))
        del audio_array, audio_array_ori
        return segments

    def init_segments_idx(self, total_frame: int, clip_frame: int = 81, overlap_frame: int = 5) -> list[tuple[int, int, int]]:
        """Initialize segment indices with overlap"""
        start_end_list = []
        min_frame = clip_frame
        for start in range(0, total_frame, clip_frame - overlap_frame):
            is_last = start + clip_frame >= total_frame
            end = min(start + clip_frame, total_frame)
            if end - start < min_frame:
                end = start + min_frame
            if ((end - start) - 1) % 4 != 0:
                end = start + (((end - start) - 1) // 4) * 4 + 1
            start_end_list.append((start, end))
            if is_last:
                break
        return start_end_list


def load_image(image: Union[str, Image.Image], to_rgb: bool = True) -> Image.Image:
    _image = image
    if isinstance(image, str):
        if os.path.isfile(image):
            _image = Image.open(image)
        else:
            raise ValueError(f"Incorrect path. {image} is not a valid path.")
    # orientation transpose
    _image = ImageOps.exif_transpose(_image)
    # convert color space to sRGB
    icc_profile = _image.info.get("icc_profile")
    if icc_profile:
        srgb_profile = ImageCms.createProfile("sRGB")
        input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        _image = ImageCms.profileToProfile(_image, input_profile, srgb_profile)
    # convert to "RGB"
    if to_rgb:
        _image = _image.convert("RGB")

    return _image


@RUNNER_REGISTER("seko_talk")
class WanAudioRunner(WanRunner):  # type:ignore
    def __init__(self, config):
        super().__init__(config)
        self.name = self.config.get("name", "WanAudioRunner")
        self.task = self.config.get("task", "i2v")
        self.prev_frame_length = self.config.get("prev_frame_length", 5)
        self.video_duration = self.config.get("video_duration", 5)

        self.frame_preprocessor = FramePreprocessorTorchVersion()

    def init_scheduler(self):
        """Initialize consistency model scheduler"""
        self.scheduler = EulerScheduler(self.config)

    def read_audio_input(self, audio_path):
        """Read audio input - handles both single and multi-person scenarios"""
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)

        if not isinstance(audio_path, str):
            return [], 0, None, 0

        # Get audio files from person objects or legacy format
        audio_files, mask_files = self.get_audio_files_from_audio_path(audio_path)

        # Load audio based on single or multi-person mode
        if len(audio_files) == 1:
            audio_array = self._audio_processor.load_audio(audio_files[0])
            audio_array = audio_array.unsqueeze(0)  # Add batch dimension for consistency
        else:
            audio_array = self._audio_processor.load_multi_person_audio(audio_files)

        audio_len = int(audio_array.shape[1] / audio_sr * target_fps)
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_audio_len.observe(audio_len)

        expected_frames = min(max(1, int(self.video_duration * target_fps)), audio_len)
        if expected_frames < int(self.video_duration * target_fps):
            logger.warning(f"Input video duration is greater than actual audio duration, using audio duration instead: audio_duration={audio_len / target_fps}, video_duration={self.video_duration}")

        # Segment audio (CLI / input_info wins over config_json; target_video_length is not merged into config)
        target_video_length = self.config.get("target_video_length", 81)
        ii = getattr(self, "input_info", None)
        if ii is not None and hasattr(ii, "target_video_length"):
            tvl = ii.target_video_length
            if tvl is not None and tvl is not UNSET and tvl > 0:
                target_video_length = tvl
        audio_segments = self._audio_processor.segment_audio(audio_array, expected_frames, target_video_length, self.prev_frame_length)

        # Mask latent for multi-person s2v
        if mask_files is not None:
            mask_latents = [self.process_single_mask(mask_file) for mask_file in mask_files]
            mask_latents = torch.cat(mask_latents, dim=0)
        else:
            mask_latents = None

        return audio_segments, expected_frames, mask_latents, len(audio_files)

    def get_audio_files_from_audio_path(self, audio_path):
        if os.path.isdir(audio_path):
            audio_files = []
            mask_files = []
            audio_config_path = os.path.join(audio_path, "config.json")
            assert os.path.exists(audio_config_path), "config.json not found in audio_path"
            with open(audio_config_path, "r") as f:
                audio_config = json.load(f)
            for talk_object in audio_config["talk_objects"]:
                audio_files.append(os.path.join(audio_path, talk_object["audio"]))
                mask_files.append(os.path.join(audio_path, talk_object["mask"]))
        else:
            audio_files = [audio_path]
            mask_files = None

        return audio_files, mask_files

    def process_single_mask(self, mask_file):
        mask_img = load_image(mask_file)
        mask_img = TF.to_tensor(mask_img).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)

        if mask_img.shape[1] == 3:  # If it is an RGB three-channel image
            mask_img = mask_img[:, :1]  # Only take the first channel

        mask_img, h, w = resize_image(
            mask_img,
            resize_mode=self.config.get("resize_mode", "adaptive"),
            bucket_shape=self.config.get("bucket_shape", None),
            fixed_area=self.config.get("fixed_area", None),
            fixed_shape=self.config.get("fixed_shape", None),
        )

        mask_latent = torch.nn.functional.interpolate(
            mask_img,  # (1, 1, H, W)
            size=(h // 16, w // 16),
            mode="bicubic",
        )

        mask_latent = (mask_latent > 0).to(torch.int8)
        return mask_latent

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            ref_img = img_path
        else:
            ref_img = load_image(img_path)
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(0).to(AI_DEVICE)

        ref_img, h, w = resize_image(
            ref_img,
            resize_mode=self.config.get("resize_mode", "adaptive"),
            bucket_shape=self.config.get("bucket_shape", None),
            fixed_area=self.config.get("fixed_area", None),
            fixed_shape=self.config.get("fixed_shape", None),
        )
        logger.info(f"[wan_audio] resize_image target_h: {h}, target_w: {w}")
        patched_h = h // self.config["vae_stride"][1] // self.config["patch_size"][1]
        patched_w = w // self.config["vae_stride"][2] // self.config["patch_size"][2]

        patched_h, patched_w = get_optimal_patched_size_with_sp(patched_h, patched_w, 1)

        latent_h = patched_h * self.config["patch_size"][1]
        latent_w = patched_w * self.config["patch_size"][2]

        if hasattr(self.input_info, "target_video_length") and self.input_info.target_video_length is not None and self.input_info.target_video_length > 0:
            target_video_length = self.input_info.target_video_length
            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w, target_video_length)
        else:
            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)

        target_shape = [latent_h * self.config["vae_stride"][1], latent_w * self.config["vae_stride"][2]]

        logger.info(f"[wan_audio] target_h: {target_shape[0]}, target_w: {target_shape[1]}, latent_h: {latent_h}, latent_w: {latent_w}")

        ref_img = torch.nn.functional.interpolate(ref_img, size=(target_shape[0], target_shape[1]), mode="bicubic")
        return ref_img, latent_shape, target_shape

    @ProfilingContext4DebugL1(
        "Run Image Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_img_encode_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def run_image_encoder(self, first_frame, last_frame=None):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.image_encoder = self.load_image_encoder()
        clip_encoder_out = self.image_encoder.visual([first_frame]).squeeze(0).to(GET_DTYPE()) if self.config.get("use_image_encoder", True) else None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.image_encoder
            torch.cuda.empty_cache()
            gc.collect()
        return clip_encoder_out

    @ProfilingContext4DebugL1(
        "Run VAE Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
        metrics_labels=["WanAudioRunner"],
    )
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

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_s2v(self):
        img, latent_shape, target_shape = self.read_image_input(self.input_info.image_path)
        if self.config.get("f2v_process", False):
            self.ref_img = img
        self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info
        self.input_info.target_shape = target_shape  # Important: set target_shape in input_info
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)

        audio_segments, expected_frames, person_mask_latens, audio_num = self.read_audio_input(self.input_info.audio_path)
        self.input_info.audio_num = audio_num
        self.input_info.with_mask = person_mask_latens is not None
        text_encoder_output = self.run_text_encoder(self.input_info)
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
            "person_mask_latens": person_mask_latens,
        }

    @ProfilingContext4DebugL2("Run Encoders Static (RS2V)")
    def _run_input_encoder_local_rs2v_static(self):
        img, latent_shape, target_shape = self.read_image_input(self.input_info.image_path)
        if self.config.get("f2v_process", False):
            self.ref_img = img
        self.input_info.latent_shape = latent_shape
        self.input_info.target_shape = target_shape
        clip_encoder_out = self.run_image_encoder(img) if self.config.get("use_image_encoder", True) else None
        vae_encode_out = self.run_vae_encoder(img)
        text_encoder_output = self.run_text_encoder(self.input_info)

        self.inputs_static = {
            "text_encoder_output": text_encoder_output,
            "image_encoder_output": {
                "clip_encoder_out": clip_encoder_out,
                "vae_encoder_out": vae_encode_out,
            },
        }
        return self.inputs_static

    @ProfilingContext4DebugL2("Run Encoders Dynamic (RS2V)")
    def _run_input_encoder_local_rs2v_dynamic(self):
        if not hasattr(self, "inputs_static") or self.inputs_static is None:
            self._run_input_encoder_local_rs2v_static()

        inputs = self.inputs_static.copy()

        person_mask_latens = getattr(self.input_info, "person_mask_latens", None)
        self.input_info.with_mask = person_mask_latens is not None

        inputs.update(
            {
                "person_mask_latens": person_mask_latens,
            }
        )

        torch.cuda.empty_cache()
        gc.collect()
        return inputs

    def prepare_prev_latents(self, prev_video: Optional[torch.Tensor], prev_frame_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare previous latents for conditioning"""
        dtype = GET_DTYPE()
        tgt_h, tgt_w = self.input_info.target_shape[0], self.input_info.target_shape[1]
        if hasattr(self.input_info, "target_video_length") and self.input_info.target_video_length is not None and self.input_info.target_video_length > 0:
            target_video_length = self.input_info.target_video_length
        else:
            target_video_length = self.config["target_video_length"]
        prev_frames = torch.zeros((1, 3, target_video_length, tgt_h, tgt_w), device=AI_DEVICE)

        if prev_video is not None:
            # Extract and process last frames
            last_frames = prev_video[:, :, -prev_frame_length:].clone().to(AI_DEVICE)
            if self.config["model_cls"] != "wan2.2_audio" and not self.config.get("f2v_process", False):
                last_frames = self.frame_preprocessor.process_prev_frames(last_frames)
            prev_frames[:, :, :prev_frame_length] = last_frames
            prev_len = (prev_frame_length - 1) // 4 + 1
        else:
            prev_len = 0

        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_encoder = self.load_vae_encoder()

        _, nframe, height, width = self.model.scheduler.latents.shape
        with ProfilingContext4DebugL1(
            "vae_encoder in init run segment",
            recorder_mode=GET_RECORDER_MODE(),
            metrics_func=monitor_cli.lightx2v_run_vae_encoder_pre_latent_duration,
            metrics_labels=["WanAudioRunner"],
        ):
            if self.config["model_cls"] == "wan2.2_audio":
                if prev_video is not None:
                    prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))
                else:
                    prev_latents = None
                prev_mask = self.model.scheduler.mask
            else:
                prev_latents = self.vae_encoder.encode(prev_frames.to(dtype))

            frames_n = (nframe - 1) * 4 + 1
            prev_mask = torch.ones((1, frames_n, height, width), device=AI_DEVICE, dtype=dtype)
            prev_frame_len = max((prev_len - 1) * 4 + 1, 0)
            prev_mask[:, prev_frame_len:] = 0
            prev_mask = self._wan_mask_rearrange(prev_mask)

        if prev_latents is not None:
            if prev_latents.shape[-2:] != (height, width):
                logger.warning(f"Size mismatch: prev_latents {prev_latents.shape} vs scheduler latents (H={height}, W={width}). Config tgt_h={tgt_h}, tgt_w={tgt_w}")
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
        return mask.transpose(0, 1).contiguous()

    def get_video_segment_num(self):
        self.video_segment_num = len(self.inputs["audio_segments"])

    def init_run(self):
        super().init_run()
        self.scheduler.set_audio_adapter(self.audio_adapter)
        if self.config.get("f2v_process", False):
            self.prev_video = self.ref_img.unsqueeze(2)
        else:
            self.prev_video = None
        if self.input_info.return_result_tensor:
            self.gen_video_final = torch.zeros((self.inputs["expected_frames"], self.input_info.target_shape[0], self.input_info.target_shape[1], 3), dtype=torch.float32, device="cpu")
            self.cut_audio_final = torch.zeros((self.inputs["expected_frames"] * self._audio_processor.audio_frame_rate), dtype=torch.float32, device="cpu")
        else:
            self.gen_video_final = None
            self.cut_audio_final = None

    @ProfilingContext4DebugL1(
        "Init run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_init_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def init_run_segment(self, segment_idx, audio_array=None):
        self.segment_idx = segment_idx
        if audio_array is not None:
            end_idx = audio_array.shape[0] // self._audio_processor.audio_frame_rate - self.prev_frame_length
            audio_tensor = torch.Tensor(audio_array).float().unsqueeze(0)
            self.segment = AudioSegment(audio_tensor, 0, end_idx)
        else:
            self.segment = self.inputs["audio_segments"][segment_idx]

        self.input_info.seed = self.input_info.seed + segment_idx
        torch.manual_seed(self.input_info.seed)
        # logger.info(f"Processing segment {segment_idx + 1}/{self.video_segment_num}, seed: {self.config.seed}")

        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not hasattr(self, "audio_encoder"):
            self.audio_encoder = self.load_audio_encoder()

        features_list = []
        for i in range(self.segment.audio_array.shape[0]):
            feat = self.audio_encoder.infer(self.segment.audio_array[i])
            feat = self.audio_adapter.forward_audio_proj(feat, self.model.scheduler.latents.shape[1])
            features_list.append(feat.squeeze(0))
        audio_features = torch.stack(features_list, dim=0)

        self.inputs["audio_encoder_output"] = audio_features
        self.inputs["previmg_encoder_output"] = self.prepare_prev_latents(self.prev_video, prev_frame_length=self.prev_frame_length)

        # Reset scheduler for non-first segments
        if segment_idx > 0:
            self.model.scheduler.reset(self.input_info.seed, self.input_info.latent_shape, self.inputs["previmg_encoder_output"])

    @ProfilingContext4DebugL1(
        "End run segment",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_end_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def end_run_segment(self, segment_idx, valid_duration=1e9):
        self.gen_video = torch.clamp(self.gen_video, -1, 1).to(torch.float)
        useful_length = self.segment.end_frame - self.segment.start_frame
        video_seg = self.gen_video[:, :, :useful_length].cpu()
        audio_seg = self.segment.audio_array[:, : useful_length * self._audio_processor.audio_frame_rate]
        audio_seg = audio_seg.sum(dim=0)  # Multiple audio tracks, mixed into one track
        video_seg = wan_vae_to_comfy(video_seg)

        # [Warning] Need check whether video segment interpolation works...
        if "video_frame_interpolation" in self.config and self.vfi_model is not None:
            target_fps = self.config["video_frame_interpolation"]["target_fps"]
            logger.info(f"Interpolating frames from {self.config.get('fps', 16)} to {target_fps}")
            video_seg = self.vfi_model.interpolate_frames(
                video_seg,
                source_fps=self.config.get("fps", 16),
                target_fps=target_fps,
            )

        if "video_super_resolution" in self.config and self.vsr_model is not None:
            # logger.info(f"Applying video super resolution with scale {self.config['video_super_resolution']['scale']}")
            video_seg = self.vsr_model.super_resolve_frames(
                video_seg,
                seed=self.config["video_super_resolution"]["seed"],
                scale=self.config["video_super_resolution"]["scale"],
            )

        if self.va_controller.recorder is not None:
            self.va_controller.pub_livestream(video_seg, audio_seg, self.gen_video[:, :, :useful_length], valid_duration=valid_duration)
        elif self.input_info.return_result_tensor:
            self.gen_video_final[self.segment.start_frame : self.segment.end_frame].copy_(video_seg)
            self.cut_audio_final[self.segment.start_frame * self._audio_processor.audio_frame_rate : self.segment.end_frame * self._audio_processor.audio_frame_rate].copy_(audio_seg)

        # Update prev_video for next iteration
        self.prev_video = self.gen_video

        del video_seg, audio_seg
        torch.cuda.empty_cache()

    @ProfilingContext4DebugL1(
        "End run segment stream",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_end_run_segment_duration,
        metrics_labels=["WanAudioRunner"],
    )
    def end_run_segment_stream(self, latents, valid_duration=1e9):
        valid_length = self.segment.end_frame - self.segment.start_frame
        frame_segments = []
        frame_idx = 0

        # frame_segment: 1*C*1*H*W, 1*C*4*H*W, 1*C*4*H*W, ...
        for origin_seg in self.run_vae_decoder_stream(latents):
            origin_seg = torch.clamp(origin_seg, -1, 1).to(torch.float)
            valid_T = min(valid_length - frame_idx, origin_seg.shape[2])

            video_seg = wan_vae_to_comfy(origin_seg[:, :, :valid_T].cpu())
            audio_start = frame_idx * self._audio_processor.audio_frame_rate
            audio_end = (frame_idx + valid_T) * self._audio_processor.audio_frame_rate
            audio_seg = self.segment.audio_array[:, audio_start:audio_end].sum(dim=0)

            if self.va_controller.recorder is not None:
                self.va_controller.pub_livestream(video_seg, audio_seg, origin_seg[:, :, :valid_T], valid_duration=valid_duration)

            frame_segments.append(origin_seg)
            frame_idx += valid_T
            cur_duration = valid_T / self.config.get("fps", 16)
            valid_duration = max(valid_duration - cur_duration, 0)
            del video_seg, audio_seg

        # Update prev_video for next iteration
        self.prev_video = torch.cat(frame_segments, dim=2)
        torch.cuda.empty_cache()

    def run_main(self):
        try:
            self.va_controller = None
            self.va_controller = VAController(self)
            logger.info(f"init va_recorder: {self.va_controller.recorder} and va_reader: {self.va_controller.reader}")

            # fixed audio segments inputs
            if self.va_controller.reader is None:
                return super().run_main()

            self.va_controller.start()
            self.init_run()
            if self.config.get("compile", False) and hasattr(self.model, "comple"):
                self.model.select_graph_for_compile(self.input_info)
            # steam audio input, video segment num is unlimited
            self.video_segment_num = 1000000
            segment_idx = 0
            fail_count, max_fail_count = 0, 10
            self.va_controller.before_control()

            while True:
                with ProfilingContext4DebugL1(f"stream segment get audio segment {segment_idx}"):
                    control = self.va_controller.next_control()
                    if control.action == "blank_to_voice":
                        self.prev_video = control.data
                    elif control.action == "switch_image":
                        self.input_info.image_path = control.data
                        self.inputs = self.run_input_encoder()
                        if self.config.get("f2v_process", False):
                            self.prev_video = self.ref_img.unsqueeze(2)
                        else:
                            self.prev_video = None
                    elif control.action == "wait":
                        time.sleep(0.01)
                        continue

                    audio_array, valid_duration = self.va_controller.reader.get_audio_segment()
                    if audio_array is None:
                        fail_count += 1
                        logger.warning(f"Failed to get audio chunk {fail_count} times")
                        if fail_count > max_fail_count:
                            raise Exception(f"Failed to get audio chunk {fail_count} times, stop reader")
                        continue

                with ProfilingContext4DebugL1(f"stream segment end2end {segment_idx}"):
                    try:
                        # reset pause signal
                        self.pause_signal = False
                        self.can_pause = valid_duration <= 1e-5
                        self.init_run_segment(segment_idx, audio_array)
                        self.check_stop()
                        latents = self.run_segment(segment_idx)
                        self.check_stop()
                        if self.config.get("use_stream_vae", False):
                            self.end_run_segment_stream(latents, valid_duration=valid_duration)
                        else:
                            self.gen_video = self.run_vae_decoder(latents)
                            self.check_stop()
                            self.end_run_segment(segment_idx, valid_duration=valid_duration)
                        segment_idx += 1
                        fail_count = 0
                    except Exception as e:
                        if "pause_signal, pause running" in str(e):
                            logger.warning(f"model infer audio pause: {e}, should continue")
                        else:
                            raise
        finally:
            if hasattr(self.model, "inputs"):
                self.end_run()
            if self.va_controller is not None:
                self.va_controller.clear()
                self.va_controller = None

    @ProfilingContext4DebugL1("Process after vae decoder")
    def process_images_after_vae_decoder(self):
        if self.input_info.return_result_tensor:
            audio_waveform = self.cut_audio_final.unsqueeze(0).unsqueeze(0)
            comfyui_audio = {"waveform": audio_waveform, "sample_rate": self._audio_processor.audio_sr}
            return {"video": self.gen_video_final, "audio": comfyui_audio}
        return {"video": None, "audio": None}

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanAudioModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanAudioModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def load_audio_encoder(self):
        audio_encoder_path = self.config.get("audio_encoder_path", os.path.join(self.config["model_path"], "TencentGameMate-chinese-hubert-large"))
        audio_encoder_offload = self.config.get("audio_encoder_cpu_offload", self.config.get("cpu_offload", False))
        dummy_model = self.config.get("dummy_model", False)
        model = SekoAudioEncoderModel(audio_encoder_path, self.config["audio_sr"], audio_encoder_offload, dummy_model=dummy_model)
        return model

    def load_audio_adapter(self):
        audio_adapter_offload = self.config.get("audio_adapter_cpu_offload", self.config.get("cpu_offload", False))
        if audio_adapter_offload:
            device = torch.device("cpu")
        else:
            device = torch.device(AI_DEVICE)
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
        if not self.config.get("dummy_model", False):
            load_from_rank0 = self.config.get("load_from_rank0", False)
            weights_dict = load_weights(self.config["adapter_model_path"], cpu_offload=audio_adapter_offload, remove_key="ca", load_from_rank0=load_from_rank0)
            audio_adapter.load_state_dict(weights_dict, strict=False)
        else:
            logger.info("[DummyModel] Skipping audio adapter weight loading, using random init")
        return audio_adapter.to(dtype=GET_DTYPE())

    def load_model(self):
        super().load_model()
        with ProfilingContext4DebugL2("Load audio encoder and adapter"):
            self.audio_encoder = self.load_audio_encoder()
            self.audio_adapter = self.load_audio_adapter()

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w, target_video_length=None):
        target_video_length = target_video_length if target_video_length is not None and target_video_length > 0 else self.config["target_video_length"]
        latent_shape = [
            self.config.get("num_channels_latents", 16),
            (target_video_length - 1) // self.config["vae_stride"][0] + 1,
            latent_h,
            latent_w,
        ]
        return latent_shape

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_cached_decoder_withflag(self, latents, is_first: bool, is_last: bool):
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.vae_decoder = self.load_vae_decoder()
        images = self.vae_decoder.cached_decode_withflag(latents.to(GET_DTYPE()), is_first, is_last)
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            del self.vae_decoder
            torch.cuda.empty_cache()
            gc.collect()
        return images

    def run_clip(self):
        infer_steps = self.model.scheduler.infer_steps
        for step_index in range(infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(step_index=step_index)
            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.model.scheduler.step_post()

        return self.model.scheduler.latents

    def run_clip_main(self):
        self.scheduler.set_audio_adapter(self.audio_adapter)

        self.model.scheduler.prepare(
            seed=self.input_info.seed, latent_shape=self.input_info.latent_shape, infer_steps=self.input_info.infer_steps, image_encoder_output=self.inputs["image_encoder_output"]
        )

        if self.config.get("model_cls") == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            self.inputs["image_encoder_output"]["vae_encoder_out"] = None

        self.input_info.seed = self.input_info.seed
        torch.manual_seed(self.input_info.seed)

        if self.config.get("f2v_process", False):
            if self.input_info.overlap_frame is None:
                self.input_info.overlap_frame = self.ref_img.unsqueeze(2)

        # 处理音频输入
        audio_clip = self.input_info.audio_clip
        features_list = []
        for i in range(audio_clip.shape[0]):
            feat = self.audio_encoder.infer(audio_clip[i])
            feat = self.audio_adapter.forward_audio_proj(feat, self.model.scheduler.latents.shape[1])
            features_list.append(feat.squeeze(0))
        audio_features = torch.stack(features_list, dim=0)

        self.inputs["audio_encoder_output"] = audio_features
        # 处理前一帧图像或latent输入
        if self.task in ["rs2v"]:
            self.inputs["previmg_encoder_output"] = {"prev_latents": self.input_info.overlap_latent}
            self.inputs["ref_state"] = self.input_info.ref_state
        else:
            self.inputs["previmg_encoder_output"] = self.prepare_prev_latents(self.input_info.overlap_frame, prev_frame_length=self.prev_frame_length)
        # 执行dit推理
        latents = self.run_clip()
        # 运行vae decoder
        if self.task in ["rs2v"]:
            gen_video = self.run_vae_cached_decoder_withflag(latents, self.input_info.is_first, self.input_info.is_last)
        else:
            gen_video = self.run_vae_decoder(latents)

        return gen_video, audio_clip, latents

    def run_clip_pipeline(self, input_info):
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        return self.run_clip_main()


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
            vae_device = torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
            "dummy_model": self.config.get("dummy_model", False),
        }
        vae_decoder = Wan2_2_VAE(**vae_config)
        return vae_decoder

    def load_vae_encoder(self):
        # offload config
        vae_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload"))
        if vae_offload:
            vae_device = torch.device("cpu")
        else:
            vae_device = torch.device(AI_DEVICE)
        vae_config = {
            "vae_path": find_torch_model_path(self.config, "vae_path", "Wan2.2_VAE.pth"),
            "device": vae_device,
            "cpu_offload": vae_offload,
            "offload_cache": self.config.get("vae_offload_cache", False),
            "dummy_model": self.config.get("dummy_model", False),
        }
        if self.config.task not in ["i2v", "s2v", "rs2v"]:
            return None
        else:
            return Wan2_2_VAE(**vae_config)

    def load_vae(self):
        vae_encoder = self.load_vae_encoder()
        vae_decoder = self.load_vae_decoder()
        return vae_encoder, vae_decoder


@RUNNER_REGISTER("seko_talk_ar")
class WanAudioARRunner(WanAudioRunner):
    def __init__(self, config):
        super().__init__(config)
        self.audio_sliding_processor = CausalAudioSlidingProcessor(
            audio_window=self.config.get("audio_window", 1.0),
            look_ahead=self.config.get("look_ahead", 0.0),
            audio_feat_window_neighbor_frame=self.config.get("audio_feat_window_neighbor_frame", 0),
            video_fps=self.config.get("target_fps", 16),
            audio_sr=self.config.get("audio_sr", 16000),
            audio_feat_fps=self.config.get("audio_feat_fps", 50),
        )

    def init_scheduler(self):
        self.scheduler = WanAudioARScheduler(self.config)

    def load_transformer(self):
        wan_model_kwargs = {"model_path": self.config["model_path"], "config": self.config, "device": self.init_device}
        lora_configs = self.config.get("lora_configs")
        if not lora_configs:
            model = WanAudioARModel(**wan_model_kwargs)
        else:
            model = build_wan_model_with_lora(WanAudioARModel, self.config, wan_model_kwargs, lora_configs, model_type="wan2.1")
        return model

    def load_audio_adapter(self):
        audio_adapter_offload = self.config.get("audio_adapter_cpu_offload", self.config.get("cpu_offload", False))
        device = torch.device("cpu") if audio_adapter_offload else torch.device(AI_DEVICE)
        self.config.setdefault("audio_num_tokens", self.config.get("num_audio_tokens", 32))
        projection_dim = self.config.get("audio_projection_dim", self.config.get("audio_feature_dim", 1024))
        audio_adapter = AudioAdapter(
            attention_head_dim=self.config["dim"] // self.config["num_heads"],
            num_attention_heads=self.config["num_heads"],
            base_num_layers=self.config["num_layers"],
            interval=1,
            audio_feature_dim=self.config.get("audio_feature_dim", 1024),
            num_tokens=self.config["audio_num_tokens"],
            time_freq_dim=256,
            projection_transformer_layers=self.config.get("projection_transformer_layers", 4),
            mlp_dims=(1024, 1024, self.config["audio_num_tokens"] * projection_dim),
            quantized=self.config.get("adapter_quantized", False),
            quant_scheme=self.config.get("adapter_quant_scheme", None),
            cpu_offload=audio_adapter_offload,
            causal_projection=True,
            projection_dim=projection_dim,
        )
        audio_adapter.to(device)
        if not self.config.get("dummy_model", False):
            load_from_rank0 = self.config.get("load_from_rank0", False)
            weights_dict = load_weights(self.config["adapter_model_path"], cpu_offload=audio_adapter_offload, remove_key="ca", load_from_rank0=load_from_rank0)
            audio_adapter.load_state_dict(weights_dict, strict=False)
        else:
            logger.info("[DummyModel] Skipping causal audio adapter weight loading, using random init")
        return audio_adapter.to(dtype=GET_DTYPE())

    def read_image_input(self, img_path):
        if isinstance(img_path, Image.Image):
            ref_img = img_path
        else:
            ref_img = load_image(img_path)

        input_target_shape = getattr(self.input_info, "target_shape", None)
        if input_target_shape is not None and input_target_shape is not UNSET and len(input_target_shape) >= 2:
            target_h = int(input_target_shape[0] or self.config.get("target_height", 480))
            target_w = int(input_target_shape[1] or self.config.get("target_width", 832))
        else:
            target_h = int(self.config.get("target_height", 480))
            target_w = int(self.config.get("target_width", 832))
        target_shape = [target_h, target_w]

        ref_img = torch.from_numpy(np.array(ref_img))
        if ref_img.ndim == 2:
            ref_img = ref_img.unsqueeze(-1).expand(-1, -1, 3)
        ref_img = rearrange(ref_img[..., :3], "H W C -> 1 C H W").contiguous().to(AI_DEVICE)
        ref_img = isotropic_crop_resize(ref_img, (target_h, target_w))
        ref_img = (ref_img - 127.5) / 127.5

        latent_h = target_h // self.config["vae_stride"][1]
        latent_w = target_w // self.config["vae_stride"][2]
        target_video_length = getattr(self.input_info, "target_video_length", None)
        if target_video_length is not None and target_video_length is not UNSET and target_video_length > 0:
            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w, target_video_length)
        else:
            latent_shape = self.get_latent_shape_with_lat_hw(latent_h, latent_w)

        logger.info(f"[wan_audio_ar] image resize target_h: {target_h}, target_w: {target_w}, latent_h: {latent_h}, latent_w: {latent_w}")
        return ref_img, latent_shape, target_shape

    def read_audio_input(self, audio_path):
        audio_sr = self.config.get("audio_sr", 16000)
        target_fps = self.config.get("target_fps", 16)
        self._audio_processor = AudioProcessor(audio_sr, target_fps)

        if not isinstance(audio_path, str):
            return [], 0, None, 0

        audio_files, mask_files = self.get_audio_files_from_audio_path(audio_path)
        if len(audio_files) == 1:
            audio_array = self._audio_processor.load_audio(audio_files[0]).unsqueeze(0)
        else:
            audio_array = self._audio_processor.load_multi_person_audio(audio_files)

        audio_len = int(audio_array.shape[1] / audio_sr * target_fps)
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_input_audio_len.observe(audio_len)

        expected_frames = min(max(1, int(self.video_duration * target_fps)), audio_len)
        if expected_frames < int(self.video_duration * target_fps):
            logger.warning(f"Input video duration is greater than actual audio duration, using audio duration instead: audio_duration={audio_len / target_fps}, video_duration={self.video_duration}")

        audio_segments = [AudioSegment(audio_array, 0, expected_frames)]

        if mask_files is not None:
            mask_latents = [self.process_single_mask(mask_file) for mask_file in mask_files]
            mask_latents = torch.cat(mask_latents, dim=0)
        else:
            mask_latents = None

        return audio_segments, expected_frames, mask_latents, len(audio_files)

    def init_run(self):
        self.gen_video_final = None
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            self.model = self.load_transformer()
            self.model.set_scheduler(self.scheduler)
        self.scheduler.set_audio_adapter(self.audio_adapter)
        self.va_controller = VAController(self)
        if self.input_info.return_result_tensor:
            self.gen_video_final = torch.zeros((self.inputs["expected_frames"], self.input_info.target_shape[0], self.input_info.target_shape[1], 3), dtype=torch.float32, device="cpu")
            self.cut_audio_final = torch.zeros((self.inputs["expected_frames"] * self._audio_processor.audio_frame_rate), dtype=torch.float32, device="cpu")
        else:
            self.cut_audio_final = None

    def get_video_segment_num(self):
        self.video_segment_num = self.model.scheduler.num_chunks

    def _encode_audio_for_ar(self):
        self.audio_segment = self.inputs["audio_segments"][0]
        self.segment = self.audio_segment
        torch.manual_seed(self.input_info.seed)
        if (self.config.get("lazy_load", False) or self.config.get("unload_modules", False)) and not hasattr(self, "audio_encoder"):
            self.audio_encoder = self.load_audio_encoder()

        features_list = []
        for i in range(self.audio_segment.audio_array.shape[0]):
            feat = self.audio_encoder.infer_causal(self.audio_segment.audio_array[i], self.audio_sliding_processor)
            feat = self.audio_adapter.forward_audio_proj(feat, feat.shape[1])
            features_list.append(feat.squeeze(0))
        self.inputs["audio_encoder_output"] = torch.stack(features_list, dim=0)
        self.inputs["previmg_encoder_output"] = {"prev_latents": None, "prev_mask": None, "prev_len": 0}

    def init_run_segment(self, segment_idx):
        self.segment_idx = segment_idx

    @ProfilingContext4DebugL1(
        "Prefill reference kv",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_prefill_reference_kv_duration,
        metrics_labels=["WanAudioARRunner"],
    )
    def prefill_reference_kv(self):
        ref_latents = self.inputs["image_encoder_output"]["vae_encoder_out"]
        ref_frames = 1 if ref_latents is None else int(ref_latents.shape[1])
        self.inputs["_ar_ref_prefill"] = True
        try:
            for step_index in range(self.model.scheduler.infer_steps):
                self.model.kv_cache_manager.current_step = step_index
                self.model.scheduler.step_pre_ref(step_index, ref_frames)
                self.model.infer(self.inputs)
        finally:
            self.inputs.pop("_ar_ref_prefill", None)

    def run_segment(self, segment_idx=0):
        infer_steps = self.model.scheduler.infer_steps
        chunk_size = int(self.model.scheduler.chunk_size)
        start = segment_idx * chunk_size
        end = start + chunk_size
        self.model.scheduler.set_timesteps(infer_steps, device=AI_DEVICE)
        xt = self.model.scheduler.noise[:, start:end].to(AI_DEVICE)
        for step_index in range(infer_steps):
            logger.info(f"==> chunk: {segment_idx + 1} / {self.video_segment_num}, step_index: {step_index + 1} / {infer_steps}")
            self.model.kv_cache_manager.current_step = step_index
            with ProfilingContext4DebugL1("step_pre"):
                self.model.scheduler.step_pre(segment_idx, step_index, xt)
            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(self.inputs)
            with ProfilingContext4DebugL1("step_post"):
                xt = self.model.scheduler.step_post(xt)
            if self.progress_callback:
                current_step = (segment_idx * infer_steps) + step_index + 1
                total_all_steps = self.video_segment_num * infer_steps
                self.progress_callback((current_step / total_all_steps) * 100, 100)
        return xt

    def decode_segment_latents(self, segment_idx: int, segment_latents: torch.Tensor) -> torch.Tensor:
        is_first = segment_idx == 0
        is_last = segment_idx == self.video_segment_num - 1
        return self.vae_decoder.cached_decode_withflag(segment_latents.to(GET_DTYPE()), is_first, is_last)

    @ProfilingContext4DebugL1("init kv cache manager")
    def init_kv_cache_manager(self):
        ref_latents = self.inputs["image_encoder_output"]["vae_encoder_out"]
        ref_num_frames = 0 if ref_latents is None else int(ref_latents.shape[1])
        kv_mgr = getattr(self.model, "kv_cache_manager", None)
        if kv_mgr is None:
            kv_mgr = KVCacheManager(config=self.config, device=torch.device(AI_DEVICE), sp_group=self.model.seq_p_group)
            self.model.kv_cache_manager = kv_mgr
        kv_mgr.ar_config = dict(self.config.get("ar_config", {}))
        kv_mgr._create_kv_caches(self.input_info.latent_shape, ref_num_frames=ref_num_frames)
        self.model.transformer_infer.kv_cache_manager = kv_mgr
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            sp_group = getattr(self.model, "seq_p_group", None)
            if sp_group is not None or torch.distributed.get_world_size() > 1:
                torch.distributed.barrier(group=sp_group)

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        try:
            self.init_run()
            if self.config.get("compile", False) and hasattr(self.model, "comple"):
                self.model.select_graph_for_compile(self.input_info)
            logger.info("start ar audio generation")
            self.check_stop()
            self._encode_audio_for_ar()
            latent_shape = list(self.input_info.latent_shape)
            latent_shape[1] = self.inputs["audio_encoder_output"].shape[1]
            latent_shape[1] = (latent_shape[1] // self.config["ar_config"]["num_frame_per_chunk"]) * self.config["ar_config"]["num_frame_per_chunk"]
            self.input_info.latent_shape = latent_shape
            self.model.scheduler.prepare(seed=self.input_info.seed, latent_shape=latent_shape, infer_steps=self.config.get("infer_steps"))
            self.get_video_segment_num()
            self.init_kv_cache_manager()
            self.prefill_reference_kv()
            segment_videos = []
            lazy_vae = self.config.get("lazy_load", False) or self.config.get("unload_modules", False)
            if lazy_vae:
                self.vae_decoder = self.load_vae_decoder()
            vae_decoder = AsyncVAEChunkDecoder.from_config(self.config, device=AI_DEVICE, vae_decoder=self.vae_decoder)

            with (
                no_sync_profiling(enabled=vae_decoder.is_async),
                ProfilingContext4DebugL1(
                    f"AR chunk total {self.video_segment_num} chunks",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["WanAudioARRunner"],
                ),
            ):
                try:
                    for segment_idx in range(self.video_segment_num):
                        logger.info(f"start chunk {segment_idx + 1}/{self.video_segment_num}")

                        with ProfilingContext4DebugL1(
                            f"chunk end2end {segment_idx + 1}/{self.video_segment_num}",
                            recorder_mode=GET_RECORDER_MODE(),
                            metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                            metrics_labels=["WanAudioARRunner"],
                        ):
                            self.check_stop()
                            self.init_run_segment(segment_idx)
                            segment_latents = self.run_segment(segment_idx)

                        vae_decoder.submit(self.decode_segment_latents, segment_idx, segment_latents)
                    segment_videos = vae_decoder.finish()
                finally:
                    if "vae_decoder" in locals():
                        vae_decoder.finish()
                    if lazy_vae:
                        del self.vae_decoder
                        torch.cuda.empty_cache()
                        gc.collect()

            self.gen_video = torch.cat(segment_videos, dim=2)
            self.check_stop()
            self.end_run_segment(0)
            result = self.process_images_after_vae_decoder()
            self.end_run()
            return result
        finally:
            if getattr(self, "va_controller", None) is not None:
                self.va_controller.clear()
                self.va_controller = None
